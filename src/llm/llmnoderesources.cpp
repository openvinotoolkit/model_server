//*****************************************************************************
// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "llmnoderesources.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "../json_parser.hpp"
#include "../logging.hpp"
#include "../status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "src/llm/llm_executor.hpp"
#include "src/llm/text_processor.hpp"

namespace ovms {

static const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";

void LLMNodeResources::loadTextProcessor(LLMNodeResources& nodeResources, const std::string& chatTemplateDirectory) {
    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("templates_directory"_a = chatTemplateDirectory);
        py::exec(R"(
            # Following the logic from:
            # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/tokenization_utils_base.py#L1837

            global json
            import json
            from pathlib import Path

            global jinja2
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)


            # Default chat template accepts only single message and outputs only it's 'content'
            # effectively turning it into a regular prompt. 
            default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"

            bos_token = ""
            eos_token = ""
            chat_template = default_chat_template

            template = None

            # Try to read template from template.jinja file
            jinja_file = Path(templates_directory + "/template.jinja")
            if jinja_file.is_file():
                template_loader = jinja2.FileSystemLoader(searchpath=templates_directory)
                jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, loader=template_loader)
                jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                jinja_env.globals["raise_exception"] = raise_exception
                template = jinja_env.get_template("template.jinja")

            # Try to read data from tokenizer_config.json
            tokenizer_config_file = Path(templates_directory + "/tokenizer_config.json")
            if tokenizer_config_file.is_file():
                f = open(templates_directory + "/tokenizer_config.json")
                data = json.load(f)
                bos_token = data.get("bos_token", "")
                bos_token = bos_token if isinstance(bos_token, str) else ""  # tokenizer_config.json allows for different types than string
                eos_token = data.get("eos_token", "")
                eos_token = eos_token if isinstance(eos_token, str) else ""  # tokenizer_config.json allows for different types than string
                chat_template = data.get("chat_template", default_chat_template)

            if template is None:
                jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                jinja_env.globals["raise_exception"] = raise_exception
                template = jinja_env.from_string(chat_template)
        )",
            py::globals(), locals);

        nodeResources.textProcessor.bosToken = locals["bos_token"].cast<std::string>();
        nodeResources.textProcessor.eosToken = locals["eos_token"].cast<std::string>();
        nodeResources.textProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::cast_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::key_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (...) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with an unexpected error");
    }
}

Status LLMNodeResources::initializeLLMNodeResources(LLMNodeResources& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    auto fsModelsPath = std::filesystem::path(nodeOptions.models_path());

    std::string basePath;
    if (fsModelsPath.is_relative()) {
        basePath = (std::filesystem::path(graphPath) / fsModelsPath).string();
    } else {
        basePath = fsModelsPath.string();
    }

    nodeResources.modelsPath = basePath;
    if (basePath.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is empty. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::exists(basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} does not exist. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::is_directory(basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is not a directory. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }

    nodeResources.schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    nodeResources.schedulerConfig.cache_size = nodeOptions.cache_size();
    nodeResources.schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    nodeResources.schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    nodeResources.schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    nodeResources.device = nodeOptions.device();

    if (!nodeOptions.draft_models_path().empty()) {
        auto fsDraftModelsPath = std::filesystem::path(nodeOptions.draft_models_path());
        std::string draftModelsPath;
        if (fsDraftModelsPath.is_relative()) {
            draftModelsPath = (std::filesystem::path(graphPath) / fsDraftModelsPath).string();
        } else {
            draftModelsPath = fsDraftModelsPath.string();
        }
        auto draftSchedulerConfig = prepareDraftModelSchedulerConfig(nodeOptions);
        auto draftModelConfig = ov::genai::draft_model(draftModelsPath, nodeOptions.draft_device(),
            ov::genai::scheduler_config(draftSchedulerConfig));
        nodeResources.pluginConfig.insert(draftModelConfig);
        nodeResources.isSpeculativePipeline = true;
    }

    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), nodeResources.pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    try {
        plugin_config_t tokenizerPluginConfig = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
        nodeResources.initializeContinuousBatchingPipeline(basePath, nodeResources.schedulerConfig, nodeResources.device, nodeResources.pluginConfig, tokenizerPluginConfig);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", basePath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", basePath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    loadTextProcessor(nodeResources, nodeResources.modelsPath);

    nodeResources.maxTokensLimit = nodeOptions.max_tokens_limit();
    nodeResources.bestOfLimit = nodeOptions.best_of_limit();

    nodeResources.initiateGeneration();

    return StatusCode::OK;
}

void LLMNodeResources::initializeContinuousBatchingPipeline(
    const std::string& basePath,
    const ov::genai::SchedulerConfig& schedulerConfig,
    const std::string& device,
    const plugin_config_t& pluginConfig,
    const plugin_config_t& tokenizerPluginConfig) {
    this->cbPipe = std::make_unique<ov::genai::ContinuousBatchingPipeline>(basePath, schedulerConfig, device, pluginConfig, tokenizerPluginConfig);
}

void LLMNodeResources::initiateGeneration() {
    if (!cbPipe) {
        throw std::logic_error("Cannot initiate generation with uninitialized pipeline");
    }
    llmExecutorWrapper = std::make_unique<LLMExecutorWrapper>(cbPipe);
}

void LLMNodeResources::notifyExecutorThread() {
    OVMS_PROFILE_FUNCTION();
    llmExecutorWrapper->notifyNewRequestArrived();
}

std::unordered_map<std::string, std::string> LLMNodeResources::prepareLLMNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string basePath) {
    std::unordered_map<std::string, std::string> LLMArguments;
    return LLMArguments;
}

ov::genai::SchedulerConfig LLMNodeResources::prepareDraftModelSchedulerConfig(const mediapipe::LLMCalculatorOptions& nodeOptions) {
    ov::genai::SchedulerConfig config;
    config.max_num_batched_tokens = nodeOptions.has_draft_max_num_batched_tokens() ? nodeOptions.draft_max_num_batched_tokens() : nodeOptions.max_num_batched_tokens();
    config.cache_size = nodeOptions.has_draft_cache_size() ? nodeOptions.draft_cache_size() : nodeOptions.cache_size();
    config.dynamic_split_fuse = nodeOptions.has_draft_dynamic_split_fuse() ? nodeOptions.draft_dynamic_split_fuse() : nodeOptions.dynamic_split_fuse();
    config.max_num_seqs = nodeOptions.has_draft_max_num_seqs() ? nodeOptions.draft_max_num_seqs() : nodeOptions.max_num_seqs();
    config.enable_prefix_caching = nodeOptions.enable_prefix_caching();
    return config;
}

}  // namespace ovms
