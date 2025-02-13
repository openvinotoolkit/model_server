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
#include "node_resources.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "../../json_parser.hpp"
#include "../../logging.hpp"
#include "../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../mediapipe_internal/mediapipe_utils.hpp"
#include "src/llm/text_processor.hpp"
#include "node_resources_initializer.hpp"

namespace ovms {

static const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";

void ContinuousBatchingNodeResourcesInitializer::loadTextProcessor(std::shared_ptr<ContinuousBatchingNodeProperties>& properties, const std::string& chatTemplateDirectory) {
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

        properties->textProcessor.bosToken = locals["bos_token"].cast<std::string>();
        properties->textProcessor.eosToken = locals["eos_token"].cast<std::string>();
        properties->textProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
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

ov::genai::SchedulerConfig ContinuousBatchingNodeResourcesInitializer::prepareDraftPipelineSchedulerConfig(const mediapipe::LLMCalculatorOptions_PipelineConfig& draftPipelineConfig) {
    ov::genai::SchedulerConfig config;
    config.max_num_batched_tokens = draftPipelineConfig.max_num_batched_tokens();
    config.cache_size = draftPipelineConfig.cache_size();
    config.dynamic_split_fuse = draftPipelineConfig.dynamic_split_fuse();
    config.max_num_seqs = draftPipelineConfig.max_num_seqs();
    config.enable_prefix_caching = draftPipelineConfig.enable_prefix_caching();
    return config;
}

ov::genai::SchedulerConfig ContinuousBatchingNodeResourcesInitializer::prepareDraftPipelineSchedulerConfigLegacy(const mediapipe::LLMCalculatorOptions& nodeOptions) {
    ov::genai::SchedulerConfig config;
    config.max_num_batched_tokens = nodeOptions.has_draft_max_num_batched_tokens() ? nodeOptions.draft_max_num_batched_tokens() : nodeOptions.max_num_batched_tokens();
    config.cache_size = nodeOptions.has_draft_cache_size() ? nodeOptions.draft_cache_size() : nodeOptions.cache_size();
    config.dynamic_split_fuse = nodeOptions.has_draft_dynamic_split_fuse() ? nodeOptions.draft_dynamic_split_fuse() : nodeOptions.dynamic_split_fuse();
    config.max_num_seqs = nodeOptions.has_draft_max_num_seqs() ? nodeOptions.draft_max_num_seqs() : nodeOptions.max_num_seqs();
    config.enable_prefix_caching = nodeOptions.enable_prefix_caching();
    return config;
}

Status ContinuousBatchingNodeResourcesInitializer::initialize(std::shared_ptr<LLMNodeResources>& nodeResources, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    auto continousBatchingPipelineConfig = nodeOptions.continuous_batching_pipeline_config();
    auto mainPipelineConfig = continousBatchingPipelineConfig.main_pipeline_config();

    auto status = parseModelsPath(mainPipelineConfig.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    nodeResources = std::make_shared<ContinuousBatchingNodeResources>();
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(nodeResources->properties);
    properties->modelsPath = getBasePath();

    properties->schedulerConfig.max_num_batched_tokens = mainPipelineConfig.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = mainPipelineConfig.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = mainPipelineConfig.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = mainPipelineConfig.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = mainPipelineConfig.enable_prefix_caching();

    properties->device = mainPipelineConfig.device();

    // Speculative decoding enabled
    properties->isSpeculativePipeline = false;
    if (continousBatchingPipelineConfig.has_draft_pipeline_config()) {
        auto draftPipelineConfig = continousBatchingPipelineConfig.draft_pipeline_config();
        auto fsDraftModelsPath = std::filesystem::path(draftPipelineConfig.models_path());
        std::string draftPipelinePath;
        if (fsDraftModelsPath.is_relative()) {
            draftPipelinePath = (std::filesystem::path(graphPath) / fsDraftModelsPath).string();
        } else {
            draftPipelinePath = fsDraftModelsPath.string();
        }
        auto draftSchedulerConfig = prepareDraftPipelineSchedulerConfig(draftPipelineConfig);
        auto draftPipeline = ov::genai::draft_model(draftPipelinePath, draftPipelineConfig.device(), ov::genai::scheduler_config(draftSchedulerConfig));
        properties->pluginConfig.insert(draftPipeline);
        properties->isSpeculativePipeline = true;
    }
    
    status = JsonParser::parsePluginConfig(mainPipelineConfig.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", mainPipelineConfig.plugin_config());
        return status;
    }

    properties->tokenizerPluginConfig = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
    try {
        properties->pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(getBasePath(), 
            properties->schedulerConfig, properties->device, 
            properties->pluginConfig, properties->tokenizerPluginConfig);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", getBasePath(), e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", getBasePath());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    loadTextProcessor(properties, getBasePath());

    properties->maxTokensLimit = mainPipelineConfig.max_tokens_limit();
    properties->bestOfLimit = mainPipelineConfig.best_of_limit();

    return StatusCode::OK;
}

Status ContinuousBatchingNodeResourcesInitializer::initializeLegacy(std::shared_ptr<LLMNodeResources>& nodeResources, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    auto status = parseModelsPath(nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    nodeResources = std::make_shared<ContinuousBatchingNodeResources>();
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(nodeResources->properties);

    properties->modelsPath = getBasePath();

    properties->schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = nodeOptions.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    properties->device = nodeOptions.device();
    properties->isSpeculativePipeline = false;
    if (!nodeOptions.draft_models_path().empty()) {
        auto fsDraftModelsPath = std::filesystem::path(nodeOptions.draft_models_path());
        std::string draftPipelinePath;
        if (fsDraftModelsPath.is_relative()) {
            draftPipelinePath = (std::filesystem::path(graphPath) / fsDraftModelsPath).string();
        } else {
            draftPipelinePath = fsDraftModelsPath.string();
        }
        auto draftSchedulerConfig = prepareDraftPipelineSchedulerConfigLegacy(nodeOptions);
        auto draftPipeline = ov::genai::draft_model(draftPipelinePath, nodeOptions.draft_device(),
            ov::genai::scheduler_config(draftSchedulerConfig));
        properties->pluginConfig.insert(draftPipeline);
        properties->isSpeculativePipeline = true;
    } else if (nodeOptions.has_draft_max_num_batched_tokens() || nodeOptions.has_draft_cache_size() || nodeOptions.has_draft_dynamic_split_fuse() || nodeOptions.has_draft_max_num_seqs() || nodeOptions.has_draft_block_size() || nodeOptions.has_draft_device()) {
        // Consider moving draft parameters to separate structure in node options, so it's validated on the proto level
        SPDLOG_ERROR("Draft model path is not provided, but draft scheduler options are set.");
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    properties->tokenizerPluginConfig = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
    try {
        properties->pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(getBasePath(), 
            properties->schedulerConfig, properties->device, 
            properties->pluginConfig, properties->tokenizerPluginConfig);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", getBasePath(), e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", getBasePath());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    loadTextProcessor(properties, getBasePath());

    properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    properties->bestOfLimit = nodeOptions.best_of_limit();

    return StatusCode::OK;
}

}  // namespace ovms
