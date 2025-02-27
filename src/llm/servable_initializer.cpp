//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../status.hpp"
#include "continuous_batching/servable_initializer.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"

namespace ovms {

static const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";

void GenAiServableInitializer::loadTextProcessor(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory) {
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

Status GenAiServableInitializer::parseModelsPath(std::string modelsPath, std::string graphPath) {
    auto fsModelsPath = std::filesystem::path(modelsPath);
    if (fsModelsPath.is_relative()) {
        this->basePath = (std::filesystem::path(graphPath) / fsModelsPath).string();
    } else {
        this->basePath = fsModelsPath.string();
    }

    if (this->basePath.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is empty. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::exists(this->basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} does not exist. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::is_directory(this->basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is not a directory. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }

    return StatusCode::OK;
}

Status initializeGenAiServable(std::shared_ptr<GenAiServable>& servable, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    Status status;
    if (nodeOptions.has_models_path()) {  // Stable initialization
        if (nodeOptions.pipeline_type() == mediapipe::LLMCalculatorOptions::CONTINUOUS_BATCHING) {
            ContinuousBatchingServableInitializer cbServableInitializer;
            status = cbServableInitializer.initialize(servable, nodeOptions, graphPath);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
                return status;
            }
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node options do not contain any recognized pipeline configuration.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else {
        if (nodeOptions.has_continuous_batching_pipeline_config()) {  // Experimental initialization
            ContinuousBatchingServableInitializer cbServableInitializer;
            status = cbServableInitializer.initializeExperimental(servable, nodeOptions, graphPath);
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node options do not contain any recognized pipeline configuration.");
            return StatusCode::INTERNAL_ERROR;
        }

        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
            return status;
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms
