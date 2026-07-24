//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <memory>
#include <string>

#include <openvino/genai/omni/pipeline.hpp>
#include <openvino/genai/omni/talker.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../json_parser.hpp"
#include "../../../logging.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../../ov_utils.hpp"
#include "../../../status.hpp"
#include "../../io_processing/parser_config_validation.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"

namespace ovms {
Status OmniModelLegacyServableInitializer::initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    servable = std::make_shared<OmniModelLegacyServable>();
    auto properties = std::static_pointer_cast<OmniModelLegacyServableProperties>(servable->getProperties());

    properties->modelsPath = parsedModelsPath;
    std::filesystem::path modelGenerationConfigPath = std::filesystem::path(parsedModelsPath) / "generation_config.json";
    if (std::filesystem::exists(modelGenerationConfigPath)) {
        properties->baseGenerationConfig = ov::genai::GenerationConfig(modelGenerationConfigPath.string());
    }

    if (nodeOptions.has_tool_parser()) {
        properties->toolParserName = nodeOptions.tool_parser();
        if (!properties->toolParserName.empty() && !isSupportedToolParserName(properties->toolParserName)) {
            SPDLOG_ERROR("Unsupported tool_parser \"{}\" specified in graph configuration. Supported tool parsers are: {}",
                properties->toolParserName, getSupportedToolParserNamesAsString());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }
    }

    if (nodeOptions.has_reasoning_parser()) {
        properties->reasoningParserName = nodeOptions.reasoning_parser();
        if (!properties->reasoningParserName.empty() && !isSupportedReasoningParserName(properties->reasoningParserName)) {
            SPDLOG_ERROR("Unsupported reasoning_parser \"{}\" specified in graph configuration. Supported reasoning parsers are: {}",
                properties->reasoningParserName, getSupportedReasoningParserNamesAsString());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }
    }
    if (nodeOptions.has_chat_template_mode()) {
        properties->chatTemplateMode = (nodeOptions.chat_template_mode() == mediapipe::LLMCalculatorOptions::JINJA)
                                           ? ChatTemplateMode::JINJA
                                           : ChatTemplateMode::MINJA;
    }

    properties->device = nodeOptions.device();
    if (properties->device.empty()) {
        properties->device = recommendTargetDevice();
        SPDLOG_INFO("No device specified for Omni model, using recommended device: {}", properties->device);
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    try {
        auto vlm = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, properties->device, properties->pluginConfig);
        auto talker = std::make_shared<ov::genai::Talker>(parsedModelsPath, properties->device, properties->pluginConfig);
        properties->pipeline = std::make_shared<ov::genai::OmniPipeline>(vlm, talker);
        properties->tokenizer = ov::genai::Tokenizer(parsedModelsPath);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during omni model node initialization for models_path: {} exception: {}", parsedModelsPath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during omni model node initialization for models_path: {}", parsedModelsPath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    loadChatTemplate(properties, parsedModelsPath);
    properties->legacyExecutor = std::make_shared<OmniModelLegacyExecutorWrapper>(properties->pipeline);
    if (nodeOptions.has_max_tokens_limit()) {
        properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    }
    properties->bestOfLimit = nodeOptions.best_of_limit();
    properties->maxModelLength = parseMaxModelLength(parsedModelsPath);
    properties->enableToolGuidedGeneration = nodeOptions.enable_tool_guided_generation();
    return StatusCode::OK;
}

}  // namespace ovms
