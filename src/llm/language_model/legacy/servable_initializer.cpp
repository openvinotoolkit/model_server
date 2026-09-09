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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
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
#include "../../servable_initializer.hpp"

namespace ovms {
Status LegacyServableInitializer::initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    servable = std::make_shared<LegacyServable>();
    auto properties = std::static_pointer_cast<LegacyServableProperties>(servable->getProperties());

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

    properties->schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = nodeOptions.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    properties->device = nodeOptions.device();
    if (properties->device.empty()) {
        properties->device = recommendTargetDevice();
        SPDLOG_INFO("No device specified, using recommended device: {}", properties->device);
    }

    if (!nodeOptions.draft_models_path().empty()) {
        auto fsDraftModelsPath = std::filesystem::path(nodeOptions.draft_models_path());
        std::string draftPipelinePath = fsDraftModelsPath.is_relative()
                                            ? (std::filesystem::path(graphPath) / fsDraftModelsPath).string()
                                            : fsDraftModelsPath.string();
        try {
            const std::string draftDevice = nodeOptions.draft_device().empty() ? properties->device : nodeOptions.draft_device();
            auto draftPipeline = ov::genai::draft_model(draftPipelinePath, draftDevice);
            properties->pluginConfig.insert(draftPipeline);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error during draft model initialization for draft_models_path: {} exception: {}", draftPipelinePath, e.what());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        } catch (...) {
            SPDLOG_ERROR("Error during draft model initialization for draft_models_path: {}", draftPipelinePath);
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }
        try {
            properties->draftModelStrategy = detectDraftModelStrategy(draftPipelinePath);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Failed to detect draft model strategy for {}: {}", draftPipelinePath, e.what());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }
        using DS = GenAiServableProperties::DraftModelStrategy;
        switch (properties->draftModelStrategy) {
        case DS::EAGLE3:
            SPDLOG_INFO("Draft model strategy: EAGLE3");
            break;
        case DS::DFLASH:
            SPDLOG_INFO("Draft model strategy: DFlash");
            break;
        case DS::MTP:
            SPDLOG_INFO("Draft model strategy: MTP (Multi-Token Prediction)");
            break;
        case DS::FAST_DRAFT:
            SPDLOG_INFO("Draft model strategy: Fast Draft");
            break;
        }
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    applyGlobalCacheDir(properties);

    // Max prompt len is NPU specific property
    if (properties->device == "NPU") {
        auto it = properties->pluginConfig.find("MAX_PROMPT_LEN");
        if (it != properties->pluginConfig.end()) {
            try {
                properties->maxPromptLength = it->second.as<int64_t>();
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Error during MAX_PROMPT_LEN property read: {}", e.what());
                return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
            }
        }
    }

    // Enforce construction of stateful pipeline on any device selected (CPU and GPU by default construct CB pipeline through CB adapter)
    properties->pluginConfig["ATTENTION_BACKEND"] = "SDPA";
    try {
        properties->pipeline = std::make_shared<ov::genai::LLMPipeline>(parsedModelsPath, properties->device, properties->pluginConfig);
        properties->tokenizer = properties->pipeline->get_tokenizer();
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", parsedModelsPath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", parsedModelsPath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    loadChatTemplate(properties, parsedModelsPath);
    properties->legacyExecutor = std::make_shared<LegacyExecutorWrapper>(properties->pipeline);
    if (nodeOptions.has_max_tokens_limit()) {
        properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    }
    properties->bestOfLimit = nodeOptions.best_of_limit();
    properties->maxModelLength = parseMaxModelLength(parsedModelsPath);
    properties->enableToolGuidedGeneration = nodeOptions.enable_tool_guided_generation();

    return StatusCode::OK;
}

}  // namespace ovms
