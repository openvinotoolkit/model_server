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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../json_parser.hpp"
#include "../../../logging.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../../ov_utils.hpp"
#include "../../../status.hpp"
#include "../../io_processing/parser_config_validation.hpp"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "servable.hpp"
#include "servable_initializer.hpp"

namespace ovms {
namespace {
// PEFT scales the LoRA update by lora_alpha / r (lora_alpha / sqrt(r) with rsLoRA), but GenAI only applies the
// alpha passed in AdapterConfig. Read the scale from adapter_config.json so request weight 1.0 matches PEFT.
// Unsupported PEFT features are rejected even when the scale is not used.
Status readLoraConfig(const std::filesystem::path& adapterPath, const std::string& alias, bool scaleFromConfig, float& scale) {
    scale = 1.0f;
    const auto configPath = adapterPath.parent_path() / "adapter_config.json";
    if (!std::filesystem::exists(configPath)) {
        SPDLOG_WARN("LoRA adapter {}: {} not found, using scale 1.0 and skipping the PEFT feature check", alias, configPath.string());
        return StatusCode::OK;
    }
    std::ifstream ifs(configPath);
    if (!ifs.is_open()) {
        SPDLOG_ERROR("LoRA adapter {}: cannot open {}", alias, configPath.string());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    rapidjson::Document config;
    rapidjson::IStreamWrapper isw(ifs);
    if (config.ParseStream(isw).HasParseError() || !config.IsObject()) {
        SPDLOG_ERROR("LoRA adapter {}: failed to parse {}", alias, configPath.string());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    auto isSet = [&config](const char* name) {
        auto it = config.FindMember(name);
        if (it == config.MemberEnd() || it->value.IsNull()) {
            return false;
        }
        if (it->value.IsBool()) {
            return it->value.GetBool();
        }
        if (it->value.IsObject()) {
            return it->value.MemberCount() > 0;
        }
        if (it->value.IsArray()) {
            return !it->value.Empty();
        }
        return true;
    };
    // GenAI applies one alpha per adapter and only LoRA A/B weights, so these PEFT features would be silently wrong.
    for (const char* unsupported : {"use_dora", "rank_pattern", "alpha_pattern", "modules_to_save"}) {
        if (isSet(unsupported)) {
            SPDLOG_ERROR("LoRA adapter {}: {} in {} is not supported", alias, unsupported, configPath.string());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }
    }
    if (!scaleFromConfig) {
        return StatusCode::OK;
    }
    auto rank = config.FindMember("r");
    auto loraAlpha = config.FindMember("lora_alpha");
    if (rank == config.MemberEnd() || !rank->value.IsNumber() || rank->value.GetDouble() <= 0 ||
        loraAlpha == config.MemberEnd() || !loraAlpha->value.IsNumber()) {
        SPDLOG_ERROR("LoRA adapter {}: {} must contain a positive r and a numeric lora_alpha", alias, configPath.string());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    const double r = rank->value.GetDouble();
    const double alpha = loraAlpha->value.GetDouble();
    const double peftScale = isSet("use_rslora") ? alpha / std::sqrt(r) : alpha / r;
    if (!std::isfinite(peftScale) || std::fabs(peftScale) > std::numeric_limits<float>::max()) {
        SPDLOG_ERROR("LoRA adapter {}: scale lora_alpha / r from {} is out of float range", alias, configPath.string());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    scale = static_cast<float>(peftScale);
    return StatusCode::OK;
}

// GenAI silently drops tensors whose names do not match its LoRA patterns (default_lora_patterns() in
// genai lora/adapter.cpp, mirrored here), so an adapter saved with e.g. "*.lora_A.default" would be a no-op.
Status checkLoraTensorNames(const std::filesystem::path& adapterPath, const std::string& alias) {
    std::ifstream ifs(adapterPath, std::ios::binary);
    uint64_t headerSize = 0;
    std::string header;
    rapidjson::Document tensors;
    if (ifs.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize)) && headerSize > 0 && headerSize <= (100ull << 20)) {
        header.resize(headerSize);
        ifs.read(header.data(), headerSize);
    }
    if (header.empty() || !ifs || tensors.Parse(header.data(), header.size()).HasParseError() || !tensors.IsObject()) {
        SPDLOG_ERROR("LoRA adapter {}: {} is not a valid safetensors file", alias, adapterPath.string());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    static const std::regex loraA("((.*)[_.](lora[_.](A|down)\\.weight))|((.*lora[12])\\.(down\\.weight))");
    static const std::regex loraB("((.*)[_.](lora[_.](B|up)\\.weight))|((.*lora[12])\\.(up\\.weight))");
    static const std::regex otherKnown("(.*)\\.(alpha|lm_head\\.weight|embed_tokens\\.weight)");
    size_t loraTensors = 0;
    std::vector<std::string> ignored;
    for (const auto& member : tensors.GetObject()) {
        const std::string name = member.name.GetString();
        if (name == "__metadata__") {
            continue;
        }
        if (std::regex_match(name, loraA) || std::regex_match(name, loraB)) {
            ++loraTensors;
        } else if (!std::regex_match(name, otherKnown)) {
            ignored.push_back(name);
        }
    }
    if (loraTensors == 0) {
        SPDLOG_ERROR("LoRA adapter {}: no tensor in {} is named like *.lora_A.weight / *.lora_B.weight, so GenAI would ignore the whole adapter (e.g. {})",
            alias, adapterPath.string(), ignored.empty() ? std::string("<no tensors>") : ignored.front());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    if (!ignored.empty()) {
        SPDLOG_WARN("LoRA adapter {}: {} tensors in {} have names GenAI does not recognize and will be ignored (e.g. {})",
            alias, ignored.size(), adapterPath.string(), ignored.front());
    }
    return StatusCode::OK;
}
}  // namespace

Status VisualLanguageModelLegacyServableInitializer::initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    servable = std::make_shared<VisualLanguageModelLegacyServable>();
    auto properties = std::static_pointer_cast<VisualLanguageModelLegacyServableProperties>(servable->getProperties());

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
    try {
        properties->chatTemplateEngine = determineChatTemplateEngine(nodeOptions);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to determine chat template engine: {}", e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    properties->schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = nodeOptions.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    properties->device = nodeOptions.device();
    if (properties->device.empty()) {
        properties->device = recommendTargetDevice();
        SPDLOG_INFO("No device specified for VLM model, using recommended device: {}", properties->device);
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    applyGlobalCacheDir(properties);

    try {
        ov::AnyMap pipelineConfig = properties->pluginConfig;
        if (nodeOptions.lora_adapters_size() > 0) {
            ov::genai::AdapterConfig adapterConfig(ov::genai::AdapterConfig::MODE_DYNAMIC);
            for (const auto& entry : nodeOptions.lora_adapters()) {
                if (entry.alias().empty()) {
                    SPDLOG_ERROR("LoRA adapter alias must not be empty (path: {})", entry.path());
                    return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
                }
                std::filesystem::path adapterPath(entry.path());
                if (adapterPath.is_relative()) {
                    adapterPath = std::filesystem::path(graphPath) / adapterPath;
                }
                if (!std::filesystem::is_regular_file(adapterPath)) {
                    SPDLOG_ERROR("LoRA adapter {}: file {} does not exist", entry.alias(), adapterPath.string());
                    return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
                }
                status = checkLoraTensorNames(adapterPath, entry.alias());
                if (!status.ok()) {
                    return status;
                }
                float scale = 1.0f;
                status = readLoraConfig(adapterPath, entry.alias(), nodeOptions.lora_scale_from_config(), scale);
                if (!status.ok()) {
                    return status;
                }
                ov::genai::Adapter adapter(adapterPath);
                if (!properties->loraAdapters.emplace(entry.alias(), LoraAdapterEntry{adapter, scale}).second) {
                    SPDLOG_ERROR("Duplicate LoRA adapter alias: {}", entry.alias());
                    return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
                }
                properties->loraAdapterOrder.push_back(entry.alias());
                // Registered with alpha 0; each request sets its own adapters and alphas.
                adapterConfig.add(adapter, 0.0f);
                SPDLOG_INFO("Registered LoRA adapter: {} -> {} (scale {})", entry.alias(), adapterPath.string(), scale);
            }
            pipelineConfig.insert(ov::genai::adapters(adapterConfig));

            // Prefix caching with adapters needs an OpenVINO GenAI build that adds the request adapters to prefix cache
            // keys; older builds can reuse KV computed with a different adapter. It stays off unless the graph enables it.
            // SDPA resets KV for every request, so only PA gets a scheduler_config.
            const auto backend = properties->pluginConfig.find("ATTENTION_BACKEND");
            const bool sdpaBackend = backend != properties->pluginConfig.end() && backend->second.as<std::string>() == "SDPA";
            if (sdpaBackend) {
                if (nodeOptions.enable_prefix_caching()) {
                    SPDLOG_WARN("enable_prefix_caching is ignored because ATTENTION_BACKEND is SDPA; SDPA resets KV for every request");
                }
            } else {
                ov::genai::SchedulerConfig schedulerConfig = properties->schedulerConfig;
                // Keep GenAI's latency-oriented default of not splitting the prompt unless the graph sets a limit.
                if (!nodeOptions.has_max_num_batched_tokens()) {
                    schedulerConfig.max_num_batched_tokens = std::numeric_limits<size_t>::max();
                }
                if (schedulerConfig.enable_prefix_caching) {
                    SPDLOG_WARN("Prefix caching is enabled with LoRA adapters. Results are correct only if OpenVINO GenAI adds the request adapters to prefix cache keys; older GenAI builds can reuse KV computed with a different adapter or weight.");
                } else {
                    SPDLOG_INFO("Prefix caching is disabled because lora_adapters are configured; multi-turn requests do not reuse KV. Set enable_prefix_caching: true if OpenVINO GenAI adds the request adapters to prefix cache keys.");
                }
                pipelineConfig.insert(ov::genai::scheduler_config(schedulerConfig));
            }
        }
        try {
            properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, properties->device, pipelineConfig);
        } catch (const std::exception& e) {
            // An explicit scheduler_config disables GenAI's own fallback to SDPA, so retry it here. Like GenAI, keep
            // the error when the graph selects the backend explicitly.
            if (pipelineConfig.count(ov::genai::scheduler_config.name()) == 0 || properties->pluginConfig.count("ATTENTION_BACKEND") > 0) {
                throw;
            }
            SPDLOG_WARN("Paged attention initialization with LoRA adapters failed, falling back to SDPA: {}", e.what());
            pipelineConfig.erase(ov::genai::scheduler_config.name());
            pipelineConfig["ATTENTION_BACKEND"] = std::string("SDPA");
            properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, properties->device, pipelineConfig);
        }
        properties->tokenizer = properties->pipeline->get_tokenizer();
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", parsedModelsPath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", parsedModelsPath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    status = loadChatTemplate(properties, parsedModelsPath);
    if (!status.ok()) {
        return status;
    }
    const auto chatTemplateAnalysis = properties->chatTemplateStatus == ChatTemplateStatus::AVAILABLE ? std::optional<ChatTemplateAnalysisResult>{analyzeChatTemplate(properties)} : std::nullopt;
    selectOutputParsers(properties, chatTemplateAnalysis);
    if (nodeOptions.has_chat_template_engine()) {
        status = validateChatTemplateAvailability(properties);
        if (!status.ok()) {
            return status;
        }
    }
    properties->legacyExecutor = std::make_shared<VisualLanguageModelLegacyExecutorWrapper>(properties->pipeline);
    if (nodeOptions.has_max_tokens_limit()) {
        properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    }
    properties->bestOfLimit = nodeOptions.best_of_limit();
    properties->maxModelLength = parseMaxModelLength(parsedModelsPath);
    properties->enableToolGuidedGeneration = nodeOptions.enable_tool_guided_generation();
    return StatusCode::OK;
}

}  // namespace ovms
