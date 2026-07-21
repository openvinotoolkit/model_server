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
#include <sstream>
#include <utility>
#include <filesystem>
#include <fstream>
#include <optional>

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/stringutils.hpp"
#include "embeddings_node_initializer_utils.hpp"
#include "embeddings_servable.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/embeddings/embeddings_calculator_ov.pb.h"
#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_istreamwrapper.hpp"
#include "src/port/rapidjson_error.hpp"

#include "src/logging.hpp"

namespace ovms {
namespace {

std::filesystem::path resolveModelsPath(const std::string& modelsPath, const std::string& basePath) {
    auto fsModelsPath = std::filesystem::path(modelsPath);
    if (fsModelsPath.is_relative()) {
        return std::filesystem::path(basePath) / fsModelsPath;
    }
    return fsModelsPath;
}

}  // namespace

std::optional<mediapipe::EmbeddingsCalculatorOVOptions_Pooling> detectEmbeddingsPoolingFromConfig(
    const std::filesystem::path& modelsPath) {
    const auto poolingConfigPath = modelsPath / "1_Pooling" / "config.json";
    if (!std::filesystem::exists(poolingConfigPath)) {
        return std::nullopt;
    }

    std::ifstream ifs(poolingConfigPath);
    if (!ifs.is_open()) {
        SPDLOG_WARN("Failed to open pooling config file: {}", poolingConfigPath.string());
        return std::nullopt;
    }

    rapidjson::Document poolingConfig;
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::ParseResult parseResult = poolingConfig.ParseStream(isw);
    if (parseResult.Code()) {
        SPDLOG_WARN("Failed to parse {}: {}",
            poolingConfigPath.string(),
            rapidjson::GetParseError_En(parseResult.Code()));
        return std::nullopt;
    }

    if (!poolingConfig.IsObject()) {
        SPDLOG_WARN("Invalid pooling config root in {}. Expected a JSON object.", poolingConfigPath.string());
        return std::nullopt;
    }

    const bool useLastToken = poolingConfig.HasMember("pooling_mode_lasttoken") &&
                              poolingConfig["pooling_mode_lasttoken"].IsBool() &&
                              poolingConfig["pooling_mode_lasttoken"].GetBool();
    const bool useMean = poolingConfig.HasMember("pooling_mode_mean_tokens") &&
                         poolingConfig["pooling_mode_mean_tokens"].IsBool() &&
                         poolingConfig["pooling_mode_mean_tokens"].GetBool();
    const bool useCls = poolingConfig.HasMember("pooling_mode_cls_token") &&
                        poolingConfig["pooling_mode_cls_token"].IsBool() &&
                        poolingConfig["pooling_mode_cls_token"].GetBool();
    const bool useMax = poolingConfig.HasMember("pooling_mode_max_tokens") &&
                        poolingConfig["pooling_mode_max_tokens"].IsBool() &&
                        poolingConfig["pooling_mode_max_tokens"].GetBool();
    const bool useMeanSqrtLen = poolingConfig.HasMember("pooling_mode_mean_sqrt_len_tokens") &&
                                poolingConfig["pooling_mode_mean_sqrt_len_tokens"].IsBool() &&
                                poolingConfig["pooling_mode_mean_sqrt_len_tokens"].GetBool();
    const bool useWeightedMean = poolingConfig.HasMember("pooling_mode_weightedmean_tokens") &&
                                 poolingConfig["pooling_mode_weightedmean_tokens"].IsBool() &&
                                 poolingConfig["pooling_mode_weightedmean_tokens"].GetBool();

    if (useMax || useMeanSqrtLen || useWeightedMean) {
        std::ostringstream unsupportedModes;
        bool first = true;
        auto appendMode = [&unsupportedModes, &first](const char* mode) {
            if (!first) {
                unsupportedModes << ", ";
            }
            unsupportedModes << mode;
            first = false;
        };
        if (useMax) {
            appendMode("pooling_mode_max_tokens");
        }
        if (useMeanSqrtLen) {
            appendMode("pooling_mode_mean_sqrt_len_tokens");
        }
        if (useWeightedMean) {
            appendMode("pooling_mode_weightedmean_tokens");
        }

        SPDLOG_WARN("Unsupported pooling mode(s) requested in {}: {}. OVMS currently supports only pooling_mode_cls_token, pooling_mode_mean_tokens and pooling_mode_lasttoken.",
            poolingConfigPath.string(),
            unsupportedModes.str());
        return std::nullopt;
    }

    const size_t enabledModes = static_cast<size_t>(useLastToken) +
                                static_cast<size_t>(useMean) +
                                static_cast<size_t>(useCls);
    if (enabledModes == 0) {
        return std::nullopt;
    }

    if (enabledModes > 1) {
        SPDLOG_WARN("Ambiguous pooling config in {}. Exactly one of pooling_mode_lasttoken, pooling_mode_mean_tokens, pooling_mode_cls_token should be true.",
            poolingConfigPath.string());
        return std::nullopt;
    }

    if (useLastToken) {
        return mediapipe::EmbeddingsCalculatorOVOptions_Pooling_LAST;
    }
    if (useMean) {
        return mediapipe::EmbeddingsCalculatorOVOptions_Pooling_MEAN;
    }
    return mediapipe::EmbeddingsCalculatorOVOptions_Pooling_CLS;
}

mediapipe::EmbeddingsCalculatorOVOptions_Pooling resolveEmbeddingsPooling(
    const std::filesystem::path& modelsPath,
    bool hasGraphPooling,
    mediapipe::EmbeddingsCalculatorOVOptions_Pooling graphPooling) {
    if (hasGraphPooling) {
        return graphPooling;
    }

    if (const auto detectedPooling = detectEmbeddingsPoolingFromConfig(modelsPath)) {
        SPDLOG_DEBUG("Detected pooling '{}' from {}",
            mediapipe::EmbeddingsCalculatorOVOptions_Pooling_Name(*detectedPooling),
            (modelsPath / "1_Pooling" / "config.json").string());
        return *detectedPooling;
    }

    return graphPooling;
}

class EmbeddingsNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "EmbeddingsCalculatorOV";

public:
    bool matches(const std::string& calculatorName) const override {
        return endsWith(calculatorName, CALCULATOR_NAME);
    }
    Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        PythonBackend* /*pythonBackend*/) override {
        auto& embeddingsServableMap = sidePackets.embeddingsServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("Embeddings node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("Embeddings node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (embeddingsServableMap.find(nodeName) != embeddingsServableMap.end()) {
            SPDLOG_ERROR("Embeddings node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        mediapipe::EmbeddingsCalculatorOVOptions nodeOptions;
        nodeConfig.node_options(0).UnpackTo(&nodeOptions);

        const auto modelsPath = resolveModelsPath(nodeOptions.models_path(), basePath);
        const auto pooling = resolveEmbeddingsPooling(modelsPath, nodeOptions.has_pooling(), nodeOptions.pooling());

        auto servable = std::make_shared<EmbeddingsServable>(
            nodeOptions.models_path(),
            nodeOptions.target_device(),
            nodeOptions.plugin_config(),
            basePath,
            pooling,
            nodeOptions.normalize_embeddings());
        servable->initialize(
            nodeOptions.models_path(),
            nodeOptions.target_device(),
            nodeOptions.plugin_config(),
            basePath);
        embeddingsServableMap.insert(std::pair<std::string, std::shared_ptr<EmbeddingsServable>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool embeddingsNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<EmbeddingsNodeInitializer>());
    return true;
}();
}  // namespace ovms
