// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_node_initializer_utils.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>

#include "src/embeddings/embeddings_calculator_ov.pb.h"
#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_istreamwrapper.hpp"
#include "src/port/rapidjson_error.hpp"
#include "src/logging.hpp"

namespace ovms {

std::optional<mediapipe::EmbeddingsCalculatorOVOptions_Pooling> detectEmbeddingsPoolingFromConfig(
    const std::filesystem::path& poolingConfigPath) {
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

    auto getBoolField = [&](const char* key) -> bool {
        return poolingConfig.HasMember(key) &&
               poolingConfig[key].IsBool() &&
               poolingConfig[key].GetBool();
    };

    static const char* kUnsupportedModes[] = {
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
    };

    for (const auto* mode : kUnsupportedModes) {
        if (getBoolField(mode)) {
            SPDLOG_WARN("Unsupported pooling mode '{}' requested in {}. OVMS currently supports only pooling_mode_cls_token, pooling_mode_mean_tokens and pooling_mode_lasttoken.",
                mode, poolingConfigPath.string());
            return std::nullopt;
        }
    }

    using Pooling = mediapipe::EmbeddingsCalculatorOVOptions_Pooling;
    static const std::pair<const char*, Pooling> kSupportedModes[] = {
        {"pooling_mode_cls_token",   mediapipe::EmbeddingsCalculatorOVOptions_Pooling_CLS},
        {"pooling_mode_mean_tokens", mediapipe::EmbeddingsCalculatorOVOptions_Pooling_MEAN},
        {"pooling_mode_lasttoken",   mediapipe::EmbeddingsCalculatorOVOptions_Pooling_LAST},
    };

    std::optional<Pooling> detected;
    for (const auto& [key, pooling] : kSupportedModes) {
        if (getBoolField(key)) {
            if (detected.has_value()) {
                SPDLOG_WARN("Ambiguous pooling config in {}. Exactly one of pooling_mode_lasttoken, pooling_mode_mean_tokens, pooling_mode_cls_token should be true.",
                    poolingConfigPath.string());
                return std::nullopt;
            }
            detected = pooling;
        }
    }

    return detected;
}

mediapipe::EmbeddingsCalculatorOVOptions_Pooling resolveEmbeddingsPooling(
    const std::filesystem::path& modelsPath,
    std::optional<mediapipe::EmbeddingsCalculatorOVOptions_Pooling> graphPooling) {
    if (graphPooling.has_value()) {
        return *graphPooling;
    }

    const auto poolingConfigPath = modelsPath / "1_Pooling" / "config.json";
    if (const auto detectedPooling = detectEmbeddingsPoolingFromConfig(poolingConfigPath)) {
        SPDLOG_INFO("Detected pooling '{}' from {}",
            mediapipe::EmbeddingsCalculatorOVOptions_Pooling_Name(*detectedPooling),
            poolingConfigPath.string());
        return *detectedPooling;
    }

    SPDLOG_WARN("Pooling mode was not specified and could not be inferred. Defaulting to CLS pooling.");
    return mediapipe::EmbeddingsCalculatorOVOptions_Pooling_CLS;
}

}  // namespace ovms
