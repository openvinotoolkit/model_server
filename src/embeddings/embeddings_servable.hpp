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
#pragma once

#include "../sidepacket_servable.hpp"
#include "embeddings_api.hpp"
#include "../filesystem.hpp"
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace ovms {

struct EmbeddingsServable : SidepacketServable {
    PoolingMode poolingMode = PoolingMode::CLS;

public:
    EmbeddingsServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) :
        SidepacketServable(modelDir, targetDevice, pluginConfig, graphPath) {
        std::filesystem::path poolingConfigPath = (parsedModelsPath / "1_Pooling/config.json");
        if (std::filesystem::exists(poolingConfigPath)) {
            std::ifstream ifs(poolingConfigPath.string());
            if (ifs.is_open()) {
                rapidjson::Document poolingConfig;
                rapidjson::IStreamWrapper isw(ifs);
                rapidjson::ParseResult parseResult = poolingConfig.ParseStream(isw);
                if (parseResult.Code()) {
                    SPDLOG_ERROR("Parsing 1_Pooling/config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
                } else {
                    if (poolingConfig.HasMember("pooling_mode_lasttoken") && poolingConfig["pooling_mode_lasttoken"].IsBool() && poolingConfig["pooling_mode_lasttoken"].IsTrue()) {
                        SPDLOG_DEBUG("Embdeddings model pooling mode: LAST_TOKEN");
                        poolingMode = PoolingMode::LAST_TOKEN;
                    } else {
                        SPDLOG_DEBUG("Default pooling mode will be set: CLS");
                    }
                }
            }
        } else {
            SPDLOG_DEBUG("Pooling mode config file {} not provided. Default pooling mode will be set: CLS", poolingConfigPath.c_str());
        }
    }
    PoolingMode getPoolingMode() {
        return poolingMode;
    }
};

using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<EmbeddingsServable>>;
}  // namespace ovms
