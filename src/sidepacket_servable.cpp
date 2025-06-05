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

#include <numeric>

#include "sidepacket_servable.hpp"
#include <spdlog/spdlog.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

#include <vector>
#include <string>
#include <memory>

#include "json_parser.hpp"
#include "status.hpp"
#include "config.hpp"

#include "filesystem.hpp"

namespace ovms {
SidepacketServable::SidepacketServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(modelDir);
    std::filesystem::path parsedModelsPath;
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath.string();
    }
    std::filesystem::path configPath = (std::filesystem::path(graphPath) / fsModelsPath / "config.json");
    if (std::filesystem::exists(configPath)) {
        SPDLOG_ERROR("CONFIG {}", configPath.string());
        std::ifstream ifs(configPath.string());
        if (ifs.is_open()) {
            rapidjson::Document modelConfig;
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::ParseResult parseResult = modelConfig.ParseStream(isw);
            if (parseResult.Code()) {
                SPDLOG_ERROR("Parsing config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            } else {
                std::vector<std::string> maxLengthFields = {"max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"};
                for (auto field : maxLengthFields) {
                    if (modelConfig.HasMember(field.c_str()) && modelConfig[field.c_str()].IsUint()) {
                        maxModelLength = modelConfig[field.c_str()].GetUint();
                        break;
                    }
                }
                if (modelConfig.HasMember("pad_token_id") && modelConfig["pad_token_id"].IsInt64()) {
                    pad_token = modelConfig["pad_token_id"].GetInt64();
                }
                if (modelConfig.HasMember("eos_token_id") && modelConfig["eos_token_id"].IsInt64()) {
                    eos_token = modelConfig["eos_token_id"].GetInt64();
                }
                if (modelConfig.HasMember("bos_token_id") && modelConfig["bos_token_id"].IsInt64()) {
                    bos_token = modelConfig["bos_token_id"].GetInt64();
                }
                if (modelConfig.HasMember("sep_token_id") && modelConfig["sep_token_id"].IsInt64()) {
                    sep_token = modelConfig["sep_token_id"].GetInt64();
                } else {
                    sep_token = eos_token;
                }
            }
        }
    }

    ov::AnyMap properties;
    auto status = JsonParser::parsePluginConfig(pluginConfig, properties);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during embeddings node plugin_config option parsing to JSON: {}", pluginConfig);
    }
    tokenizer = std::make_shared<ov::genai::Tokenizer>(parsedModelsPath);

    ov::Core core;
    std::shared_ptr<ov::Model> m_model = core.read_model(parsedModelsPath / std::filesystem::path("openvino_model.xml"), {}, properties);
    compiledModel = core.compile_model(m_model, targetDevice, properties);
    auto& ovmsConfig = ovms::Config::instance();
    uint32_t numberOfParallelInferRequests = 1;
    if (ovmsConfig.nireq() > 0) {
        // nireq is set globally for all models in ovms startup parameters
        numberOfParallelInferRequests = ovmsConfig.nireq();
    }
    try {
        numberOfParallelInferRequests = compiledModel.get_property(ov::optimal_number_of_infer_requests);
    } catch (const ov::Exception& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(compiledModel, numberOfParallelInferRequests);
}

}  // namespace ovms
