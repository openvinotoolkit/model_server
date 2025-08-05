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

#define SET_TOKEN_ID(token, token_id_name)                                                 \
    if (modelConfig.HasMember(token_id_name) && modelConfig[token_id_name].IsInt64() && modelConfig[token_id_name].GetInt64() != 0) { \
        token = modelConfig[token_id_name].GetInt64();                                  \
    }

#define SET_TOKEN(token)\
    if(!token.has_value()){\
        if(tokenizerConfig.HasMember(#token) && tokenizerConfig[#token].IsString()){\
            auto tokenizedInputs = tokenizer->encode(tokenizerConfig[#token].GetString());\
            token = reinterpret_cast<int64_t*>(tokenizedInputs.input_ids.data())[0]; \
        }\
    }


SidepacketServable::SidepacketServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(modelDir);
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath.string();
    }
    std::filesystem::path configPath = (parsedModelsPath / "config.json");
    if (std::filesystem::exists(configPath)) {
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
                SET_TOKEN_ID(pad_token, "pad_token_id");
                SET_TOKEN_ID(eos_token, "eos_token_id");
                SET_TOKEN_ID(bos_token, "bos_token_id");
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
    std::filesystem::path tokenizerConfigPath = (std::filesystem::path(graphPath) / fsModelsPath / "tokenizer_config.json");
    if (std::filesystem::exists(tokenizerConfigPath)) {
        std::ifstream ifs(tokenizerConfigPath.string());
        if (ifs.is_open()) {
            rapidjson::Document tokenizerConfig;
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::ParseResult parseResult = tokenizerConfig.ParseStream(isw);
            if (parseResult.Code()) {
                SPDLOG_ERROR("Parsing config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            } else {
                SET_TOKEN(pad_token);
                SET_TOKEN(eos_token);
                SET_TOKEN(bos_token);
                if(!sep_token.has_value()){
                    if(tokenizerConfig.HasMember("sep_token") && tokenizerConfig["sep_token"].IsString()){
                        auto tokenizedInputs = tokenizer->encode(tokenizerConfig["sep_token"].GetString());
                        sep_token = reinterpret_cast<int64_t*>(tokenizedInputs.input_ids.data())[0];
                    }
                    else{
                        sep_token = eos_token;
                    }
                }
            }
        }
    }

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
