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

#include "embeddings_servable.hpp"
#include <spdlog/spdlog.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

#include <vector>
#include <string>
#include <memory>

#include "../json_parser.hpp"
#include "../status.hpp"
#include "../config.hpp"

#include "../filesystem.hpp"

namespace ovms {
EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& model_dir,
    const std::string& target_device,
    const ov::AnyMap& properties) {
    ov::Core core;
    std::shared_ptr<ov::Model> m_model = core.read_model(model_dir / std::filesystem::path("openvino_model.xml"), {}, properties);
    compiledModel = core.compile_model(m_model, target_device, properties);
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
    prepareInferenceRequestsQueue(numberOfParallelInferRequests);
}

void EmbeddingsModel::prepareInferenceRequestsQueue(const uint32_t& numberOfParallelInferRequests) {
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(compiledModel, numberOfParallelInferRequests);
}

EmbeddingsServable::EmbeddingsServable(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::EmbeddingsCalculatorOVOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    std::string model_dir = nodeOptions.models_path();
    std::string configPath = FileSystem::appendSlash(model_dir) + "config.json";
    if (std::filesystem::exists(configPath.c_str())) {
        std::ifstream ifs(configPath);
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
            }
        }
    }
    std::string target_device = nodeOptions.target_device();
    ov::AnyMap embeddingsPoperties;
    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), embeddingsPoperties);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during embeddings node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
    }
    auto fsModelsPath = std::filesystem::path(model_dir);
    std::filesystem::path parsedModelsPath;
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath.string();
    }
    tokenizer = std::make_shared<ov::genai::Tokenizer>(parsedModelsPath);
    embeddings = std::make_shared<EmbeddingsModel>(parsedModelsPath.string(), target_device, embeddingsPoperties);
}

}  // namespace ovms
