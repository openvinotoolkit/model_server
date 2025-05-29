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

#include "../json_parser.hpp"
#include "../status.hpp"

#include "../filesystem.hpp"

namespace ovms {
EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties) {
    ov::Core core;
    std::shared_ptr<ov::Model> m_model = core.read_model(model_dir / "openvino_model.xml", {}, properties);
    compiledModel = core.compile_model(m_model, device, properties);
    uint32_t numberOfParallelInferRequests = 1;
    prepareInferenceRequestsQueue(numberOfParallelInferRequests);
}

ov::Tensor EmbeddingsModel::infer(ov::Tensor& inputsIds, ov::Tensor& attentionMask, ov::Tensor& typeIds) {
    ov::InferRequest m_request = this->compiledModel.create_infer_request();
    m_request.set_tensor("input_ids", inputsIds);
    m_request.set_tensor("attention_mask", attentionMask);
    // if(typeIds.get_size() > 0){
    //     m_request.set_tensor("token_type_ids", typeIds);
    // }
    for (auto& input : m_request.get_compiled_model().inputs()) {
        if (input.get_any_name() == "token_type_ids") {
            ov::Tensor token_type_ids{ov::element::i64, inputsIds.get_shape()};
            std::fill_n(token_type_ids.data<int64_t>(), attentionMask.get_size(), 0);
            m_request.set_tensor("token_type_ids", token_type_ids);
            break;
        }
    }
    m_request.start_async();
    m_request.wait();
    return m_request.get_tensor("token_embeddings");
}

void EmbeddingsModel::prepareInferenceRequestsQueue(const uint32_t& numberOfParallelInferRequests ) {
    // if (numberOfParallelInferRequests == 0) {
    //     return Status(StatusCode::INVALID_NIREQ, "Exceeded allowed nireq value");
    // }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(compiledModel, numberOfParallelInferRequests);
    //SET_IF_ENABLED(this->getMetricReporter().inferReqQueueSize, numberOfParallelInferRequests);
    // auto batchSize = getBatchSize();
    // SPDLOG_INFO("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
    //     getName(),
    //     getVersion(),
    //     batchSize.has_value() ? batchSize.value().toString() : std::string{"none"},
    //     numberOfParallelInferRequests);
}

EmbeddingsServable::EmbeddingsServable(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig) {
    mediapipe::EmbeddingsCalculatorOVOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    std::string model_dir = nodeOptions.models_path();
    std::string configPath = FileSystem::appendSlash(model_dir) + "config.json";
    SPDLOG_ERROR("1");
    if (std::filesystem::exists(configPath.c_str())) {
        SPDLOG_ERROR("2");
        std::ifstream ifs(configPath);
        if (ifs.is_open()) {
            SPDLOG_ERROR("3");    
            rapidjson::Document modelConfig;
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::ParseResult parseResult = modelConfig.ParseStream(isw);
            if (parseResult.Code()) {
                SPDLOG_ERROR("Parsing config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            } else {
                SPDLOG_ERROR("4");
                std::vector<std::string> maxLengthFields = {"max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"};
                for (auto field : maxLengthFields) {
                    SPDLOG_ERROR("5");
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
    std::string device = nodeOptions.device();
    ov::AnyMap embeddingsPoperties;
    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), embeddingsPoperties);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during embeddings node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
    }
    tokenizer = std::make_shared<ov::genai::Tokenizer>(std::filesystem::path(model_dir));
    embeddings = std::make_shared<EmbeddingsModel>(std::filesystem::path(model_dir), device, embeddingsPoperties);
}

ov::Tensor EmbeddingsServable::infer(std::vector<std::string>& prompts) {
    auto tokens = tokenizer->encode(prompts);
    ov::Tensor typeIds;
    return embeddings->infer(tokens.input_ids, tokens.attention_mask, typeIds);
}

ov::Tensor EmbeddingsServable::infer(std::vector<std::vector<int64_t>>& tokenized_documents) {
    auto received_batch_size = tokenized_documents.size();
    size_t tokens = 0;
    size_t token_count_of_longest_document = 0;
    for (const auto& document_tokens : tokenized_documents) {
        token_count_of_longest_document = std::max(token_count_of_longest_document, document_tokens.size());
        tokens += document_tokens.size();
    }
    auto inputsIds = ov::Tensor{
        ov::element::i64,
        ov::Shape{received_batch_size, token_count_of_longest_document}};
    auto attentionMask = ov::Tensor{
        ov::element::i64,
        ov::Shape{received_batch_size, token_count_of_longest_document}};
    try {
        for (size_t i = 0; i < received_batch_size; i++) {
            int64_t* input_ids_start = reinterpret_cast<int64_t*>(inputsIds.data()) + i * token_count_of_longest_document;
            std::fill(input_ids_start, input_ids_start + token_count_of_longest_document, pad_token);
            std::copy(tokenized_documents.at(i).data(), tokenized_documents.at(i).data() + tokenized_documents.at(i).size(), input_ids_start);

            int64_t* attention_mask_start = reinterpret_cast<int64_t*>(attentionMask.data()) + i * token_count_of_longest_document;
            std::fill(attention_mask_start, attention_mask_start + token_count_of_longest_document, 0);
            std::fill(attention_mask_start, attention_mask_start + tokenized_documents.at(i).size(), 1);
        }
    } catch (std::out_of_range& e) {
        SPDLOG_DEBUG("Caught exception from preparing embeddings inputs(): {}", e.what());
    } catch (std::exception& e) {
        SPDLOG_DEBUG("Caught generic exception from preparing embeddings inputs: {}", e.what());
    }
    //EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME ??
    auto typeIds = ov::Tensor{ov::element::i64, ov::Shape{received_batch_size, token_count_of_longest_document}};
    int64_t* token_type_ids_start = reinterpret_cast<int64_t*>(typeIds.data());
    std::fill(token_type_ids_start, token_type_ids_start + received_batch_size * token_count_of_longest_document, 1);
    return embeddings->infer(inputsIds, attentionMask, typeIds);
}

}  // namespace ovms