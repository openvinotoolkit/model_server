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
#include <string>
#include <unordered_map>

#pragma warning(push)
#pragma warning(disable : 6001 6385 6386 6326 6011 4309 6246 4005 4456)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include <adapters/inference_adapter.h>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../precision.hpp"
#include "../profiler.hpp"
#include "../executingstreamidguard.hpp"
#include "../model_metric_reporter.hpp"
#include "embeddings_api.hpp"
#include "src/embeddings/embeddings_calculator_ov.pb.h"
#include "embeddings_servable.hpp"

using namespace rapidjson;
using namespace ovms;
class EmbeddingsServable;

namespace mediapipe {

const std::string EMBEDDINGS_SESSION_SIDE_PACKET_TAG = "EMBEDDINGS_NODE_RESOURCES";

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

class EmbeddingsCalculatorOV : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string EMBEDDINGS_MODEL_INPUT_IDS_NAME;
    static const std::string EMBEDDINGS_MODEL_ATTENTION_MASK_NAME;
    static const std::string EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME;

    mediapipe::Timestamp timestamp{0};

protected:
    std::shared_ptr<ovms::EmbeddingsServable> embeddings_session{nullptr};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->InputSidePackets().Tag(EMBEDDINGS_SESSION_SIDE_PACKET_TAG).Set<ovms::EmbeddingsServableMap>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculatorOV [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculatorOV  [Node: {}] Open start", cc->NodeName());
        auto servableMap = cc->InputSidePackets()
                               .Tag(EMBEDDINGS_SESSION_SIDE_PACKET_TAG)
                               .Get<ovms::EmbeddingsServableMap>();
        auto it = servableMap.find(cc->NodeName());
        RET_CHECK(it != servableMap.end()) << "Could not find initialized Embeddings node named: " << cc->NodeName();
        embeddings_session = it->second;
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculatorOV [Node: {}] Open end", cc->NodeName());

        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        RET_CHECK(embeddings_session != nullptr);
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::InvalidArgumentError("Input is empty");
        }
        InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request body: {}", payload.body);
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request uri: {}", payload.uri);
        ovms::EmbeddingsHandler handler(*payload.parsedJson);
        auto parseRequestStartTime = std::chrono::high_resolution_clock::now();
        absl::Status status = handler.parseRequest();
        if (!status.ok()) {
            return status;
        }
        double time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseRequestStartTime).count();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings request deserialization time: {} ms", time / 1000);

        ov::Tensor embeddingsTensor;
        size_t received_batch_size = 1;
        size_t max_context_length = 1024;  // default allowed input length. Otherwise, it will be read from model config.json file
        ModelMetricReporter unused(nullptr, nullptr, "unused", 1);
        ov::genai::TokenizedInputs tokens;
        ov::Tensor typeIds;
        if (embeddings_session->getMaxModelLength().has_value()) {
            max_context_length = embeddings_session->getMaxModelLength().value();
        } else {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "max_position_embeddings nor max_trained_positions included in config.json. Using default value {}", max_context_length);
        }
        try {
            auto input = handler.getInput();
            if (auto strings = std::get_if<std::vector<std::string>>(&input)) {
                received_batch_size = strings->size();
                ov::AnyMap params = {};
                if (cc->Options<EmbeddingsCalculatorOVOptions>().truncate()) {
                    params = {{"max_length", max_context_length}};
                }
                tokens = embeddings_session->getTokenizer().encode(*strings, params);
                RET_CHECK(tokens.input_ids.get_shape().size() == 2);
                size_t input_ids_size = tokens.input_ids.get_shape()[1];
                if (input_ids_size > max_context_length) {
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Input size {} exceeds max_context_length {}", input_ids_size, max_context_length);
                    return absl::InvalidArgumentError(absl::StrCat("Input length ", input_ids_size, " longer than allowed ", max_context_length));
                }
                if (embeddings_session->getNumberOfModelInputs() == 3) {
                    typeIds = ov::Tensor{ov::element::i64, tokens.input_ids.get_shape()};
                    std::fill_n(typeIds.data<int64_t>(), tokens.input_ids.get_size(), 0);
                }
                size_t attendedTokens = 0;
                if (tokens.attention_mask.get_element_type() == ov::element::Type_t::i64) {
                    for (int i = 0; i < tokens.attention_mask.get_size(); i++) {
                        attendedTokens += reinterpret_cast<int64_t*>(tokens.attention_mask.data())[i];
                    }
                } else if (tokens.attention_mask.get_element_type() == ov::element::Type_t::i32) {
                    for (int i = 0; i < tokens.attention_mask.get_size(); i++) {
                        attendedTokens += reinterpret_cast<int32_t*>(tokens.attention_mask.data())[i];
                    }
                } else if (tokens.attention_mask.get_element_type() == ov::element::Type_t::i8) {
                    for (int i = 0; i < tokens.attention_mask.get_byte_size(); i++) {
                        attendedTokens += reinterpret_cast<uint8_t*>(tokens.attention_mask.data())[i];
                    }
                } else {
                    return absl::InternalError("Attention mask element type invalid.");
                }
                handler.setPromptTokensUsage(attendedTokens);
            } else if (auto tokenized_documents = std::get_if<std::vector<std::vector<int64_t>>>(&input)) {
                received_batch_size = tokenized_documents->size();
                size_t numberOfTokens = 0;
                size_t token_count_of_longest_document = 0;
                for (const auto& document_tokens : *tokenized_documents) {
                    token_count_of_longest_document = std::max(token_count_of_longest_document, document_tokens.size());
                    numberOfTokens += document_tokens.size();
                }
                handler.setPromptTokensUsage(numberOfTokens);
                tokens.input_ids = ov::Tensor{
                    ov::element::i64,
                    ov::Shape{received_batch_size, token_count_of_longest_document}};
                size_t input_ids_size = tokens.input_ids.get_shape()[1];
                if (input_ids_size > max_context_length) {
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Input size {} exceeds max_context_length {}", input_ids_size, max_context_length);
                    return absl::InvalidArgumentError(absl::StrCat("Input length ", input_ids_size, " longer than allowed ", max_context_length));
                }
                tokens.attention_mask = ov::Tensor{
                    ov::element::i64,
                    ov::Shape{received_batch_size, token_count_of_longest_document}};
                try {
                    for (size_t i = 0; i < received_batch_size; i++) {
                        int64_t* input_ids_start = reinterpret_cast<int64_t*>(tokens.input_ids.data()) + i * token_count_of_longest_document;
                        std::fill(input_ids_start, input_ids_start + token_count_of_longest_document, embeddings_session->getPadToken().value_or(0));
                        std::copy(tokenized_documents->at(i).data(), tokenized_documents->at(i).data() + tokenized_documents->at(i).size(), input_ids_start);

                        int64_t* attention_mask_start = reinterpret_cast<int64_t*>(tokens.attention_mask.data()) + i * token_count_of_longest_document;
                        std::fill(attention_mask_start, attention_mask_start + token_count_of_longest_document, 0);
                        std::fill(attention_mask_start, attention_mask_start + tokenized_documents->at(i).size(), 1);
                    }
                } catch (std::out_of_range& e) {
                    SPDLOG_DEBUG("Caught exception from preparing embeddings inputs(): {}", e.what());
                } catch (std::exception& e) {
                    SPDLOG_DEBUG("Caught generic exception from preparing embeddings inputs: {}", e.what());
                }
                if (embeddings_session->getNumberOfModelInputs() == 3) {
                    typeIds = ov::Tensor{ov::element::i64, ov::Shape{received_batch_size, token_count_of_longest_document}};
                    int64_t* token_type_ids_start = reinterpret_cast<int64_t*>(typeIds.data());
                    std::fill(token_type_ids_start, token_type_ids_start + received_batch_size * token_count_of_longest_document, 1);
                }
            } else {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings input is of not supported type");
                return absl::InvalidArgumentError("Input should be string, array of strings or array of integers");
            }
            auto executingStreamIdGuard = std::make_unique<ExecutingStreamIdGuard>(embeddings_session->getInferRequestsQueue(), unused);
            ov::InferRequest& inferRequest = executingStreamIdGuard->getInferRequest();
            inferRequest.set_tensor(EMBEDDINGS_MODEL_INPUT_IDS_NAME, tokens.input_ids);
            inferRequest.set_tensor(EMBEDDINGS_MODEL_ATTENTION_MASK_NAME, tokens.attention_mask);
            if (embeddings_session->getNumberOfModelInputs() == 3) {
                inferRequest.set_tensor(EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME, typeIds);
            }
            inferRequest.start_async();
            inferRequest.wait();
            std::string outputTensorName;
            if (inferRequest.get_compiled_model().outputs().size() == 2) {  // GTE
                // Search by number of dimensions, should be 3
                bool found = false;
                for (const auto& output : inferRequest.get_compiled_model().outputs()) {
                    if (output.get_partial_shape().size() == 3) {
                        outputTensorName = output.get_any_name();
                        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Multiple embedding model outputs found, 3-dim output with name {} will be used", outputTensorName);
                        found = true;
                        break;
                    }
                }
                RET_CHECK(found);
            } else {  // BGE
                RET_CHECK(inferRequest.get_compiled_model().outputs().size() == 1);
                outputTensorName = inferRequest.get_compiled_model().outputs().begin()->get_any_name();
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Single embedding model output found with name {}", outputTensorName);
            }
            embeddingsTensor = inferRequest.get_tensor(outputTensorName.c_str());
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught exception from session infer(): {}", e.what());
            LOG(INFO) << e.what();
            RET_CHECK(false);
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught unknown exception from session infer()");
            RET_CHECK(false);
        }

        RET_CHECK(embeddingsTensor.get_shape().size() == 3);
        RET_CHECK(embeddingsTensor.get_shape()[0] == received_batch_size);
        RET_CHECK(embeddingsTensor.get_element_type() == ov::element::f32);

        auto parseResponseStartTime = std::chrono::high_resolution_clock::now();
        StringBuffer buffer;
        PoolingMode mode;
        if (cc->Options<EmbeddingsCalculatorOVOptions>().pooling() == mediapipe::EmbeddingsCalculatorOVOptions::LAST) {
            mode = PoolingMode::LAST;
        } else {
            mode = PoolingMode::CLS;
        }
        status = handler.parseResponse(buffer, embeddingsTensor, cc->Options<EmbeddingsCalculatorOVOptions>().normalize_embeddings(), mode, tokens.attention_mask);
        if (!status.ok()) {
            return status;
        }
        time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseResponseStartTime).count();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings response deserialization time: {} ms", time / 1000);
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string(buffer.GetString()), timestamp);
        return absl::OkStatus();
    }
};
const std::string EmbeddingsCalculatorOV::INPUT_TAG_NAME{"REQUEST_PAYLOAD"};
const std::string EmbeddingsCalculatorOV::OUTPUT_TAG_NAME{"RESPONSE_PAYLOAD"};
const std::string EmbeddingsCalculatorOV::EMBEDDINGS_MODEL_INPUT_IDS_NAME{"input_ids"};
const std::string EmbeddingsCalculatorOV::EMBEDDINGS_MODEL_ATTENTION_MASK_NAME{"attention_mask"};
const std::string EmbeddingsCalculatorOV::EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME{"token_type_ids"};

REGISTER_CALCULATOR(EmbeddingsCalculatorOV);

}  // namespace mediapipe
