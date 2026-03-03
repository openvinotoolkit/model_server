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

#include <openvino/openvino.hpp>
#include <adapters/inference_adapter.h>
#include "src/port/rapidjson_writer.hpp"

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

    absl::Status tokenizeStrings(ov::genai::Tokenizer& tokenizer, const std::vector<std::string>& inputStrings, const ov::AnyMap& parameters, ov::genai::TokenizedInputs& tokens) {
        tokens = tokenizer.encode(inputStrings, parameters);
        RET_CHECK(tokens.input_ids.get_shape().size() == 2);

        return absl::OkStatus();
    }

    absl::Status isInputIdSizeOk(size_t inputIdsSize, size_t maxContextLength) {
        if (inputIdsSize > maxContextLength) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Input size {} exceeds maxContextLength {}", inputIdsSize, maxContextLength);
            return absl::InvalidArgumentError(absl::StrCat("Input length ", inputIdsSize, " longer than allowed ", maxContextLength));
        }
        return absl::OkStatus();
    }

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

        ov::Tensor embeddingsTensor;
        size_t receivedBatchSize = 1;
        size_t maxContextLength = 1024;  // default allowed input length. Otherwise, it will be read from model config.json file
        ov::genai::TokenizedInputs tokens;
        ov::Tensor typeIds;
        if (embeddings_session->getMaxModelLength().has_value()) {
            maxContextLength = embeddings_session->getMaxModelLength().value();
        } else {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "max_position_embeddings nor max_trained_positions included in config.json. Using default value {}", maxContextLength);
        }
        const bool useTokenizeEndpoint = TokenizeParser::isTokenizeEndpoint(payload.uri);
        if (useTokenizeEndpoint) {
            ovms::TokenizeRequest tokenizeRequest;
            absl::Status parsingStatus = ovms::TokenizeParser::parseTokenizeRequest(*payload.parsedJson, tokenizeRequest);
            if (!parsingStatus.ok()) {
                return parsingStatus;
            }
            auto input = tokenizeRequest.input;
            if (auto strings = std::get_if<std::vector<std::string>>(&input)) {
                auto tokenizationStatus = this->tokenizeStrings(embeddings_session->getTokenizer(), *strings, tokenizeRequest.parameters, tokens);
                if (!tokenizationStatus.ok()) {
                    return tokenizationStatus;
                }
            } else {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings tokenize input is of not supported type");
                return absl::InvalidArgumentError("Input should be string or array of strings");
            }

            StringBuffer responseBuffer;
            auto responseStatus = ovms::TokenizeParser::parseTokenizeResponse(responseBuffer, tokens, tokenizeRequest.parameters);
            if (!responseStatus.ok()) {
                return responseStatus;
            }
            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string(responseBuffer.GetString()), timestamp);
            return absl::OkStatus();
        }
        ovms::EmbeddingsHandler handler(*payload.parsedJson);
        auto parseRequestStartTime = std::chrono::high_resolution_clock::now();
        absl::Status status = handler.parseRequest();

        if (!status.ok()) {
            return status;
        }
        double time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseRequestStartTime).count();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings request deserialization time: {} ms", time / 1000);

        ModelMetricReporter unused(nullptr, nullptr, "unused", 1);
        std::unique_ptr<ExecutingStreamIdGuard> executingStreamIdGuard;
        std::unique_ptr<ExecutingStreamIdGuard> executingStreamIdGuardForPostprocessingModel;
        try {
            auto input = handler.getInput();
            if (auto strings = std::get_if<std::vector<std::string>>(&input)) {
                ov::AnyMap& params = handler.getParameters();
                receivedBatchSize = strings->size();
                if (cc->Options<EmbeddingsCalculatorOVOptions>().truncate() && params.find("max_length") == params.end()) {
                    params["max_length"] = maxContextLength;
                }
                if (embeddings_session->isStatic()) {
                    params["pad_to_max_length"] = true;
                    params["max_length"] = maxContextLength;
                }
                absl::Status tokenizationStatus = this->tokenizeStrings(embeddings_session->getTokenizer(), *strings, params, tokens);
                if (!tokenizationStatus.ok()) {
                    return tokenizationStatus;
                }

                size_t inputIdsSize = tokens.input_ids.get_shape()[1];
                auto sizeCheckStatus = this->isInputIdSizeOk(inputIdsSize, maxContextLength);
                if (!sizeCheckStatus.ok()) {
                    return sizeCheckStatus;
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
            } else if (auto tokenizedDocuments = std::get_if<std::vector<std::vector<int64_t>>>(&input)) {
                receivedBatchSize = tokenizedDocuments->size();
                size_t numberOfTokens = 0;
                size_t tokenCountOfLongestDocument = 0;
                for (const auto& document_tokens : *tokenizedDocuments) {
                    tokenCountOfLongestDocument = std::max(tokenCountOfLongestDocument, document_tokens.size());
                    numberOfTokens += document_tokens.size();
                }
                handler.setPromptTokensUsage(numberOfTokens);
                tokens.input_ids = ov::Tensor{
                    ov::element::i64,
                    ov::Shape{receivedBatchSize, tokenCountOfLongestDocument}};
                size_t inputIdsSize = tokens.input_ids.get_shape()[1];
                auto sizeCheckStatus = this->isInputIdSizeOk(inputIdsSize, maxContextLength);
                if (!sizeCheckStatus.ok()) {
                    return sizeCheckStatus;
                }

                tokens.attention_mask = ov::Tensor{
                    ov::element::i64,
                    ov::Shape{receivedBatchSize, tokenCountOfLongestDocument}};
                try {
                    for (size_t i = 0; i < receivedBatchSize; i++) {
                        int64_t* inputIdsStart = reinterpret_cast<int64_t*>(tokens.input_ids.data()) + i * tokenCountOfLongestDocument;
                        std::fill(inputIdsStart, inputIdsStart + tokenCountOfLongestDocument, embeddings_session->getPadToken().value_or(0));
                        std::copy(tokenizedDocuments->at(i).data(), tokenizedDocuments->at(i).data() + tokenizedDocuments->at(i).size(), inputIdsStart);

                        int64_t* attentionMaskStart = reinterpret_cast<int64_t*>(tokens.attention_mask.data()) + i * tokenCountOfLongestDocument;
                        std::fill(attentionMaskStart, attentionMaskStart + tokenCountOfLongestDocument, 0);
                        std::fill(attentionMaskStart, attentionMaskStart + tokenizedDocuments->at(i).size(), 1);
                    }
                } catch (std::out_of_range& e) {
                    SPDLOG_DEBUG("Caught exception from preparing embeddings inputs(): {}", e.what());
                } catch (std::exception& e) {
                    SPDLOG_DEBUG("Caught generic exception from preparing embeddings inputs: {}", e.what());
                }
                if (embeddings_session->getNumberOfModelInputs() == 3) {
                    typeIds = ov::Tensor{ov::element::i64, ov::Shape{receivedBatchSize, tokenCountOfLongestDocument}};
                    int64_t* tokenTypeIidsStart = reinterpret_cast<int64_t*>(typeIds.data());
                    std::fill(tokenTypeIidsStart, tokenTypeIidsStart + receivedBatchSize * tokenCountOfLongestDocument, 1);
                }
            } else {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings input is of not supported type");
                return absl::InvalidArgumentError("Input should be string, array of strings or array of integers");
            }

            std::vector<ov::Tensor> embeddingsTensors;
            std::vector<ov::Tensor> embeddingsAttentionMasks;
            std::string outputTensorName;
            ModelMetricReporter unused2(nullptr, nullptr, "unused2", 1);
            // NPU embeddings dynamic model case for batch size grater than 1
            if (embeddings_session->getTargetDevice() == "NPU" && receivedBatchSize > 1) {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings batch NPU request split for BS {}", receivedBatchSize);
                size_t inputIdsSize = tokens.input_ids.get_shape()[1];
                size_t attentionMaskSize = tokens.attention_mask.get_shape()[1];
                size_t typeIdsSize = 0;
                if (embeddings_session->getNumberOfModelInputs() == 3) {
                    typeIdsSize = tokens.attention_mask.get_shape()[1];
                }
                for (uint64_t i = 0; i < receivedBatchSize; i++) {
                    auto executingStreamIdGuardForMultiBatch = std::make_unique<ExecutingStreamIdGuard>(embeddings_session->getInferRequestsQueue(), unused);
                    ov::InferRequest& inferRequest = executingStreamIdGuardForMultiBatch->getInferRequest();
                    std::vector<uint64_t> startingBatchDimension = {i, 0};
                    std::vector<uint64_t> slicedDimensionEndForIdsTensor = {i + 1, inputIdsSize};
                    std::vector<uint64_t> slicedDimensionEndForAttentionMask = {i + 1, attentionMaskSize};
                    std::vector<uint64_t> slicedDimensionEndForTypeIds = {i + 1, typeIdsSize};
                    ov::Tensor oneBatchInputIdsTensor = ov::Tensor(tokens.input_ids, startingBatchDimension, slicedDimensionEndForIdsTensor);
                    ov::Tensor oneBatchAttentionMaskTensor = ov::Tensor(tokens.attention_mask, startingBatchDimension, slicedDimensionEndForAttentionMask);

                    inferRequest.set_tensor(EMBEDDINGS_MODEL_INPUT_IDS_NAME, oneBatchInputIdsTensor);
                    inferRequest.set_tensor(EMBEDDINGS_MODEL_ATTENTION_MASK_NAME, oneBatchAttentionMaskTensor);

                    if (embeddings_session->getNumberOfModelInputs() == 3) {
                        ov::Tensor oneBatchTypeIdsTensor = ov::Tensor(typeIds, startingBatchDimension, slicedDimensionEndForTypeIds);
                        inferRequest.set_tensor(EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME, oneBatchTypeIdsTensor);
                    }

                    inferRequest.start_async();
                    inferRequest.wait();
                    if (inferRequest.get_compiled_model().outputs().size() >= 2) {  // GTE
                        int targetOutputIndex = embeddings_session->getTargetOutputIndex();
                        RET_CHECK(targetOutputIndex >= 0) << "No output with 3 dimensions found";  // this should never happen as pipeline is unavailable if pooling operation could not be added
                        outputTensorName = inferRequest.get_compiled_model().outputs()[targetOutputIndex].get_any_name();
                        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Multiple embedding model outputs found, 3-dim output with name {} will be used", outputTensorName);
                    } else {  // BGE
                        RET_CHECK(inferRequest.get_compiled_model().outputs().size() == 1);
                        outputTensorName = inferRequest.get_compiled_model().outputs().begin()->get_any_name();
                        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Single embedding model output found with name {}", outputTensorName);
                    }

                    auto outputTensor = inferRequest.get_tensor(outputTensorName.c_str());
                    auto outputShape = outputTensor.get_shape();
                    auto outputElementType = outputTensor.get_element_type();
                    ov::Tensor newTensor(outputElementType, outputShape);

                    // Copy data from the original output tensor to the new one
                    std::memcpy(newTensor.data(), outputTensor.data(), outputTensor.get_byte_size());
                    embeddingsTensors.push_back(newTensor);
                    embeddingsAttentionMasks.push_back(oneBatchAttentionMaskTensor);
                }
            } else {
                // Standard CPU/GPU, NPU BS=1 path
                executingStreamIdGuard = std::make_unique<ExecutingStreamIdGuard>(embeddings_session->getInferRequestsQueue(), unused);
                ov::InferRequest& inferRequest = executingStreamIdGuard->getInferRequest();
                inferRequest.set_tensor(EMBEDDINGS_MODEL_INPUT_IDS_NAME, tokens.input_ids);
                inferRequest.set_tensor(EMBEDDINGS_MODEL_ATTENTION_MASK_NAME, tokens.attention_mask);
                if (embeddings_session->getNumberOfModelInputs() == 3) {
                    inferRequest.set_tensor(EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME, typeIds);
                }
                inferRequest.start_async();
                inferRequest.wait();
                if (inferRequest.get_compiled_model().outputs().size() >= 2) {  // GTE
                    int targetOutputIndex = embeddings_session->getTargetOutputIndex();
                    RET_CHECK(targetOutputIndex >= 0) << "No output with 3 dimensions found";  // this should never happen as pipeline is unavailable if pooling operation could not be added
                    outputTensorName = inferRequest.get_compiled_model().outputs()[targetOutputIndex].get_any_name();
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Multiple embedding model outputs found, 3-dim output with name {} will be used", outputTensorName);
                } else {  // BGE
                    RET_CHECK(inferRequest.get_compiled_model().outputs().size() == 1);
                    outputTensorName = inferRequest.get_compiled_model().outputs().begin()->get_any_name();
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Single embedding model output found with name {}", outputTensorName);
                }
                embeddingsTensor = inferRequest.get_tensor(outputTensorName.c_str());
            }

            // NPU embeddings dynamic model case
            if (embeddings_session->isNpuPostprocessingRequired()) {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "NPU embeddings dynamic model additional inference");
                executingStreamIdGuardForPostprocessingModel = std::make_unique<ExecutingStreamIdGuard>(embeddings_session->getPostProcInferRequestsQueue(), unused2);
                ov::InferRequest& inferRequestForPostprocessingMode = executingStreamIdGuardForPostprocessingModel->getInferRequest();
                if (receivedBatchSize > 1) {
                    inferRequestForPostprocessingMode.set_tensors("attention_mask", embeddingsAttentionMasks);
                    inferRequestForPostprocessingMode.set_tensors("embedding_hidden_state", embeddingsTensors);
                } else {
                    ov::Shape embeddingsResultShape = embeddingsTensor.get_shape();

                    RET_CHECK(embeddingsResultShape.size() > 1) << "Embeddings result shape must have more than 1 dimension";
                    const size_t sequenceLength = embeddingsResultShape[1];
                    const size_t originalMaskSize = tokens.attention_mask.get_size();
                    RET_CHECK(sequenceLength >= originalMaskSize) << "Attention mask size mismatch for post_request embeddings NPU request";

                    // Create attention mask tensor matching the embedding output shape
                    ov::Tensor attentionMaskTensor{ov::element::i64, {1, sequenceLength}};

                    // Copy original attention mask
                    std::copy_n(tokens.attention_mask.data<int64_t>(), originalMaskSize, attentionMaskTensor.data<int64_t>());

                    // When prefill-chunk is enabled, the input sequence length is aligned to the chunk size.
                    // For example, if the input sequence length is 3800 and the chunk size is 1024, the input
                    // sequence length will be reset to 4096. In this case, the attentionMaskTensor size is 4096,
                    // which is greater than the original tokens.attention_mask size of 3800. We need to zero-fill
                    // the remaining elements in the attentionMaskTensor to ensure correct masking behavior.
                    if (sequenceLength > originalMaskSize) {
                        std::fill_n(attentionMaskTensor.data<int64_t>() + originalMaskSize,
                            sequenceLength - originalMaskSize,
                            0);
                    }

                    // Run post-processing inference
                    inferRequestForPostprocessingMode.set_tensor("attention_mask", attentionMaskTensor);
                    inferRequestForPostprocessingMode.set_tensor("embedding_hidden_state", embeddingsTensor);
                }

                inferRequestForPostprocessingMode.start_async();
                inferRequestForPostprocessingMode.wait();

                embeddingsTensor = inferRequestForPostprocessingMode.get_tensor(outputTensorName.c_str());
            }
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught exception from session infer(): {}", e.what());
            LOG(INFO) << e.what();
            RET_CHECK(false);
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught unknown exception from session infer()");
            RET_CHECK(false);
        }

        RET_CHECK(embeddingsTensor.get_shape().size() == 2);
        RET_CHECK(embeddingsTensor.get_shape()[0] == receivedBatchSize);
        RET_CHECK(embeddingsTensor.get_element_type() == ov::element::f32);  // do we still need it?

        auto parseResponseStartTime = std::chrono::high_resolution_clock::now();
        StringBuffer buffer;
        status = handler.parseResponse(buffer, embeddingsTensor);
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
