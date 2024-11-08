//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#pragma GCC diagnostic pop

#include <adapters/inference_adapter.h>
#include <rapidjson/writer.h>

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../precision.hpp"
#include "../profiler.hpp"
#include "embeddings_api.hpp"
#include "src/embeddings/embeddings_calculator.pb.h"

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

class EmbeddingsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string EMBEDDINGS_MODEL_INPUT_IDS_NAME;

    mediapipe::Timestamp timestamp{0};

protected:
    std::shared_ptr<::InferenceAdapter> tokenizer_session{nullptr};
    std::shared_ptr<::InferenceAdapter> embeddings_session{nullptr};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->InputSidePackets().Tag("TOKENIZER_SESSION").Set<std::shared_ptr<InferenceAdapter>>();
        cc->InputSidePackets().Tag("EMBEDDINGS_SESSION").Set<std::shared_ptr<InferenceAdapter>>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        tokenizer_session = cc->InputSidePackets()
                                .Tag("TOKENIZER_SESSION")
                                .Get<std::shared_ptr<::InferenceAdapter>>();
        embeddings_session = cc->InputSidePackets()
                                 .Tag("EMBEDDINGS_SESSION")
                                 .Get<std::shared_ptr<::InferenceAdapter>>();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator  [Node: {}] Open start", cc->NodeName());

        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator [Node: {}] Open end", cc->NodeName());

        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        RET_CHECK(tokenizer_session != nullptr);
        RET_CHECK(embeddings_session != nullptr);
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::InvalidArgumentError("Input is empty");
        }
        InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request body: {}", payload.body);
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request uri: {}", payload.uri);
        EmbeddingsHandler handler(*payload.parsedJson);
        auto parseRequestStartTime = std::chrono::high_resolution_clock::now();
        absl::Status status = handler.parseRequest();
        if (!status.ok()) {
            return status;
        }
        double time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseRequestStartTime).count();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings request deserialization time: {} ms", time / 1000);

        // Automatically deduce tokenizer input name
        std::vector<std::string> tokenizerInputNames = tokenizer_session->getInputNames();
        std::vector<std::string> embeddingsInputNames = embeddings_session->getInputNames();
        RET_CHECK(tokenizerInputNames.size() == 1);
        const std::string& tokenizerInputName = tokenizerInputNames.at(0);
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Tokenizer input name detected: {}", tokenizerInputName);

        ::InferenceInput tokenizerInputMap;
        size_t received_batch_size = 1;
        ::InferenceOutput embeddingsOutputMap;
        try {
            ::InferenceOutput tokenizerOutputMap;
            auto input = handler.getInput();
            if (auto strings = std::get_if<std::vector<std::string>>(&input)) {
                received_batch_size = strings->size();
                tokenizerInputMap[tokenizerInputName] = ov::Tensor{
                    ov::element::string,
                    ov::Shape{received_batch_size},
                    strings->data()};
                tokenizerOutputMap = tokenizer_session->infer(tokenizerInputMap);
            } else {
                // TODO: input already tokenized
                return absl::InvalidArgumentError("not implemented");
            }
            ::InferenceInput embeddingsInputMap;
            // Check if tokenizer produced at least the number of outputs as there are inputs in embedding model
            RET_CHECK(tokenizerOutputMap.size() >= embeddingsInputNames.size());
            for (const auto& embeddingsInputName : embeddingsInputNames) {
                auto it = tokenizerOutputMap.find(embeddingsInputName);
                RET_CHECK(it != tokenizerOutputMap.end());
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embedding model input {} is connected with matching tokenizer output", embeddingsInputName);
                embeddingsInputMap[embeddingsInputName] = it->second;
                if (embeddingsInputName == "attention_mask") {
                    if (received_batch_size == 1) {
                        handler.setPromptTokensUsage(it->second.get_size());
                        continue;
                    }
                    size_t attendedTokens = 0;
                    if (it->second.get_element_type() == ov::element::Type_t::i64) {
                        for (int i = 0; i < it->second.get_size(); i++) {
                            attendedTokens += reinterpret_cast<int64_t*>(it->second.data())[i];
                        }
                    } else if (it->second.get_element_type() == ov::element::Type_t::i32) {
                        for (int i = 0; i < it->second.get_size(); i++) {
                            attendedTokens += reinterpret_cast<int32_t*>(it->second.data())[i];
                        }
                    } else {
                        for (int i = 0; i < it->second.get_byte_size(); i++) {
                            attendedTokens += reinterpret_cast<uint8_t*>(it->second.data())[i];
                        }
                    }
                    handler.setPromptTokensUsage(attendedTokens);
                }
            }
            size_t max_context_length = 512;  // default allowed input length. Otherwise, it will be read from model rt_info>config>max_position_embeddings in the model.xml file
            ov::AnyMap modelConfig = embeddings_session->getModelConfig();
            try {
                if (modelConfig.count("max_position_embeddings")) {
                    max_context_length = modelConfig["max_position_embeddings"].as<size_t>();
                } else if (modelConfig.count("max_trained_positions")) {
                    max_context_length = modelConfig["max_trained_positions"].as<size_t>();
                } else {
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "model_info->max_position_embeddings nor max_trained_positions included in model rt_info. Using default value {}", max_context_length);
                }
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Detected model context size: ", max_context_length);
            } catch (...) {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Can not read model context length from rt_info. Using default value {}", max_context_length);
            }
            RET_CHECK(tokenizerOutputMap[EMBEDDINGS_MODEL_INPUT_IDS_NAME].get_shape().size() == 2);
            size_t input_ids_size = tokenizerOutputMap[EMBEDDINGS_MODEL_INPUT_IDS_NAME].get_shape()[1];
            if (input_ids_size > max_context_length) {
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Input size {} exceeds max_context_length {}", input_ids_size, max_context_length);
                return absl::InvalidArgumentError(absl::StrCat("Input length ", input_ids_size, " longer than allowed ", max_context_length));
            }
            // creating output to avoid copying big tensor
            ov::Shape outShape = embeddingsInputMap.at(EMBEDDINGS_MODEL_INPUT_IDS_NAME).get_shape();
            bool foundMatchinEmbeddingOutput{false};
            std::string outputNameToSet;
            for (auto& name : embeddings_session->getOutputNames()) {
                ov::PartialShape outPShape = embeddings_session->getOutputShape(name);
                if (outPShape.size() != 3)
                    continue;
                try {
                    outShape.emplace_back(outPShape[2].get_length());
                    foundMatchinEmbeddingOutput = true;
                    outputNameToSet = name;
                    break;
                } catch (std::exception& e) {
                    LOG(ERROR) << "Failed to get 3rd dimension of output" << outputNameToSet;
                    return absl::InternalError(absl::StrCat("Failed to get 3rd dimension of output: ", outputNameToSet));
                }
            }
            if (!foundMatchinEmbeddingOutput) {
                LOG(INFO) << "Failed to find matching output for correct output setting optimization";
                return absl::InternalError("Could not find output with 3 dimensions in embeddings model");
            }
            ov::Tensor outputTensor(embeddings_session->getOutputDatatype(outputNameToSet), outShape);
            embeddingsOutputMap.emplace(outputNameToSet, std::move(outputTensor));
            embeddings_session->infer(embeddingsInputMap, embeddingsOutputMap);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught exception from session infer(): {}", e.what());
            LOG(INFO) << e.what();
            RET_CHECK(false);
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught unknown exception from session infer()");
            RET_CHECK(false);
        }
        ov::Tensor embeddingsTensor;
        if (embeddingsOutputMap.size() == 2) {  // GTE
            // Search by number of dimensions, should be 3
            bool found = false;
            for (const auto& [name, tensor] : embeddingsOutputMap) {
                if (tensor.get_shape().size() == 3) {
                    embeddingsTensor = tensor;
                    SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Multiple embedding model outputs found, 3-dim output with name {} will be used", name);
                    found = true;
                    break;
                }
            }
            RET_CHECK(found);
        } else {  // BGE
            RET_CHECK(embeddingsOutputMap.size() == 1);
            embeddingsTensor = embeddingsOutputMap.begin()->second;
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Single embedding model output found with name {}", embeddingsOutputMap.begin()->first);
        }

        RET_CHECK(embeddingsTensor.get_shape().size() == 3);
        RET_CHECK(embeddingsTensor.get_shape()[0] == received_batch_size);
        RET_CHECK(embeddingsTensor.get_element_type() == ov::element::f32);

        auto parseResponseStartTime = std::chrono::high_resolution_clock::now();
        StringBuffer buffer;
        status = handler.parseResponse(buffer, embeddingsTensor, cc->Options<EmbeddingsCalculatorOptions>().normalize_embeddings());
        if (!status.ok()) {
            return status;
        }
        time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseResponseStartTime).count();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings response deserialization time: {} ms", time / 1000);
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string(buffer.GetString()), timestamp);
        return absl::OkStatus();
    }
};
const std::string EmbeddingsCalculator::INPUT_TAG_NAME{"REQUEST_PAYLOAD"};
const std::string EmbeddingsCalculator::OUTPUT_TAG_NAME{"RESPONSE_PAYLOAD"};
const std::string EmbeddingsCalculator::EMBEDDINGS_MODEL_INPUT_IDS_NAME{"input_ids"};

REGISTER_CALCULATOR(EmbeddingsCalculator);

}  // namespace mediapipe
