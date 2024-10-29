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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <exception>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#pragma GCC diagnostic pop

#include <adapters/inference_adapter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
#include "absl/strings/escaping.h"
#include "src/rerank/rerank_calculator.pb.h"
#include "src/rerank/rerank_utils.hpp"

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

class RerankCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;

protected:
    std::shared_ptr<::InferenceAdapter> tokenizer_session{nullptr};
    std::shared_ptr<::InferenceAdapter> rerank_session{nullptr};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->InputSidePackets().Tag("TOKENIZER_SESSION").Set<std::shared_ptr<InferenceAdapter>>();
        cc->InputSidePackets().Tag("RERANK_SESSION").Set<std::shared_ptr<InferenceAdapter>>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "RerankCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        tokenizer_session = cc->InputSidePackets().Tag("TOKENIZER_SESSION").Get<std::shared_ptr<::InferenceAdapter>>();
        rerank_session = cc->InputSidePackets().Tag("RERANK_SESSION").Get<std::shared_ptr<::InferenceAdapter>>();
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "RerankCalculator  [Node: {}] Open start", cc->NodeName());
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "RerankCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }

    std::vector<int64_t> ComputeTokensForString(std::string str) const {
        if (tokenizer_session->getInputNames().size() != 1) {
            throw std::runtime_error("Tokenizer session should have only one input");
        }
        if (tokenizer_session->getOutputNames().size() != 2) {
            throw std::runtime_error("Tokenizer session should have only two outputs");
        }
        auto tokenizerInputName = tokenizer_session->getInputNames()[0];
        ::InferenceInput tokenizerInputMap;
        tokenizerInputMap[tokenizerInputName] = ov::Tensor(ov::element::string, ov::Shape{1}, &str);
        ::InferenceOutput tokenizerOutputMap = tokenizer_session->infer(tokenizerInputMap);
        if (tokenizerOutputMap.size() != 2) {
            throw std::runtime_error("Tokenizer session should have only two outputs");
        }
        if (tokenizerOutputMap.count("input_ids") != 1) {
            throw std::runtime_error("Tokenizer session should have input_ids output");
        }
        if (tokenizerOutputMap.count("attention_mask") != 1) {
            throw std::runtime_error("Tokenizer session should have attention_mask output");
        }
        auto input_ids = tokenizerOutputMap.at("input_ids");
        if (input_ids.get_shape().size() != 2) {
            throw std::runtime_error("input_ids should have 2 dimensions");
        }
        if (input_ids.get_shape()[0] != 1) {
            throw std::runtime_error("input_ids should have 1 batch size");
        }
        auto input_ids_data = (int64_t*)input_ids.data();
        std::vector<int64_t> tokens;
        for (size_t i = 0; i < input_ids.get_shape()[1]; i++) {
            tokens.push_back(input_ids_data[i]);
        }
        return tokens;
    }

    std::pair<ov::Tensor, ov::Tensor> PrepareTokens(const RerankHandler& handler, int64_t bos, int64_t eos, int64_t pad) const {
        // Compute Query Tokens
        auto query_tokens = ComputeTokensForString(handler.getQuery());

        // Iterate over documents and compute tokens
        std::vector<std::vector<int64_t>> document_tokens;
        for (const auto& doc : handler.getDocumentsList()) {
            auto doc_tokens = ComputeTokensForString(doc);
            document_tokens.emplace_back(std::move(doc_tokens));
        }

        // Get max length of document tokens
        size_t max_length = 0;
        for (const auto& doc_tokens : document_tokens) {
            max_length = std::max(max_length, doc_tokens.size());
        }

        // Width of the final tensor
        size_t total_max_length = max_length + 4/*special tokens*/ + query_tokens.size();
        

        // Combine query and document tokens

        // Schema:
        /*
            BOS_TOKEN < QUERY TOKENS > EOS_TOKEN EOS_TOKEN < DOCUMENT_1  TOKENS > EOS_TOKEN
            BOS_TOKEN < QUERY TOKENS > EOS_TOKEN EOS_TOKEN < DOCUMENT_2  TOKENS > EOS_TOKEN
            BOS_TOKEN < QUERY TOKENS > EOS_TOKEN EOS_TOKEN < DOCUMENT_3  TOKENS > EOS_TOKEN
            BOS_TOKEN < QUERY TOKENS > EOS_TOKEN EOS_TOKEN < DOCUMENT_...TOKENS > EOS_TOKEN
        */

        size_t batch_size = handler.getDocumentsList().size();
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{batch_size, total_max_length});
        auto attention_mask = ov::Tensor(ov::element::i64, ov::Shape{batch_size, total_max_length});

        for (size_t i = 0; i < handler.getDocumentsList().size(); i++) {
            int64_t* input_ids_data = ((int64_t*)input_ids.data()) + i * total_max_length;
            int64_t* attention_mask_data = ((int64_t*)attention_mask.data()) + i * total_max_length;
            std::fill(input_ids_data, input_ids_data + total_max_length, pad);
            std::fill(attention_mask_data, attention_mask_data + total_max_length, 0);
            input_ids_data[0] = bos;
            std::memcpy(input_ids_data + 1, query_tokens.data(), query_tokens.size() * sizeof(int64_t));
            input_ids_data[query_tokens.size() + 1] = eos;
            input_ids_data[query_tokens.size() + 2] = eos;
            std::memcpy(input_ids_data + query_tokens.size() + 3, document_tokens[i].data(), document_tokens[i].size() * sizeof(int64_t));
            input_ids_data[query_tokens.size() + 3 + document_tokens[i].size()] = eos;
            std::fill(attention_mask_data, attention_mask_data + 1 + query_tokens.size() + 3 + document_tokens[i].size(), 1);
        }

        // TODO: Log the prepared data?

        return std::make_pair(input_ids, attention_mask);
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        RET_CHECK(tokenizer_session != nullptr);
        RET_CHECK(rerank_session != nullptr);
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::InvalidArgumentError("Input is empty");
        }
        InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Request body: {}", payload.body);
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Request uri: {}", payload.uri);
        RerankHandler handler(*payload.parsedJson);
        absl::Status status = handler.parseRequest();
        if (!status.ok()) {
            return status;
        }

        size_t batch_size = handler.getDocumentsList().size();

        try {

            int64_t bos_token = tokenizer_session->getModelConfig().at("bos_token_id").as<int64_t>();
            int64_t eos_token = tokenizer_session->getModelConfig().at("eos_token_id").as<int64_t>();
            int64_t pad_token = tokenizer_session->getModelConfig().at("pad_token_id").as<int64_t>();

            auto [input_ids, attention_mask] = PrepareTokens(handler, bos_token, eos_token, pad_token);

            RET_CHECK(rerank_session->getInputNames().size() == 2);
            RET_CHECK(rerank_session->getOutputNames().size() == 1);
            // check if input names are correct, may be in any order
            RET_CHECK(rerank_session->getInputNames()[0] == "input_ids" || rerank_session->getInputNames()[1] == "input_ids");
            RET_CHECK(rerank_session->getInputNames()[0] == "attention_mask" || rerank_session->getInputNames()[1] == "attention_mask");
            RET_CHECK(rerank_session->getOutputNames()[0] == "logits");

            ::InferenceInput rerankInputMap;
            rerankInputMap["input_ids"] = input_ids;
            rerankInputMap["attention_mask"] = attention_mask;
            // TODO: Support 3 input models?
    
            ::InferenceOutput rerankOutputMap = rerank_session->infer(rerankInputMap);
            RET_CHECK(rerankOutputMap.size() == 1);
            RET_CHECK(rerankOutputMap.count("logits") == 1);

            auto logits = rerankOutputMap.at("logits");

            RET_CHECK(logits.get_shape().size() == 2);
            RET_CHECK(logits.get_shape()[0] == batch_size);

            std::vector<float> scores;
            int logits_dim = logits.get_shape()[1];

            if (logits_dim > 1) {
                // Extract the second column
                for (int i = 0; i < batch_size; ++i) {
                    float logit = ((float*)logits.data())[i * logits_dim + 1];
                    scores.push_back(1 / (1 + std::exp(-logit)));
                }
            } else {
                // Flatten the logits
                for (int i = 0; i < batch_size; ++i) {
                    float logit = ((float*)logits.data())[i];
                    std::cout << logit << std::endl;
                    scores.push_back(1 / (1 + std::exp(-logit)));
                }
            }

            // Print scores for verification
            for (const auto& score : scores) {
                std::cout << score << " ";
            }
            std::cout << std::endl;

            // TODO: Support other fields of input

            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string("TODO"), timestamp);  // Add serialization
            return absl::OkStatus();
        
        } catch (ov::AssertFailure& e) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "OpenVINO Assert Failure: {}", e.what());
            return absl::InternalError(e.what());
        } catch (std::runtime_error& e) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "runtime_error: {}", e.what());
            return absl::InternalError(e.what());
        } catch (...) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "Unknown error");
            return absl::InternalError("Unknown error");
        }
    }
};
const std::string RerankCalculator::INPUT_TAG_NAME{"REQUEST_PAYLOAD"};
const std::string RerankCalculator::OUTPUT_TAG_NAME{"RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(RerankCalculator);

}  // namespace mediapipe
