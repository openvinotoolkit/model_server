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
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>

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

    int64_t bos_token{0};
    int64_t eos_token{0};
    int64_t sep_token{0};
    int64_t pad_token{0};

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
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "RerankCalculator  [Node: {}] Open start", cc->NodeName());
        tokenizer_session = cc->InputSidePackets().Tag("TOKENIZER_SESSION").Get<std::shared_ptr<::InferenceAdapter>>();
        rerank_session = cc->InputSidePackets().Tag("RERANK_SESSION").Get<std::shared_ptr<::InferenceAdapter>>();
        try {
            this->bos_token = tokenizer_session->getModelConfig().at("bos_token_id").as<int64_t>();
            this->eos_token = tokenizer_session->getModelConfig().at("eos_token_id").as<int64_t>();
            if (tokenizer_session->getModelConfig().count("sep_token_id") == 0) {
                this->sep_token = this->eos_token;
            } else {
                this->sep_token = tokenizer_session->getModelConfig().at("sep_token_id").as<int64_t>();
            }
            this->pad_token = tokenizer_session->getModelConfig().at("pad_token_id").as<int64_t>();
        } catch (ov::AssertFailure& e) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "OpenVINO Assert Failure: {}", e.what());
            return absl::InternalError(e.what());
        } catch (...) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "Unknown error");
            return absl::InternalError("Unknown error");
        }
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "RerankCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }

    std::vector<int64_t> ComputeTokensForString(std::string str) const {
        if (tokenizer_session->getInputNames().size() != 1)
            throw std::runtime_error("Tokenizer session should have only one input");
        if (tokenizer_session->getOutputNames().size() != 2)
            throw std::runtime_error("Tokenizer session should have only two outputs");

        auto tokenizer_input_name = tokenizer_session->getInputNames()[0];
        ::InferenceInput tokenizer_input_map;
        tokenizer_input_map[tokenizer_input_name] = ov::Tensor(ov::element::string, ov::Shape{1}, &str);
        ::InferenceOutput tokenizer_output_map = tokenizer_session->infer(tokenizer_input_map);

        if (tokenizer_output_map.size() != 2)
            throw std::runtime_error("Tokenizer session should have only two outputs");
        if (tokenizer_output_map.count("input_ids") != 1)
            throw std::runtime_error("Tokenizer session should have input_ids output");
        if (tokenizer_output_map.count("attention_mask") != 1)
            throw std::runtime_error("Tokenizer session should have attention_mask output");

        auto input_ids = tokenizer_output_map.at("input_ids");
        if (input_ids.get_shape().size() != 2)
            throw std::runtime_error("input_ids should have 2 dimensions");
        if (input_ids.get_shape()[0] != 1)
            throw std::runtime_error("input_ids should have 1 batch size");

        int64_t* input_ids_data = reinterpret_cast<int64_t*>(input_ids.data());
        return std::vector<int64_t>(input_ids_data, input_ids_data + input_ids.get_shape()[1]);
    }

    std::pair<ov::Tensor, ov::Tensor> ComputeTokensForBatchedString(std::vector<std::string> strings) const {
        if (tokenizer_session->getInputNames().size() != 1)
            throw std::runtime_error("Tokenizer session should have only one input");
        if (tokenizer_session->getOutputNames().size() != 2)
            throw std::runtime_error("Tokenizer session should have only two outputs");

        auto tokenizer_input_name = tokenizer_session->getInputNames()[0];
        ::InferenceInput tokenizer_input_map;
        tokenizer_input_map[tokenizer_input_name] = ov::Tensor(ov::element::string, ov::Shape{strings.size()}, strings.data());
        ::InferenceOutput tokenizer_output_map = tokenizer_session->infer(tokenizer_input_map);

        if (tokenizer_output_map.size() != 2)
            throw std::runtime_error("Tokenizer session should have only two outputs");
        if (tokenizer_output_map.count("input_ids") != 1)
            throw std::runtime_error("Tokenizer session should have input_ids output");
        if (tokenizer_output_map.count("attention_mask") != 1)
            throw std::runtime_error("Tokenizer session should have attention_mask output");

        auto input_ids = tokenizer_output_map.at("input_ids");
        if (input_ids.get_shape().size() != 2)
            throw std::runtime_error("input_ids should have 2 dimensions");
        if (input_ids.get_shape()[0] != strings.size())
            throw std::runtime_error("input_ids should have batch size equal to number of tokenized strings");

        auto attention_mask = tokenizer_output_map.at("attention_mask");
        if (attention_mask.get_shape().size() != 2)
            throw std::runtime_error("attention_mask should have 2 dimensions");
        if (attention_mask.get_shape()[0] != strings.size())
            throw std::runtime_error("attention_mask should have batch size equal to number of tokenized strings");

        return std::make_pair(input_ids, attention_mask);
    }

    std::pair<ov::Tensor, ov::Tensor> PrepareInputsForRerankModel(const RerankHandler& handler) const {
        // Compute Query Tokens
        auto query_tokens = ComputeTokensForString(handler.getQuery());

        // Comkpute Document Tokens
        auto [doc_input_ids, doc_attention_mask] = ComputeTokensForBatchedString(handler.getDocumentsList());

        size_t tokens_count_of_longest_document = doc_input_ids.get_shape()[1];
        size_t total_tokens_count_per_batch = tokens_count_of_longest_document + 4 /*special tokens*/ + query_tokens.size();
        size_t batch_size = handler.getDocumentsList().size();
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{batch_size, total_tokens_count_per_batch});
        auto attention_mask = ov::Tensor(ov::element::i64, ov::Shape{batch_size, total_tokens_count_per_batch});

        // Combine query and document tokens
        // Schema (tokenizer must be exported without --add_special_tokens flag, we will add it manually)
        /*
            BOS_TOKEN  <QUERY TOKENS>  EOS_TOKEN SEP_TOKEN  <DOCUMENT_1 TOKENS>  EOS_TOKEN
            BOS_TOKEN  <QUERY TOKENS>  EOS_TOKEN SEP_TOKEN  <DOCUMENT_2 TOKENS>  EOS_TOKEN
            BOS_TOKEN  <QUERY TOKENS>  EOS_TOKEN SEP_TOKEN  <DOCUMENT_3 TOKENS>  EOS_TOKEN
            BOS_TOKEN  <QUERY TOKENS>  EOS_TOKEN SEP_TOKEN  <DOCUMENT_N TOKENS>  EOS_TOKEN
        */

        // TODO: Error when exceeding max length
        // TODO: Consider secondary dimension (max tokens per batch?)
        for (size_t i = 0; i < batch_size; i++) {
            int64_t* input_ids_data = reinterpret_cast<int64_t*>(input_ids.data()) + i * total_tokens_count_per_batch;
            int64_t* attention_mask_data = reinterpret_cast<int64_t*>(attention_mask.data()) + i * total_tokens_count_per_batch;

            int64_t* doc_input_ids_data = reinterpret_cast<int64_t*>(doc_input_ids.data()) + i * tokens_count_of_longest_document;

            input_ids_data[0] = this->bos_token;
            std::memcpy(input_ids_data + 1, query_tokens.data(), query_tokens.size() * sizeof(int64_t));
            input_ids_data[query_tokens.size() + 1] = this->eos_token;
            input_ids_data[query_tokens.size() + 2] = this->sep_token;
            std::memcpy(input_ids_data + 1 + query_tokens.size() + 2, doc_input_ids_data, tokens_count_of_longest_document * sizeof(int64_t));

            input_ids_data[total_tokens_count_per_batch - 1] = this->pad_token;

            auto it = std::find(doc_input_ids_data, doc_input_ids_data + tokens_count_of_longest_document, this->pad_token);
            size_t pad_token_index = (it != doc_input_ids_data + tokens_count_of_longest_document) ? std::distance(doc_input_ids_data, it) : tokens_count_of_longest_document;

            input_ids_data[1 + query_tokens.size() + 2 + pad_token_index] = this->eos_token;

            // attention_mask
            std::fill(attention_mask_data, attention_mask_data + total_tokens_count_per_batch, int64_t(0));
            std::fill(attention_mask_data, attention_mask_data + 1 + query_tokens.size() + 2 + pad_token_index + 1, int64_t(1));
        }

        return std::make_pair(input_ids, attention_mask);
    }

    std::vector<float> ComputeScoresUsingRerankModel(ov::Tensor input_ids, ov::Tensor attention_mask) const {
        if (rerank_session->getInputNames().size() != 2)  // TODO: Support 3 inputs with token_type_ids
            throw std::runtime_error("Rerank model should have 2 inputs");
        if (rerank_session->getOutputNames().size() != 1)  // There should be only one output when exported with --task text-classification
            throw std::runtime_error("Rerank model should have 1 output");

        // Validate input/output names
        if (rerank_session->getInputNames()[0] != "input_ids" && rerank_session->getInputNames()[1] != "input_ids")
            throw std::runtime_error("Rerank model should have input_ids input");
        if (rerank_session->getInputNames()[0] != "attention_mask" && rerank_session->getInputNames()[1] != "attention_mask")
            throw std::runtime_error("Rerank model should have attention_mask input");
        if (rerank_session->getOutputNames()[0] != "logits")
            throw std::runtime_error("Rerank model should have logits output");

        ::InferenceInput rerank_input_map;
        rerank_input_map["input_ids"] = input_ids;
        rerank_input_map["attention_mask"] = attention_mask;

        ::InferenceOutput rerank_output_map = rerank_session->infer(rerank_input_map);
        if (rerank_output_map.size() != 1)
            throw std::runtime_error("Rerank model results should have 1 output");
        if (rerank_output_map.count("logits") != 1)
            throw std::runtime_error("Rerank model results should have logits output");

        auto logits = rerank_output_map.at("logits");

        if (logits.get_shape().size() != 2)  // 2D tensor
            throw std::runtime_error("Logits should be 2D tensor");
        if (logits.get_shape()[0] != input_ids.get_shape()[0])
            throw std::runtime_error("Batch size mismatch");

        std::vector<float> scores;
        int logits_dim = logits.get_shape()[1];

        if (logits_dim > 1) {
            // Extract the second column
            for (int i = 0; i < input_ids.get_shape()[0]; ++i) {
                float logit = reinterpret_cast<float*>(logits.data())[i * logits_dim + 1];  // TODO: Untested, model has second dimension=1, taken from OpenVINOReranker
                scores.push_back(1 / (1 + std::exp(-logit)));
            }
        } else {
            // Flatten the logits
            for (int i = 0; i < input_ids.get_shape()[0]; ++i) {
                float logit = reinterpret_cast<float*>(logits.data())[i];
                scores.push_back(1 / (1 + std::exp(-logit)));
            }
        }

        return scores;
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

        try {
            // Prepare inputs for rerank model
            auto [input_ids, attention_mask] = PrepareInputsForRerankModel(handler);
            auto scores = ComputeScoresUsingRerankModel(input_ids, attention_mask);

            // Print scores for verification until we have serialization
            for (const auto& score : scores) {
                std::cout << score << " ";
                // 0.343912 0.00104043
            }
            std::cout << std::endl;

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
