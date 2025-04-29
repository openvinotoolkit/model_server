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

#pragma warning(push)
#pragma warning(disable : 6001 6385 6386 6326 6011 4309 6246 4005 4456)
#include "absl/strings/escaping.h"
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
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
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
    static constexpr size_t NUMBER_OF_SPECIAL_TOKENS = 4;

    std::chrono::time_point<std::chrono::system_clock> created;

    int64_t bos_token{0};
    int64_t eos_token{0};
    int64_t sep_token{0};
    int64_t pad_token{0};

    uint64_t max_position_embeddings{512};

    size_t max_allowed_chunks{0};  // Read from options in ::Open()

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

        const auto& options = cc->Options<RerankCalculatorOptions>();
        this->max_allowed_chunks = options.max_allowed_chunks();
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Max allowed chunks: {}", this->max_allowed_chunks);

        try {
            // special tokens
            this->bos_token = rerank_session->getModelConfig().at("bos_token_id").as<int64_t>();
            this->eos_token = rerank_session->getModelConfig().at("eos_token_id").as<int64_t>();
            if (rerank_session->getModelConfig().count("sep_token_id") == 0) {
                this->sep_token = this->eos_token;
            } else {
                this->sep_token = rerank_session->getModelConfig().at("sep_token_id").as<int64_t>();
            }
            this->pad_token = rerank_session->getModelConfig().at("pad_token_id").as<int64_t>();

            // max_position_embeddings
            if (options.has_max_position_embeddings()) {
                this->max_position_embeddings = options.max_position_embeddings();
                SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Options defined max_position_embeddings: {}", this->max_position_embeddings);
            } else {
                auto maxPositionEmbeddingsIt = rerank_session->getModelConfig().find("max_position_embeddings");
                if (maxPositionEmbeddingsIt != rerank_session->getModelConfig().end()) {
                    this->max_position_embeddings = maxPositionEmbeddingsIt->second.as<int64_t>();
                    SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Model max_position_embeddings: {}", this->max_position_embeddings);
                } else {
                    auto maxTrainedPositionsIt = rerank_session->getModelConfig().find("max_trained_positions");
                    if (maxTrainedPositionsIt != rerank_session->getModelConfig().end()) {
                        this->max_position_embeddings = maxTrainedPositionsIt->second.as<int64_t>();
                        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Model max_position_embeddings (inherited from max_trained_positions): {}", this->max_position_embeddings);
                    } else {
                        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Model missing max_position_embeddings and max_trained_positions in config, using default value: {}", this->max_position_embeddings);
                    }
                }
            }

            // post-validation
            if (this->max_position_embeddings <= 2 * NUMBER_OF_SPECIAL_TOKENS) {
                SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "max_position_embeddings should be larger than 2 * NUMBER_OF_SPECIAL_TOKENS");
                return absl::InvalidArgumentError("max_position_embeddings should be larger than 2 * NUMBER_OF_SPECIAL_TOKENS");
            }
        } catch (ov::AssertFailure& e) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "OpenVINO Assert Failure: {}", e.what());
            return absl::InternalError(e.what());
        } catch (std::out_of_range& e) {
            SPDLOG_LOGGER_ERROR(rerank_calculator_logger, "{}", e.what());
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
        if (input_ids.get_element_type() != ov::element::i64)
            throw std::runtime_error("input_ids should have i64 element type");  // TODO: Add support for other precisions?

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
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Starting inference tokenizer model");
        ::InferenceOutput tokenizer_output_map = tokenizer_session->infer(tokenizer_input_map);
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Finished inference tokenizer model");

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
        if (input_ids.get_element_type() != ov::element::i64)
            throw std::runtime_error("input_ids should have i64 element type");

        auto attention_mask = tokenizer_output_map.at("attention_mask");
        if (attention_mask.get_shape().size() != 2)
            throw std::runtime_error("attention_mask should have 2 dimensions");
        if (attention_mask.get_shape()[0] != strings.size())
            throw std::runtime_error("attention_mask should have batch size equal to number of tokenized strings");
        if (attention_mask.get_element_type() != ov::element::i64)
            throw std::runtime_error("attention_mask should have i64 element type");  // TODO: Add support for other precisions?

        return std::make_pair(input_ids, attention_mask);
    }

    std::pair<ov::Tensor, ov::Tensor> PrepareInputsForRerankModel(const RerankHandler& handler, std::vector<size_t>& chunk_mapping) const {
        // Validate batch size before tokenizing
        if (handler.getDocumentsList().size() > this->max_allowed_chunks)
            throw std::runtime_error("Number of documents exceeds max_allowed_chunks");
        // TODO: Validate max string length for some arbitrary size

        // Compute Query Tokens
        auto query_tokens = ComputeTokensForString(handler.getQuery());

        // Truncate last tokens if exceeding max_position_embeddings / 2 as mentioned in cohere doc:
        // https://docs.cohere.com/v2/docs/reranking-best-practices#queries
        const size_t max_query_tokens = this->max_position_embeddings / 2;
        if (query_tokens.size() > max_query_tokens) {
            query_tokens.resize(max_query_tokens);
            SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Number of query tokens: {} exceeded half of max_position_embeddings: {}, truncating to {}", query_tokens.size(), this->max_position_embeddings, max_query_tokens);
        } else {
            SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Number of query tokens: {}", query_tokens.size());
        }

        // Compute Document Tokens
        auto [doc_input_ids, doc_attention_mask] = ComputeTokensForBatchedString(handler.getDocumentsList());

        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "\nMax position embeddings: {}\nQuery tokens: {}\nSpecial tokens: {}\nRemaining space for chunk: {}",
            this->max_position_embeddings, query_tokens.size(), NUMBER_OF_SPECIAL_TOKENS, this->max_position_embeddings - query_tokens.size() - NUMBER_OF_SPECIAL_TOKENS);
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Number of documents: {}; with max token count: {} before chunking", doc_input_ids.get_shape()[0], doc_input_ids.get_shape()[1]);

        // max_tokens_per_chunk can never be <= 0 since query_tokens.size() is at max half of max_position_embeddings
        // and max_position_embeddings is at least 2 * NUMBER_OF_SPECIAL_TOKENS
        size_t max_tokens_per_chunk = this->max_position_embeddings - query_tokens.size() - NUMBER_OF_SPECIAL_TOKENS;
        ov::Tensor out_input_ids, out_attention_mask;
        auto status = chunkDocuments(
            doc_input_ids,
            doc_attention_mask,
            out_input_ids, out_attention_mask,
            chunk_mapping, max_tokens_per_chunk,
            this->max_allowed_chunks, this->pad_token);
        if (!status.ok()) {
            throw std::runtime_error(std::string{"Chunking failed: "} + std::string(status.message()));
        }

        doc_input_ids = out_input_ids;
        doc_attention_mask = out_attention_mask;

        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Number of chunks: {}; with max token count: {} after chunking", doc_input_ids.get_shape()[0], doc_input_ids.get_shape()[1]);

        size_t tokens_count_of_longest_document = doc_input_ids.get_shape()[1];
        if (tokens_count_of_longest_document > max_tokens_per_chunk)
            throw std::runtime_error("tokens_count_of_longest_document exceeds max_tokens_per_chunk");  // should never happen
        size_t total_tokens_count_per_batch = tokens_count_of_longest_document + NUMBER_OF_SPECIAL_TOKENS + query_tokens.size();
        size_t batch_size = doc_input_ids.get_shape()[0];
        if (batch_size != chunk_mapping.size())
            throw std::runtime_error("error");  // should never happen

        if (total_tokens_count_per_batch > this->max_position_embeddings)
            throw std::runtime_error("Query tokens count + special tokens + tokens count of longest document exceeds max_position_embeddings");

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

        for (size_t i = 0; i < batch_size; i++) {
            int64_t* input_ids_data = reinterpret_cast<int64_t*>(input_ids.data()) + i * total_tokens_count_per_batch;
            int64_t* attention_mask_data = reinterpret_cast<int64_t*>(attention_mask.data()) + i * total_tokens_count_per_batch;

            int64_t* doc_input_ids_data = reinterpret_cast<int64_t*>(doc_input_ids.data()) + i * tokens_count_of_longest_document;

            // Fill input_ids
            input_ids_data[0] = this->bos_token;
            std::memcpy(input_ids_data + 1, query_tokens.data(), query_tokens.size() * sizeof(int64_t));
            input_ids_data[query_tokens.size() + 1] = this->eos_token;
            input_ids_data[query_tokens.size() + 2] = this->sep_token;
            std::memcpy(input_ids_data + 1 + query_tokens.size() + 2, doc_input_ids_data, tokens_count_of_longest_document * sizeof(int64_t));

            input_ids_data[total_tokens_count_per_batch - 1] = this->pad_token;

            auto it = std::find(doc_input_ids_data, doc_input_ids_data + tokens_count_of_longest_document, this->pad_token);
            size_t pad_token_index = (it != doc_input_ids_data + tokens_count_of_longest_document) ? std::distance(doc_input_ids_data, it) : tokens_count_of_longest_document;

            input_ids_data[1 + query_tokens.size() + 2 + pad_token_index] = this->eos_token;

            // Fill attention_mask
            std::fill(attention_mask_data, attention_mask_data + total_tokens_count_per_batch, int64_t(0));
            std::fill(attention_mask_data, attention_mask_data + 1 + query_tokens.size() + 2 + pad_token_index + 1, int64_t(1));
        }

        return std::make_pair(input_ids, attention_mask);
    }

    std::vector<float> ComputeScoresUsingRerankModel(ov::Tensor input_ids, ov::Tensor attention_mask, const std::vector<size_t>& chunkMapping, size_t actual_batch_size) const {
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

        if (input_ids.get_shape()[1] > this->max_position_embeddings)
            throw std::runtime_error("exceeding max_position_embeddings");  // should never happen

        ::InferenceInput rerank_input_map;
        rerank_input_map["input_ids"] = input_ids;
        rerank_input_map["attention_mask"] = attention_mask;

        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Starting inference rerank model");
        ::InferenceOutput rerank_output_map = rerank_session->infer(rerank_input_map);
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Finished inference rerank model");
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
        scores.resize(actual_batch_size, 0);

        size_t logits_dim = logits.get_shape()[1];

        for (int i = 0; i < input_ids.get_shape()[0]; ++i) {
            size_t score_index = chunkMapping[i];
            if (score_index >= actual_batch_size)
                throw std::runtime_error("score_index out of bounds");  // should never happen
            float logit = logits_dim > 1 ? reinterpret_cast<float*>(logits.data())[i * logits_dim + 1] : reinterpret_cast<float*>(logits.data())[i];
            float score = 1 / (1 + std::exp(-logit));
            float current_highest_score = scores[score_index];
            scores[score_index] = std::max(current_highest_score, score);
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
            std::vector<size_t> chunk_mapping;
            auto [input_ids, attention_mask] = PrepareInputsForRerankModel(handler, chunk_mapping);

            // Compute scores using rerank model
            size_t batch_size = handler.getDocumentsList().size();
            auto scores = ComputeScoresUsingRerankModel(
                input_ids,
                attention_mask,
                chunk_mapping,
                batch_size);

            // Serialize scores
            StringBuffer buffer;
            status = handler.parseResponse(buffer, scores);
            if (!status.ok()) {
                return status;
            }
            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string(buffer.GetString()), cc->InputTimestamp());
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
