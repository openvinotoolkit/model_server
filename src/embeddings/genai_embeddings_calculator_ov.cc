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
#include "src/port/rapidjson_writer.hpp"

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../precision.hpp"
#include "../profiler.hpp"
#include "../executingstreamidguard.hpp"
#include "../model_metric_reporter.hpp"
#include "embeddings_api.hpp"
#include "src/embeddings/embeddings_calculator_ov.pb.h"
#include "genai_embeddings_servable.hpp"

using namespace rapidjson;
using namespace ovms;
class GenaiEmbeddingsServable;

namespace mediapipe {

const std::string EMBEDDINGS_SESSION_SIDE_PACKET_TAG = "GENAI_EMBEDDINGS_NODE_RESOURCES";

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

// Helper function to print nested vectors
void printVariant(const ov::genai::EmbeddingResults& v) {
    std::visit([](const auto& data) {
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << std::setw(4) << val << " ";
            }
            std::cout << "\n";
        }
    }, v);
}

class GenaiEmbeddingsCalculatorOV : public CalculatorBase {
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

protected:
    std::shared_ptr<ovms::GenaiEmbeddingsServable> embeddings_session{nullptr};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->InputSidePackets().Tag(EMBEDDINGS_SESSION_SIDE_PACKET_TAG).Set<ovms::GenaiEmbeddingsServableMap>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "GenaiEmbeddingsCalculatorOV [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "GenaiEmbeddingsCalculatorOV  [Node: {}] Open start", cc->NodeName());
        auto servableMap = cc->InputSidePackets()
                               .Tag(EMBEDDINGS_SESSION_SIDE_PACKET_TAG)
                               .Get<ovms::GenaiEmbeddingsServableMap>();
        auto it = servableMap.find(cc->NodeName());
        RET_CHECK(it != servableMap.end()) << "Could not find initialized Embeddings node named: " << cc->NodeName();
        embeddings_session = it->second;
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "GenaiEmbeddingsCalculatorOV [Node: {}] Open end", cc->NodeName());

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
        //size_t received_batch_size = 1;
        size_t max_context_length = 1024;  // default allowed input length. Otherwise, it will be read from model config.json file
        ov::genai::TokenizedInputs tokens;
        ov::Tensor typeIds;
        if (embeddings_session->getMaxModelLength().has_value()) {
            max_context_length = embeddings_session->getMaxModelLength().value();
        } else {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "max_position_embeddings nor max_trained_positions included in config.json. Using default value {}", max_context_length);
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

        try {
            auto input = handler.getInput();
            if (auto strings = std::get_if<std::vector<std::string>>(&input)) {
                ov::AnyMap& params = handler.getParameters();
                //size_t received_batch_size = strings->size();
                if (cc->Options<EmbeddingsCalculatorOVOptions>().truncate() && params.find("max_length") == params.end()) {
                    params["max_length"] = max_context_length;
                }

                
                // TODO:handler.setPromptTokensUsage(attendedTokens);

                ov::genai::EmbeddingResults documents_embeddings = embeddings_session->m_pipeline->embed_documents(*strings);
                std::cout << std::endl << "documents_embeddings:" << std::endl;
                printVariant(documents_embeddings);
            } else if (auto tokenized_documents = std::get_if<std::vector<std::vector<int64_t>>>(&input)) { 
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Tokens on input {}", tokenized_documents->size());
                return absl::InvalidArgumentError(absl::StrCat("Tokens on input "));
            }

            
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught exception from session infer(): {}", e.what());
            LOG(INFO) << e.what();
            RET_CHECK(false);
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Caught unknown exception from session infer()");
            RET_CHECK(false);
        }

        // TODO:time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - parseResponseStartTime).count();
        //SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embeddings response deserialization time: {} ms", time / 1000);
        // TODO:buffer.GetString()
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string("buffer.GetString()"), timestamp);
        return absl::OkStatus();
    }
};
const std::string GenaiEmbeddingsCalculatorOV::INPUT_TAG_NAME{"REQUEST_PAYLOAD"};
const std::string GenaiEmbeddingsCalculatorOV::OUTPUT_TAG_NAME{"RESPONSE_PAYLOAD"};
const std::string GenaiEmbeddingsCalculatorOV::EMBEDDINGS_MODEL_INPUT_IDS_NAME{"input_ids"};
const std::string GenaiEmbeddingsCalculatorOV::EMBEDDINGS_MODEL_ATTENTION_MASK_NAME{"attention_mask"};
const std::string GenaiEmbeddingsCalculatorOV::EMBEDDINGS_MODEL_TOKEN_TYPE_IDS_NAME{"token_type_ids"};

REGISTER_CALCULATOR(GenaiEmbeddingsCalculatorOV);

}  // namespace mediapipe
