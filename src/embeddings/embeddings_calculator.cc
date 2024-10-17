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
#include "src/embeddings/embeddings_calculator.pb.h"

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

class EmbeddingsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

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
        std::vector<std::string> input_strings;
        bool isBase64 = false;
        if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request body: {}", payload.body);
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request uri: {}", payload.uri);
            if (!payload.parsedJson->IsObject())
                return absl::InvalidArgumentError("Received json is not an object");
            auto it = payload.parsedJson->FindMember("input");
            if (it != payload.parsedJson->MemberEnd()) {
                if (it->value.IsString()) {
                    input_strings.push_back(it->value.GetString());
                } else if (it->value.IsArray()) {
                    for (auto& input : it->value.GetArray()) {
                        if (!input.IsString())
                            return absl::InvalidArgumentError("every element in input array should be string");
                        input_strings.push_back(input.GetString());
                    }
                } else {
                    return absl::InvalidArgumentError("input should be string or array of strings");
                }
            } else {
                return absl::InvalidArgumentError("input field is required");
            }
            it = payload.parsedJson->FindMember("encoding_format");
            if (it != payload.parsedJson->MemberEnd()) {
                if (it->value.IsString()) {
                    if (it->value.GetString() == std::string("base64")) {
                        isBase64 = true;
                    }
                } else {
                    return absl::InvalidArgumentError("encoding_format should be string");
                }
            }
            // TODO: dimensions (optional)
            // TODO: user (optional)
        } else {
            return absl::InvalidArgumentError("Input is empty");
        }
        // Automatically deduce tokenizer input name
        std::vector<std::string> tokenizerInputNames = tokenizer_session->getInputNames();
        std::vector<std::string> embeddingsInputNames = embeddings_session->getInputNames();
        RET_CHECK(tokenizerInputNames.size() == 1);
        const std::string& tokenizerInputName = tokenizerInputNames.at(0);
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Tokenizer input name detected: {}", tokenizerInputName);

        ::InferenceInput tokenizerInputMap;
        tokenizerInputMap[tokenizerInputName] = ov::Tensor{
            ov::element::string,
            ov::Shape{input_strings.size()},
            input_strings.data()};
        ::InferenceOutput embeddingsOutputMap;
        try {
            ::InferenceOutput tokenizerOutputMap = tokenizer_session->infer(tokenizerInputMap);
            ::InferenceInput embeddingsInputMap;
            // Check if tokenizer produced at least the number of outputs as there are inputs in embedding model
            RET_CHECK(tokenizerOutputMap.size() >= embeddingsInputNames.size());
            for (const auto& embeddingsInputName : embeddingsInputNames) {
                auto it = tokenizerOutputMap.find(embeddingsInputName);
                RET_CHECK(it != tokenizerOutputMap.end());
                SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Embedding model input {} is connected with matching tokenizer output", embeddingsInputName);
                embeddingsInputMap[embeddingsInputName] = it->second;
            }
            embeddingsOutputMap = embeddings_session->infer(embeddingsInputMap);
            RET_CHECK(embeddingsOutputMap.size() > 0);
        } catch (const std::exception& e) {
            LOG(INFO) << "Caught exception from session infer():" << e.what();
            RET_CHECK(false);
        } catch (...) {
            LOG(INFO) << "Caught unknown exception from session infer()";
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
        RET_CHECK(embeddingsTensor.get_shape()[0] == input_strings.size());
        RET_CHECK(embeddingsTensor.get_element_type() == ov::element::f32);

        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        writer.StartObject();

        writer.String("object");
        writer.String("list");

        writer.String("data");
        writer.StartArray();
        const auto& options = cc->Options<EmbeddingsCalculatorOptions>();
        bool normalize = options.normalize_embeddings();
        // TODO: mean pooling

        ov::Shape outputShape = embeddingsTensor.get_shape();
        size_t batchSize = outputShape[0];
        for (size_t i = 0; i < batchSize; i++) {
            size_t stride = i * outputShape[1] * outputShape[2];
            std::vector<float> data(reinterpret_cast<float*>(embeddingsTensor.data()) + stride, reinterpret_cast<float*>(embeddingsTensor.data()) + stride + outputShape[2]);
            writer.StartObject();
            writer.String("object");
            writer.String("embedding");
            writer.String("embedding");
            if (normalize) {
                double square_sum = std::inner_product(data.begin(), data.end(), data.begin(), double(0.0));
                double denom = std::max(std::sqrt(square_sum), double(1e-12));
                std::transform(data.begin(), data.end(), data.begin(),
                    [denom](auto& element) { return element / denom; });
            }
            if (isBase64) {
                std::string_view sv(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
                std::string escaped;
                absl::Base64Escape(sv, &escaped);
                writer.String(escaped.c_str());
            } else {
                writer.StartArray();
                for (auto value : data) {
                    writer.Double(value);
                }
                writer.EndArray();
            }
            writer.String("index");
            writer.Int(i);
            writer.EndObject();
        }

        writer.EndArray();
        writer.EndObject();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string(buffer.GetString()), timestamp);
        return absl::OkStatus();
    }
};
const std::string EmbeddingsCalculator::INPUT_TAG_NAME{"REQUEST_PAYLOAD"};
const std::string EmbeddingsCalculator::OUTPUT_TAG_NAME{"RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(EmbeddingsCalculator);

}  // namespace mediapipe
