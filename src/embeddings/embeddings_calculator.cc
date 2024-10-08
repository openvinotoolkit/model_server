// TODO: move to separate target

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
#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#pragma GCC diagnostic pop

#include <adapters/inference_adapter.h>
#include <fmt/ranges.h>
#include <openvino/openvino.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../llm/http_payload.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
#include "absl/strings/escaping.h"

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

class EmbeddingsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;

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
            std::string response = "";
            InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request body: {}", payload.body);
            SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Request uri: {}", payload.uri);
            if (!payload.parsedJson->IsObject())
                return absl::InvalidArgumentError("Received json is not an object");
            auto it = payload.parsedJson->FindMember("input");
            if (it != payload.parsedJson->MemberEnd()) {
                if (it->value.IsString()) {
                    response += it->value.GetString();
                    input_strings.push_back(it->value.GetString());
                } else if (it->value.IsArray()) {
                    for (auto& input : it->value.GetArray()) {
                        if (!input.IsString())
                            return absl::InvalidArgumentError("every element in input array should be string");
                        response += input.GetString();
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
                    response += it->value.GetString();
                } else {
                    return absl::InvalidArgumentError("encoding_format should be string");
                }
            }
            it = payload.parsedJson->FindMember("dimensions");
            if (it != payload.parsedJson->MemberEnd()) {
                if (it->value.IsInt()) {
                    response += std::to_string(it->value.GetInt());
                } else {
                    return absl::InvalidArgumentError("dimensions should be string or array of strings");
                }
            }
            it = payload.parsedJson->FindMember("user");
            if (it != payload.parsedJson->MemberEnd()) {
                if (it->value.IsString()) {
                    response += it->value.GetString();
                } else {
                    return absl::InvalidArgumentError("user should be string");
                }
            }
        } else {
            return absl::InvalidArgumentError("Input is empty");
        }

        // Automatically deduce tokenizer input name
        std::vector<std::string> tokenizerInputNames = tokenizer_session->getInputNames();
        std::vector<std::string> embeddingsInputNames = embeddings_session->getInputNames();
        RET_CHECK(tokenizerInputNames.size() == 1);
        const std::string& tokenizerInputName = tokenizerInputNames.at(0);
        LOG(INFO) << "Tokenizer input name detected: " << tokenizerInputName;

        ::InferenceInput tokenizerInputMap;
        tokenizerInputMap[tokenizerInputName] = ov::Tensor{
            ov::element::string,
            ov::Shape{input_strings.size()},
            input_strings.data()};

        ::InferenceOutput embeddingsOutputMap;
        try {
            ::InferenceOutput tokenizerOutputMap = tokenizer_session->infer(tokenizerInputMap);
            ::InferenceInput embeddingsInputMap;
            RET_CHECK(tokenizerOutputMap.size() >= embeddingsInputNames.size());
            for (const auto& embeddingsInputName : embeddingsInputNames) {
                auto it = tokenizerOutputMap.find(embeddingsInputName);
                RET_CHECK(it != tokenizerOutputMap.end());
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

        // TODO (bstrzele) (dtrawins): Support both kinds of models with 1 or 2 outputs?
        // TODO: At the moment hardcoded for 1 output only supporting bge-large-en-v1.5, but will not work with rerank/other models
        // TODO: Must require last_hidden_state kind of tensor

        ov::Tensor lastHiddenStateTensor; 
        if (embeddingsOutputMap.size() == 2) { // GTE
            // Search by number of dimensions, should be 3
            bool found = false;
            for (const auto& [name, tensor] : embeddingsOutputMap) {
                if (tensor.get_shape().size() == 3) {
                    lastHiddenStateTensor = tensor;
                    found = true;
                    break;
                } 
            }
            RET_CHECK(found);
        } else { // BGE
            RET_CHECK(embeddingsOutputMap.size() == 1);
            lastHiddenStateTensor = embeddingsOutputMap.begin()->second;
            LOG(INFO) << "Embedding output name detected automatically: " << embeddingsOutputMap.begin()->first;
        }

        RET_CHECK(lastHiddenStateTensor.get_shape().size() == 3);
        LOG(INFO) << lastHiddenStateTensor.get_shape();
        // TODO: Batch size must be equal to number of input strings
        RET_CHECK(lastHiddenStateTensor.get_shape()[0] == input_strings.size());
        RET_CHECK(lastHiddenStateTensor.get_element_type() == ov::element::f32);

        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        writer.StartObject();

        writer.String("object");
        writer.String("list");

        writer.String("data");
        writer.StartArray();
        bool normalize = true;
        // TODO: mean pooling

        ov::Shape output_shape = lastHiddenStateTensor.get_shape();
        for (int i = 0; i < output_shape[0]; i++) {
            size_t stride = i * output_shape[1] * output_shape[2];
            std::vector<float> data(reinterpret_cast<float*>(lastHiddenStateTensor.data()) + stride, reinterpret_cast<float*>(lastHiddenStateTensor.data()) + stride + output_shape[2]);
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
