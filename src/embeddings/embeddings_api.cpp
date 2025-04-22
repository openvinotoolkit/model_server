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
#include "embeddings_api.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>

#include "../logging.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6386 6011 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6001 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

using namespace rapidjson;

namespace ovms {

std::variant<EmbeddingsRequest, std::string> EmbeddingsRequest::fromJson(rapidjson::Document* parsedJson) {
    enum class InputType {
        NONE,
        STRING,
        INT,
        INT_VEC
    };
    EmbeddingsRequest request;
    std::vector<std::string> input_strings;
    std::vector<std::vector<int64_t>> input_tokens;

    if (!parsedJson->IsObject())
        return "Received json is not an object";

    auto it = parsedJson->FindMember("input");
    if (it != parsedJson->MemberEnd()) {
        if (it->value.IsString()) {
            input_strings.push_back(it->value.GetString());
        } else if (it->value.IsArray()) {
            InputType input_type = InputType::NONE;
            for (auto& input : it->value.GetArray()) {
                if (input.IsArray()) {
                    if (input_type != InputType::NONE && input_type != InputType::INT_VEC)
                        return "input must be homogeneous";
                    input_type = InputType::INT_VEC;
                    std::vector<int64_t> ints;
                    ints.reserve(input.GetArray().Size());
                    for (auto& val : input.GetArray()) {
                        if (val.IsInt())
                            ints.push_back(val.GetInt());
                        else
                            return "input must be homogeneous";
                    }
                    input_tokens.emplace_back(std::move(ints));
                } else if (input.IsString()) {
                    if (input_type != InputType::NONE && input_type != InputType::STRING)
                        return "input must be homogeneous";
                    input_type = InputType::STRING;
                    input_strings.push_back(input.GetString());
                } else if (input.IsInt()) {
                    if (input_type != InputType::NONE && input_type != InputType::INT)
                        return "input must be homogeneous";
                    input_type = InputType::INT;
                    if (input_tokens.size() == 0) {
                        input_tokens.push_back(std::vector<int64_t>());
                    }
                    input_tokens[0].push_back(input.GetInt());
                } else {
                    return "every element in input array should be either string or int";
                }
            }
        } else {
            return "input should be string, array of strings or array of integers";
        }
    } else {
        return "input field is required";
    }

    it = parsedJson->FindMember("encoding_format");
    request.encoding_format = EncodingFormat::FLOAT;
    if (it != parsedJson->MemberEnd()) {
        if (it->value.IsString()) {
            if (it->value.GetString() == std::string("base64")) {
                request.encoding_format = EncodingFormat::BASE64;
            } else if (it->value.GetString() == std::string("float")) {
                request.encoding_format = EncodingFormat::FLOAT;
            } else {
                return "encoding_format should either base64 or float";
            }
        } else {
            return "encoding_format should be string";
        }
    }

    // TODO: dimensions (optional)
    // TODO: user (optional)
    if (input_strings.size() > 0) {
        request.input = input_strings;
    }
    if (input_tokens.size() > 0) {
        request.input = input_tokens;
    }
    return request;
}

absl::Status EmbeddingsHandler::parseRequest() {
    // Parsed JSON is not guaranteed to be valid, we may reach this point via multipart content type request with no valid JSON parser
    if (this->doc.HasParseError()) {
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Non-json request received in embeddings calculator");
        return absl::InvalidArgumentError("Non-json request received in embeddings calculator");
    }

    auto parsed = EmbeddingsRequest::fromJson(&(this->doc));

    if (auto error = std::get_if<std::string>(&parsed)) {
        return absl::InvalidArgumentError(*error);
    }
    this->request = std::get<EmbeddingsRequest>(parsed);
    return absl::OkStatus();
}

std::variant<std::vector<std::string>, std::vector<std::vector<int64_t>>>& EmbeddingsHandler::getInput() {
    return request.input;
}
EmbeddingsRequest::EncodingFormat EmbeddingsHandler::getEncodingFormat() const {
    return request.encoding_format;
}

void EmbeddingsHandler::setPromptTokensUsage(int promptTokens) {
    this->promptTokens = promptTokens;
}

#pragma warning(push)
#pragma warning(disable : 4267)
absl::Status EmbeddingsHandler::parseResponse(StringBuffer& buffer, const ov::Tensor& embeddingsTensor, const bool normalizeEmbeddings) {
    Writer<StringBuffer> writer(buffer);
    writer.StartObject();

    writer.String("object");
    writer.String("list");

    writer.String("data");
    writer.StartArray();
    // TODO: mean pooling

    ov::Shape outputShape = embeddingsTensor.get_shape();
    if (outputShape.size() != 3) {
        return absl::InvalidArgumentError("Invalid embeddings tensor shape");
    }
    size_t batchSize = outputShape[0];
    for (size_t batchIterator = 0; batchIterator < batchSize; batchIterator++) {
        size_t stride = batchIterator * outputShape[1] * outputShape[2];
        size_t size = outputShape[2];
        float* dataPtr = reinterpret_cast<float*>(embeddingsTensor.data()) + stride;
        float* dataPtrEnd = dataPtr + size;
        writer.StartObject();
        writer.String("object");
        writer.String("embedding");
        writer.String("embedding");
        if (normalizeEmbeddings) {
            double square_sum = std::inner_product(dataPtr, dataPtrEnd, dataPtr, double(0.0));
            double denom = std::max(std::sqrt(square_sum), double(1e-12));
            std::transform(dataPtr, dataPtrEnd, dataPtr,
                [denom](auto& element) { return element / denom; });
        }
        if (getEncodingFormat() == EmbeddingsRequest::EncodingFormat::BASE64) {
            std::string_view sv2(reinterpret_cast<char*>(dataPtr), outputShape[2] * sizeof(float));
            std::string escaped;
            absl::Base64Escape(sv2, &escaped);
            writer.String(escaped.c_str());
        } else {
            writer.StartArray();
            for (size_t i = 0; i < size; ++i) {
                writer.Double(dataPtr[i]);
            }
            writer.EndArray();
        }
        writer.String("index");
        writer.Uint(batchIterator);
        writer.EndObject();
    }

    writer.EndArray();

    writer.String("usage");
    writer.StartObject();
    writer.String("prompt_tokens");
    writer.Uint(promptTokens);
    writer.String("total_tokens");
    writer.Uint(promptTokens);
    writer.EndObject();

    writer.EndObject();
    return absl::OkStatus();
}
#pragma warning(pop)
}  // namespace ovms
