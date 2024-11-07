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
#include <variant>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include <rapidjson/writer.h>

#include "absl/strings/escaping.h"
#include "rapidjson/document.h"

using namespace rapidjson;

std::variant<EmbeddingsRequest, std::string> EmbeddingsRequest::fromJson(rapidjson::Document* parsedJson) {
    EmbeddingsRequest request;
    std::vector<std::string> input_strings;

    if (!parsedJson->IsObject())
        return "Received json is not an object";

    auto it = parsedJson->FindMember("input");
    if (it != parsedJson->MemberEnd()) {
        if (it->value.IsString()) {
            input_strings.push_back(it->value.GetString());
        } else if (it->value.IsArray()) {
            for (auto& input : it->value.GetArray()) {
                // TODO: is array of ints
                // TODO: is int
                if (!input.IsString())
                    return "every element in input array should be string";
                input_strings.push_back(input.GetString());
            }
        } else {
            return "input should be string or array of strings";
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
    request.input = input_strings;
    return request;
}

absl::Status EmbeddingsHandler::parseRequest() {
    auto parsed = EmbeddingsRequest::fromJson(&(this->doc));

    if (auto error = std::get_if<std::string>(&parsed)) {
        return absl::InvalidArgumentError(*error);
    }
    this->request = std::get<EmbeddingsRequest>(parsed);
    return absl::OkStatus();
}

std::variant<std::vector<std::string>, std::vector<std::vector<int>>>& EmbeddingsHandler::getInput() {
    return request.input;
}
EncodingFormat EmbeddingsHandler::getEncodingFormat() const {
    return request.encoding_format;
}

void EmbeddingsHandler::setPromptTokensUsage(int promptTokens) {
    this->promptTokens = promptTokens;
}

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
    for (size_t i = 0; i < batchSize; i++) {
        size_t stride = i * outputShape[1] * outputShape[2];
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
        if (getEncodingFormat() == EncodingFormat::BASE64) {
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
        writer.Int(i);
        writer.EndObject();
    }

    writer.EndArray();

    writer.String("usage");
    writer.StartObject();
    writer.String("prompt_tokens");
    writer.Int(promptTokens);
    writer.String("total_tokens");
    writer.Int(promptTokens);
    writer.EndObject();

    writer.EndObject();
    return absl::OkStatus();
}
