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

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_writer.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

using namespace rapidjson;

namespace ovms {

std::variant<EmbeddingsRequest, std::string> EmbeddingsRequest::fromJson(rapidjson::Document* parsedJson) {
    EmbeddingsRequest request;
    if (!parsedJson->IsObject())
        return "Received json is not an object";

    auto parsedInput = TokenizeParser::parseInput(*parsedJson, "input");

    if (std::holds_alternative<std::string>(parsedInput)) {
        return std::get<std::string>(parsedInput);
    } else {
        auto inputVariant = std::get<EmbeddingsRequest::InputDataType>(parsedInput);
        if (std::holds_alternative<std::vector<std::string>>(inputVariant)) {
            request.input = std::get<std::vector<std::string>>(inputVariant);
        } else if (std::holds_alternative<std::vector<std::vector<int64_t>>>(inputVariant)) {
            request.input = std::get<std::vector<std::vector<int64_t>>>(inputVariant);
        } else {
            return "input must be either array of strings or array of array of integers";
        }
    }

    auto it = parsedJson->FindMember("encoding_format");
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

TokenizeRequest::InputDataType& EmbeddingsHandler::getInput() {
    return request.input;
}
EmbeddingsRequest::EncodingFormat EmbeddingsHandler::getEncodingFormat() const {
    return request.encoding_format;
}
ov::AnyMap& EmbeddingsHandler::getParameters() {
    return request.parameters;
}

void EmbeddingsHandler::setPromptTokensUsage(int promptTokens) {
    this->promptTokens = promptTokens;
}

#pragma warning(push)
#pragma warning(disable : 4267)
absl::Status EmbeddingsHandler::parseResponse(StringBuffer& buffer, const ov::Tensor& embeddingsTensor) {
    Writer<StringBuffer> writer(buffer);
    writer.StartObject();

    writer.String("object");
    writer.String("list");

    writer.String("data");
    writer.StartArray();

    const float* last_hidden_state_data = embeddingsTensor.data<float>();

    std::vector<std::vector<float>> result;
    const auto shape = embeddingsTensor.get_shape();

    if (shape.size() != 2) {
        return absl::InvalidArgumentError("Invalid embeddings tensor shape");
    }

    const size_t batch_size = shape[0];
    const size_t hidden_size = shape[1];

    for (size_t batch = 0; batch < batch_size; batch++) {
        const auto batch_offset = batch * hidden_size;
        const float* batch_data = last_hidden_state_data + batch_offset;
        const std::vector<float> batch_result(batch_data, batch_data + hidden_size);
        result.push_back(batch_result);

        writer.StartObject();
        writer.String("object");
        writer.String("embedding");
        writer.String("embedding");
        if (getEncodingFormat() == EmbeddingsRequest::EncodingFormat::BASE64) {
            std::string_view sv2(reinterpret_cast<const char*>(batch_result.data()), batch_result.size() * sizeof(float));
            std::string escaped;
            absl::Base64Escape(sv2, &escaped);
            writer.String(escaped.c_str());
        } else {
            writer.StartArray();
            for (size_t i = 0; i < batch_result.size(); ++i) {
                writer.Double(batch_result[i]);
            }
            writer.EndArray();
        }
        writer.String("index");
        writer.Uint(batch);
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
