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
#pragma once

#include <string>
#include <variant>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6001 6385 6386 6011)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include <openvino/runtime/tensor.hpp>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#pragma warning(pop)

namespace ovms {

struct EmbeddingsRequest {
    enum class EncodingFormat {
        FLOAT,
        BASE64
    };
    std::variant<std::vector<std::string>, std::vector<std::vector<int64_t>>> input;
    EncodingFormat encoding_format;

    static std::variant<EmbeddingsRequest, std::string> fromJson(rapidjson::Document* request);
};

class EmbeddingsHandler {
    rapidjson::Document& doc;
    EmbeddingsRequest request;
    size_t promptTokens = 0;

public:
    EmbeddingsHandler(rapidjson::Document& document) :
        doc(document) {}

    std::variant<std::vector<std::string>, std::vector<std::vector<int64_t>>>& getInput();
    EmbeddingsRequest::EncodingFormat getEncodingFormat() const;

    absl::Status parseRequest();
    absl::Status parseResponse(rapidjson::StringBuffer& buffer, const ov::Tensor& embeddingsTensor, const bool normalizeEmbeddings);
    void setPromptTokensUsage(int promptTokens);
};
}  // namespace ovms
