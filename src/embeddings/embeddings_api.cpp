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

#include <variant>

#include "rapidjson/document.h"

std::variant<EmbeddingsRequest, std::string> EmbeddingsRequest::from_json(rapidjson::Document* parsedJson) {
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
