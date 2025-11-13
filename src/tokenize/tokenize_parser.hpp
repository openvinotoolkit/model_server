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
#pragma once

#include <string>
#include <variant>
#include <vector>

#include <openvino/runtime/tensor.hpp>
#include <openvino/genai/tokenizer.hpp>

#pragma warning(push)
#pragma warning(disable : 6001 6385 6386 6011 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

namespace ovms {

struct TokenizeRequest {
    using InputDataType = std::variant<std::vector<std::string>, std::vector<std::vector<int64_t>>, std::vector<std::vector<std::string>>>;
    InputDataType input;
    ov::AnyMap parameters = {};
};

class TokenizeParser {
public:
    static std::variant<TokenizeRequest::InputDataType, std::string> parseInput(rapidjson::Document& parsedJson, const std::string& field_name);
    static absl::Status parseTokenizeResponse(rapidjson::StringBuffer& buffer, const ov::genai::TokenizedInputs& tokens, const ov::AnyMap& parameters = {});
    static absl::Status parseTokenizeRequest(rapidjson::Document& parsedJson, TokenizeRequest& request);
    static std::variant<TokenizeRequest, std::string> validateTokenizeRequest(rapidjson::Document& parsedJson);
};
}  // namespace ovms
