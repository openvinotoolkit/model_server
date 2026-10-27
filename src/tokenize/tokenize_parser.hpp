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

#include <optional>
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
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#pragma warning(pop)

namespace ovms {

struct TokenizeRequest {
    // A plain string is the unbatched form. parseInput() returns it as such, and the request parsers turn it into
    // a vector of one string and clear inputIsBatch, so a parsed request's input never holds a std::string.
    using InputDataType = std::variant<std::string, std::vector<std::string>, std::vector<std::vector<int64_t>>, std::vector<std::vector<std::string>>>;
    InputDataType input;
    ov::AnyMap parameters = {};
    // True when "text" was given as an array of strings, even if the array holds a single string.
    bool inputIsBatch = false;
};

class TokenizeParser {
public:
    static constexpr const char* TOKENIZE_ENDPOINT_SUFFIX = "tokenize";
    static constexpr size_t MAX_LENGTH_LIMIT = 1000000;
    static absl::StatusOr<TokenizeRequest::InputDataType> parseInput(rapidjson::Document& parsedJson, const std::string& field_name);
    static absl::Status parseTokenizeResponse(rapidjson::StringBuffer& buffer, const ov::genai::TokenizedInputs& tokens, const ov::AnyMap& parameters = {}, bool inputIsBatch = false);
    static absl::Status parseTokenizeRequest(rapidjson::Document& parsedJson, TokenizeRequest& request, std::optional<size_t> maxModelLength = std::nullopt);
    static std::variant<TokenizeRequest, std::string> validateTokenizeRequest(rapidjson::Document& parsedJson, std::optional<size_t> maxModelLength = std::nullopt);
    static bool isTokenizeEndpoint(const std::string& uri);
};
}  // namespace ovms
