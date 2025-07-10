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

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <optional>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "base_response_parser.hpp"

namespace ovms {
class Hermes3ResponseParser : public BaseResponseParser {
protected:
    // Tool calls are wrapped in <tool_call> and </tool_call> tags
    std::string toolCallStartTag = "<tool_call>";
    int64_t toolCallStartTokenId = 128002;  // This is the token ID for <tool_call> in Hermes3 tokenizer
    std::string toolCallEndTag = "</tool_call>";
    int64_t toolCallEndTokenId = 128013;  // This is the token ID for </tool_call> in Hermes3 tokenizer

public:
    Hermes3ResponseParser() = delete;
    explicit Hermes3ResponseParser(ov::genai::Tokenizer& tokenizer) :
        BaseResponseParser(tokenizer) {}

    ParsedResponse parse(const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk) override;
};
}  // namespace ovms
