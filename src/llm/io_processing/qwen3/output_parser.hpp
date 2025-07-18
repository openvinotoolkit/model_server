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
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../base_output_parser.hpp"

namespace ovms {

class Qwen3OutputParser : public BaseOutputParser {
protected:
    // Tool calls are wrapped in <tool_call> and </tool_call> tags
    std::string toolCallStartTag = "<tool_call>";
    int64_t toolCallStartTokenId = 151657;  // This is the token ID for <tool_call> in Qwen3 tokenizer
    std::string toolCallEndTag = "</tool_call>";
    int64_t toolCallEndTokenId = 151658;  // This is the token ID for </tool_call> in Qwen3 tokenizer

    std::string reasoningStartTag = "<think>";
    int64_t reasoningStartTokenId = 151667;  // This is the token ID for <think> in Qwen3 tokenizer
    std::string reasoningEndTag = "</think>";
    int64_t reasoningEndTokenId = 151668;  // This is the token ID for </think> in Qwen3 tokenizer

    // Storing last two chunks of arguments to return delta with delay.
    // We do this to properly close arguments when tool call end tag is received.
    // With support for more models this could be moved to the base class.
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};

public:
    Qwen3OutputParser() = delete;
    explicit Qwen3OutputParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    ParsedOutput parse(const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk) override;
};
}  // namespace ovms
