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
#include "base_response_parser.hpp"

namespace ovms {
struct StreamingState {
    bool isToolCallStarted = false;        // Indicates if the tool call has started
    bool isToolCallEnded = false;          // Indicates if the tool call has ended
    std::string toolCallContent;           // Content of the tool call
    std::string toolCallArgumentsContent;  // Arguments of the tool call
};

class Qwen3ResponseParser : public BaseResponseParser {
    StreamingState streamingState;  // State for streaming responses
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

public:
    Qwen3ResponseParser() = delete;
    explicit Qwen3ResponseParser(ov::genai::Tokenizer& tokenizer) :
        BaseResponseParser(tokenizer) {}

    ParsedResponse parse(const std::vector<int64_t>& generatedTokens) override;
    // TODO: move to interface when ready
    void parseChunk(const std::string& chunk);
};
}  // namespace ovms
