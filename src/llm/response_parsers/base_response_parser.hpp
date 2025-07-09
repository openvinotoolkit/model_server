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

namespace ovms {
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

using ToolCalls = std::vector<ToolCall>;

struct ParsedResponse {
    // Content without tool calls and reasoning
    std::string content;
    // Tool calls extracted from the response
    ToolCalls toolCalls;
    // Decoded reasoning from the response
    std::string reasoning;
    // Number of reasoning tokens in the response
    size_t reasoningTokenCount = 0;
};

// Enum used to track current processing phase, used in streaming mode
enum ProcessingPhase {
    CONTENT,
    REASONING,
    TOOL_CALLS
};

class BaseResponseParser {
protected:
    ov::genai::Tokenizer tokenizer;

public:
    BaseResponseParser() = delete;
    explicit BaseResponseParser(ov::genai::Tokenizer& tokenizer) :
        tokenizer(tokenizer) {}
    virtual ~BaseResponseParser() = default;
    virtual ParsedResponse parse(const std::vector<int64_t>& generatedTokens) = 0;
    virtual std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse) = 0;
};
}  // namespace ovms
