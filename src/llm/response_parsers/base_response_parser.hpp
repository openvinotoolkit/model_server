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

#include "partial_json_builder.hpp"

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
    ProcessingPhase processingPhase = CONTENT;
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    int toolCallIndex = -1;  // Index to track the current tool call being processed, -1 means we are not processing any tool call yet
public:
    BaseResponseParser() = delete;
    explicit BaseResponseParser(ov::genai::Tokenizer& tokenizer) :
        tokenizer(tokenizer) {}
    virtual ~BaseResponseParser() = default;

    // Common function to wrap first delta with full function name in a JSON object that conforms to OpenAI API response format:
    // {"tool_calls":[{"id": <id>, "type": "function", "index":<index>,"function":<delta>}]}
    static rapidjson::Document wrapFirstDelta(const std::string& functionName, int toolCallIndex);
    // Common function to wrap subsequent deltas in a JSON object that conforms to OpenAI API response format
    // {"tool_calls":[{"index":0,"function":<delta>}]}
    static rapidjson::Document wrapDelta(const rapidjson::Document& delta, int toolCallIndex);

    virtual ParsedResponse parse(const std::vector<int64_t>& generatedTokens) = 0;
    // Parse model output chunk in the streaming mode. If in result of processing the chunk we cannot produce meaningful response, we return std::nullopt.
    // Otherwise we return a JSON object containing the delta that conforms to OpenAI API.
    virtual std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse) = 0;
};
}  // namespace ovms
