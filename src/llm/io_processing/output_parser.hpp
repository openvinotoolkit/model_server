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

#include <memory>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>
#include <unordered_set>

#include "base_output_parser.hpp"
// FIXME
#include "../apis/openai_request.hpp"

namespace ovms {

class OutputParser {
    // Public types and enums
public:
    enum TagLookupStatus {
        NOT_FOUND,
        FOUND_COMPLETE,
        FOUND_INCOMPLETE
    };

    class StreamOutputCache {
        std::string buffer;

    public:
        TagLookupStatus lookupTag(const std::string& tag) const;
        TagLookupStatus lookupTags(const std::unordered_set<std::string>& tags) const;
        void add(const std::string& chunk);
        void clear();
        const std::string& getBuffer() const;
    };

    enum ProcessingPhase {
        UNKNOWN,
        CONTENT,
        REASONING,
        TOOL_CALLS_PROCESSING_TOOL,
        TOOL_CALLS_WAITING_FOR_TOOL
    };

private:
    ov::genai::Tokenizer tokenizer;
    std::unique_ptr<BaseOutputParser> toolParser = nullptr;       // Tool parser for extracting tool calls
    std::unique_ptr<BaseOutputParser> reasoningParser = nullptr;  // Reasoning parser for extracting reasoning content

    // Streaming related members
    ProcessingPhase processingPhase = UNKNOWN;
    StreamOutputCache streamOutputCache;

    // Parsing methods below read chunks from streamOutputCache hence no string argument is needed

    // Regular content parsing method does not require finishReason as content is always parsed
    rapidjson::Document parseContentChunk(ProcessingPhase newPhase = CONTENT);

    std::optional<rapidjson::Document> parseToolCallChunk(ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase = TOOL_CALLS_PROCESSING_TOOL);
    std::optional<rapidjson::Document> parseReasoningChunk(ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase = REASONING);

public:
    OutputParser() = delete;
    explicit OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName);

    bool isToolParserAvailable() const;
    bool isReasoningParserAvailable() const;

    void enableImmediateToolParsing();
    std::string getToolParserStartTag() const;

    // Parse model output in the unary mode. Returns ParsedOutput containing data extracted by internal parsers.
    ParsedOutput parse(const std::vector<int64_t>& generatedTokens, const bool toolsAvailable, const ToolsSchemas_t& toolNameSchemaMap);

    // Parse model output chunk in the steaming mode. Returns a JSON object containing the delta that conforms to OpenAI API
    // or nullopt if no response can be produced.
    std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason);

    bool requiresStreamingWithSpecialTokens() const {
        return (reasoningParser && reasoningParser->requiresStreamingWithSpecialTokens()) &&
               (toolParser && toolParser->requiresStreamingWithSpecialTokens());
    }
};
}  // namespace ovms
