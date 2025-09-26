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
#include <openvino/genai/generation_handle.hpp>
#include <map>
#include <unordered_map>
#include <unordered_set>
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
    std::string arguments;  // JSON "{"a":1, "b":"SOME_STRING"}" TODO rename to know in context that's JSON
};

using ToolsSchemas_t = std::map<std::string, std::string>;
using ToolCalls = std::vector<ToolCall>;

struct ParsedOutput {
    // Content without tool calls and reasoning
    std::string content;
    // Tool calls extracted from the response
    ToolCalls toolCalls;
    // Decoded reasoning from the response
    std::string reasoning;
};

enum class ParameterType_t {
    STRING,
    NUMBER,
    BOOLEAN,
    ARRAY,
    OBJECT,
    UNKNOWN
};
using ParametersTypeMap_t = std::unordered_map<std::string, ParameterType_t>;          // param name -> param type
using ToolsParameterTypeMap_t = std::unordered_map<std::string, ParametersTypeMap_t>;  // tool name -> (param name -> param type)

class BaseOutputParser {
protected:
    ov::genai::Tokenizer tokenizer;
    // Flag indicating whether parsing start tag has been injected into the prompt
    // if true, parser should assume start tag already appeared and start parsing immediately
    bool immediateParsingEnabled = false;

public:
    BaseOutputParser() = delete;
    explicit BaseOutputParser(ov::genai::Tokenizer& tokenizer) :
        tokenizer(tokenizer) {}
    virtual ~BaseOutputParser() = default;

    // Common function to wrap first delta with full function name in a JSON object that conforms to OpenAI API response format:
    // {"tool_calls":[{"id": <id>, "type": "function", "index":<index>,"function":<delta>}]}
    static rapidjson::Document wrapFirstDelta(const std::string& functionName, int toolCallIndex);
    // Common function to wrap subsequent deltas in a JSON object that conforms to OpenAI API response format
    // {"tool_calls":[{"index":0,"function":<delta>}]}
    static rapidjson::Document wrapDelta(const rapidjson::Document& delta, int toolCallIndex);

    // Calling this method should put parser into immediate parsing mode where it starts parsing immediately, without seeking the start tag.
    void enableImmediateParsing();

    bool isImmediateParsingEnabled() const;

    // --- Specialized output parsers interface ---

    // Parse model output and extract relevant information to parsedOutput fields. Raw generated tokens are provided as an argument.
    // Additionally parsedOutput.content is already filled with decoded content when this method is called, enabling chain or parsing.
    // Parser is also responsible for removing extracted part from the parsedOutput.content if necessary.
    virtual void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) = 0;

    // Parse model output chunk in the streaming mode. If in result of processing the chunk we cannot produce meaningful response, we return std::nullopt.
    // Otherwise we return a JSON object containing the delta that conforms to OpenAI API.
    virtual std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) = 0;

    // Get the tag that marks the beginning of the segment that should be processed by the parser.
    // This method is used in streaming mode to determine if the parser should start processing the content.
    // If empty string is returned, it means that the parser will never start processing the content.
    virtual const std::string& getParsingStartTag() const = 0;

    // Get a vector of additional tags that mark beginning of the segment that should be processed by the parser.
    // These tags are considered only if they are the first output produced by the model.
    // In streaming mode it means that they are considered only in UNKNOWN phase.
    virtual const std::unordered_set<std::string>& getSpecialParsingStartTags() const = 0;

    // Get the tag that marks the end of the segment that should be processed by the parser.
    // This method is used in streaming mode to determine if the parser should stop processing the content.
    // If empty string is returned, it means that the parser will keep processing until the end of the content.
    virtual const std::string& getParsingEndTag() const = 0;

    // Indicates whether the parser requires special tokens to be present in the streaming output.
    // If true, the tokenizer used in the TextStreamer should be configured to not skip special tokens.
    // This is important for parsers that rely on special tokens to identify parsing boundaries or
    // specific segments of the output.
    virtual bool requiresStreamingWithSpecialTokens() const {
        return false;
    }
};
}  // namespace ovms
