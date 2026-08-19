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
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "output_parsing_config.hpp"

#include "src/llm/apis/tool_schema_wrapper.hpp"

namespace ovms {
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON "{"a":1, "b":"SOME_STRING"}"
};

using ToolCalls_t = std::vector<ToolCall>;

struct ParsedOutput {
    // Content without tool calls and reasoning
    std::string content;
    // Tool calls extracted from the response
    ToolCalls_t toolCalls;
    // Decoded reasoning from the response
    std::string reasoning;
};

enum class ParameterType {
    STRING,
    NUMBER,
    BOOLEAN,
    ARRAY,
    OBJECT,
    UNKNOWN
};
using ParametersTypeMap_t = std::unordered_map<std::string, ParameterType>;            // param name -> param type
using ToolsParameterTypeMap_t = std::unordered_map<std::string, ParametersTypeMap_t>;  // tool name -> (param name -> param type)

// Tool-schema helpers shared between tag/attribute-style parsers (e.g. qwen3coder, minicpm5).
// Builds a parameter name -> ParameterType map from a single tool's JSON schema.
ParametersTypeMap_t parseToolSchema(const rapidjson::Value& schema);
// Builds a tool name -> (parameter name -> ParameterType) map from all tools' schemas.
ToolsParameterTypeMap_t createToolsParametersTypesMap(const ToolsSchemas_t& toolsSchemas);

class BaseOutputParser {
protected:
    ov::genai::Tokenizer tokenizer;

    // Parsing configuration set by sub-class constructors.
    OutputParsingConfig parsingConfig;

    // Token IDs resolved from parsingConfig.tokenIdStartTags on construction.
    // Maps token_id -> tag_string so the OutputParser can synthesise the boundary
    // text when a token-ID-based phase transition fires.
    std::unordered_map<int64_t, std::string> resolvedStartTokenToTag;

    // When true, the chat template has already emitted the parser's start tag as the
    // trailing tokens of the prompt, so the model output is expected to begin already
    // inside the parsed segment (e.g. reasoning) without producing the start tag itself.
    bool implicitStart = false;

    // Resolve tokenIdStartTags → resolvedStartTokenToTag using the tokenizer.
    // Called once from the constructor.
    void resolveSpecialTokenIds() {
        for (const auto& tag : parsingConfig.tokenIdStartTags) {
            if (tag.empty())
                continue;
            const auto tensor = tokenizer.encode(tag, ov::genai::add_special_tokens(false)).input_ids;
            if (tensor.get_size() == 1) {
                resolvedStartTokenToTag[tensor.data<int64_t>()[0]] = tag;
            }
        }
    }

public:
    BaseOutputParser() = delete;
    explicit BaseOutputParser(ov::genai::Tokenizer& tokenizer) :
        tokenizer(tokenizer) {}

    explicit BaseOutputParser(ov::genai::Tokenizer& tokenizer, OutputParsingConfig config) :
        tokenizer(tokenizer),
        parsingConfig(std::move(config)) {
        resolveSpecialTokenIds();
    }

    virtual ~BaseOutputParser() = default;

    virtual void resetState() {}

    void setImplicitStart(bool value) { implicitStart = value; }
    bool isImplicitStart() const { return implicitStart; }

    const OutputParsingConfig& getParsingConfig() const { return parsingConfig; }
    const std::unordered_map<int64_t, std::string>& getResolvedStartTokenToTag() const { return resolvedStartTokenToTag; }

    std::string buildParsingConfigStringRepresentation() const;
    // Common function to wrap first delta with full function name in a JSON object that conforms to OpenAI API response format:
    // {"tool_calls":[{"id": <id>, "type": "function", "index":<index>,"function":<delta>}]}
    static rapidjson::Document wrapFirstDelta(const std::string& functionName, int toolCallIndex);
    // Common function to wrap subsequent deltas in a JSON object that conforms to OpenAI API response format
    // {"tool_calls":[{"index":0,"function":<delta>}]}
    static rapidjson::Document wrapDelta(const rapidjson::Document& delta, int toolCallIndex);

    // --- Specialized output parsers interface ---

    virtual std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) = 0;
};
}  // namespace ovms
