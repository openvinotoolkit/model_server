//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <map>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/apis/tool_schema_wrapper.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"

namespace ovms {

// MiniCPM5 tool call format (attribute-style XML):
// <function name="get_weather"><param name="city">Beijing</param><param name="unit">celsius</param></function>
//
// Multiple <function ...>...</function> blocks may appear concatenated.
// Reasoning (<think>...</think>) is stripped earlier in the chain by the reasoning parser,
// so this tool parser only deals with <function ...> blocks.

struct Minicpm5Functool {
    std::string name;
    void clear() {
        name.clear();
        argumentsAsDocument.SetObject();
    }
    rapidjson::Document argumentsAsDocument;
    Minicpm5Functool() {
        argumentsAsDocument.SetObject();
    }
};

struct Minicpm5ToolParserImpl {
    enum class State {
        Content,             // (C) looking for <function
        InsideFunctionName,  // (IFN) inside <function name="..." — reading attribute until >
        InsideFunction,      // (IF) inside function body — looking for <param or </function>
        InsideParamName,     // (IPN) inside <param name="..." — reading attribute until >
        InsideParam,         // (IP) inside param value — reading until </param>
    };
    // STATE DEMARKATION
    /*
    Content
    <function name="...">        <- InsideFunctionName reads up to >
    InsideFunction
    (<param name="...">          <- InsideParamName reads up to >
    InsideParam</param>InsideFunction)*
    </function>Content
    */

    explicit Minicpm5ToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap);

    /*
     * Feed a chunk; returns any completed tool calls found so far.
     */
    std::optional<ToolCalls_t> parseChunk(const std::string& chunk);

    std::optional<std::string> getCurrentFunctionName() const;

    Status removeToolCallsFromContentIfNeeded(std::string& outContent);

    State getCurrentState() const { return this->currentState; }
    size_t getLastProcessedPosition() const { return this->lastProcessedPosition; }

private:
    const ToolsParameterTypeMap_t& toolsParametersTypeMap;
    const bool removeNewlineAroundParameters = true;
    State currentState = State::Content;
    Minicpm5Functool currentFunction;
    std::string currentParameterName;
    std::string streamContent;
    size_t lastProcessedPosition{0};

    struct ToolCallPositions {
        std::stack<size_t> begin;
        std::stack<size_t> end;
    };
    ToolCallPositions toolCallPositions;

    void addParameterToCurrentFunctionDoc(std::string& parameterValueAsString);

    /*
     * Process streamContent from lastProcessedPosition until a state change.
     * Returns true if the state changed, false when no further progress is possible.
     * Dispatches to the per-state handlers below.
     */
    bool parseUntilStateChange(ToolCalls_t& toolCalls);

    // Per-state handlers. Each advances lastProcessedPosition and/or currentState as far
    // as the currently buffered streamContent allows. Extracted from parseUntilStateChange.
    void handleContentState();
    void handleInsideFunctionNameState();
    void handleInsideFunctionState(ToolCalls_t& toolCalls);
    void handleInsideParamNameState();
    void handleInsideParamState();

    /*
     * Extract the value of the name="..." (or name='...') attribute from an opening tag
     * that starts at tagStart (the '<') in streamContent.  The tag ends at the first '>'
     * after tagStart.  Returns the attribute value, or empty string on failure.
     * Sets *closingAngle (if non-null) to the position of the '>' that closes the tag.
     */
    static std::string extractNameAttribute(const std::string& content, size_t nameAttrValueStart, size_t tagEnd);
};

class Minicpm5ToolParser : public BaseOutputParser {
public:
    // Tag literals used by the state machine
    static const std::string FUNCTION_START_TAG;  // <function
    static const std::string NAME_ATTR_PREFIX;    // name="  (or name=')
    static const std::string XML_TAG_END;         // >
    static const std::string PARAM_START_TAG;     // <param
    static const std::string PARAM_END_TAG;       // </param>
    static const std::string FUNCTION_END_TAG;    // </function>

private:
    const ToolsSchemas_t& toolSchemas;
    ToolsParameterTypeMap_t toolsParametersTypes;
    bool filledParametersTypesMap{false};
    Minicpm5ToolParserImpl streamParser;
    int toolCallIndex{-1};
    ToolCalls_t currentToolCalls;
    rapidjson::Document currentJson;
    std::set<int> returnedFirstDeltas;
    std::set<int> returnedCompleteDeltas;

public:
    Minicpm5ToolParser() = delete;
    explicit Minicpm5ToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas);

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;

    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> startTags = {FUNCTION_START_TAG};
        return startTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags = {};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        static const std::string EMPTY_STRING = "";
        return EMPTY_STRING;
    }

private:
    std::optional<rapidjson::Document> sendFirstDeltaIfNeeded(const std::string& currentFunctionName);
    std::optional<rapidjson::Document> sendFullDelta(const ToolCalls_t& toolCalls);
    void lazyFillInitToolParametersTypesMap();
};

}  // namespace ovms

template <>
struct fmt::formatter<ovms::Minicpm5ToolParserImpl::State> : fmt::formatter<std::string> {
    auto format(const ovms::Minicpm5ToolParserImpl::State& state, fmt::format_context& ctx) const {
        std::unordered_map<ovms::Minicpm5ToolParserImpl::State, std::string> stateMap = {
            {ovms::Minicpm5ToolParserImpl::State::Content, "Content"},
            {ovms::Minicpm5ToolParserImpl::State::InsideFunctionName, "InsideFunctionName"},
            {ovms::Minicpm5ToolParserImpl::State::InsideFunction, "InsideFunction"},
            {ovms::Minicpm5ToolParserImpl::State::InsideParamName, "InsideParamName"},
            {ovms::Minicpm5ToolParserImpl::State::InsideParam, "InsideParam"},
        };
        auto it = stateMap.find(state);
        if (it != stateMap.end()) {
            return fmt::formatter<std::string>::format(it->second, ctx);
        } else {
            return fmt::formatter<std::string>::format("Unknown", ctx);
        }
    }
};
