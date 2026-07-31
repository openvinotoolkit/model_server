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
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/apis/tool_schema_wrapper.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"

namespace ovms {

// Onyx (new drop) tool-call framing. The assistant turn is routed with the harmony
// envelope " to=<recipient><|message|>...{<|eom|>|<|eot|>}", and when the recipient is a
// function the body is an ATEM XML block -- Anthropic-style, structurally identical to
// qwen3coder's <tool_call>/<function=>/<parameter=> walk, just with "atem:" tags:
//
//    to=get_weather<|message|><atem:function_calls>
//   <atem:invoke name="get_weather">
//   <atem:parameter name="gps">37.7749,-122.4194</atem:parameter>
//   <atem:parameter name="time">2026-07-30T18:00:00Z</atem:parameter>
//   </atem:invoke>
//   </atem:function_calls><|eom|>
//
// The parser triggers on the fixed "<atem:function_calls>" marker (the "to=<recipient>"
// recipient is variable and must NOT be used as a trigger -- the model emits the BARE tool
// name, e.g. "to=get_weather", not "to=functions.get_weather"), and reads the authoritative
// name from <atem:invoke name="...">. The " to=<name><|message|>" prefix and the trailing
// terminator are stripped by OnyxReasoningParser (which runs first, see OutputParser::parse()).
//
// Parameter VALUES are rendered untyped/unquoted, so -- exactly like Qwen3CoderToolParser --
// each value is serialized into the JSON arguments blob according to the tool JSON schema
// (string->quoted, integer/number->numeric, bool/array/object->parsed, otherwise a
// best-effort JSON parse falling back to string).
using ParametersValues_t = std::map<std::string, std::string>;

struct OnyxFunctool {
    std::string name;
    rapidjson::Document argumentsAsDocument;
    OnyxFunctool() {
        argumentsAsDocument.SetObject();
    }
    void clear() {
        name.clear();
        argumentsAsDocument.SetObject();
    }
};

// Pure state machine that accumulates raw generated text and hands back fully assembled tool
// calls -- mirrors Qwen3CoderToolParserImpl. Holds the parameter-type map BY REFERENCE (bound
// either to the owning OnyxToolParser's lazily-filled map, or to a static empty map for the
// default constructor used by the direct-impl unit tests, where plain-string values need no
// schema to serialize correctly).
struct OnyxToolParserImpl {
    enum class State {
        Content,              // expect tool start tag or end of content
        InsideToolCall,       // after "<atem:function_calls>", expect "<atem:invoke name=\""
        InsideFunctionName,   // reading the invoke name, expect the closing "\">"
        InsideFunction,       // expect a "<atem:parameter name=\"" or the "</atem:invoke>" end
        InsideParameterName,  // reading a parameter name, expect the closing "\">"
        InsideParameter,      // reading a parameter value, expect "</atem:parameter>"
        AfterFunction         // after "</atem:invoke>", expect "</atem:function_calls>"
    };

    OnyxToolParserImpl();
    explicit OnyxToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap);

    // Return all tool calls fully closed ("</atem:function_calls>" seen) in the aggregated
    // content so far that were not returned before -- nullopt if none completed yet.
    std::optional<ToolCalls_t> parseChunk(const std::string& chunk);
    std::optional<std::string> getCurrentFunctionName() const;
    Status removeToolCallsFromContentIfNeeded(std::string& outContent);
    State getCurrentState() const {
        return this->currentState;
    }
    size_t getLastProcessedPosition() const {
        return this->lastProcessedPosition;
    }

private:
    const ToolsParameterTypeMap_t& toolsParametersTypeMap;
    // Onyx renders parameter values tight ("...\">VALUE</atem:parameter>"), so unlike
    // qwen3coder there is no surrounding-newline convention to trim.
    const bool removeNewlineAroundParameters = false;
    State currentState = State::Content;
    OnyxFunctool currentFunction;
    std::string currentParameterName;
    std::string streamContent;  // content accumulated from stream chunks
    size_t lastProcessedPosition{0};
    struct ToolCallPositions {
        std::stack<size_t> begin;
        std::stack<size_t> end;
    };
    ToolCallPositions toolCallPositions;

    void addParameterToCurrentFunctionDoc(std::string& parameterValueAsString);
    // Process streamContent from lastProcessedPosition until a state change happens; return
    // true if the state changed (keep looping), false once no more progress is possible.
    bool parseUntilStateChange(ToolCalls_t& toolCalls);
};

class OnyxToolParser : public BaseOutputParser {
public:
    static const std::string TOOL_START_TAG;      // "<atem:function_calls>"
    static const std::string TOOL_END_TAG;        // "</atem:function_calls>"
    static const std::string FUNCTION_NAME_TAG;   // "<atem:invoke name=\""
    static const std::string FUNCTION_END_TAG;    // "</atem:invoke>"
    static const std::string PARAMETER_NAME_TAG;  // "<atem:parameter name=\""
    static const std::string PARAMETER_END_TAG;   // "</atem:parameter>"
    static const std::string NAME_ATTR_END_TAG;   // "\">" -- closes an invoke/parameter name

private:
    const ToolsSchemas_t& toolSchemas;  // filled outside; kept as reference (may change)
    ToolsParameterTypeMap_t toolsParametersTypes;
    bool filledParametersTypesMap{false};
    OnyxToolParserImpl streamParser;
    int toolCallIndex{-1};
    std::set<int> returnedFirstDeltas;
    std::set<int> returnedCompleteDeltas;

    std::optional<rapidjson::Document> sendFirstDeltaIfNeeded(const std::string& functionName);
    std::optional<rapidjson::Document> sendFullDelta(const ToolCalls_t& toolCalls);
    void lazyFillInitToolParametersTypesMap();

public:
    OnyxToolParser() = delete;
    explicit OnyxToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas);

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> startTags{TOOL_START_TAG};
        return startTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        return TOOL_END_TAG;
    }
    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms

template <>
struct fmt::formatter<ovms::OnyxToolParserImpl::State> : fmt::formatter<std::string> {
    auto format(const ovms::OnyxToolParserImpl::State& state, fmt::format_context& ctx) const {
        std::unordered_map<ovms::OnyxToolParserImpl::State, std::string> stateMap = {
            {ovms::OnyxToolParserImpl::State::Content, "Content"},
            {ovms::OnyxToolParserImpl::State::InsideToolCall, "InsideToolCall"},
            {ovms::OnyxToolParserImpl::State::InsideFunctionName, "InsideFunctionName"},
            {ovms::OnyxToolParserImpl::State::InsideFunction, "InsideFunction"},
            {ovms::OnyxToolParserImpl::State::InsideParameterName, "InsideParameterName"},
            {ovms::OnyxToolParserImpl::State::InsideParameter, "InsideParameter"},
            {ovms::OnyxToolParserImpl::State::AfterFunction, "AfterFunction"}};
        auto it = stateMap.find(state);
        if (it != stateMap.end()) {
            return fmt::formatter<std::string>::format(it->second, ctx);
        } else {
            return fmt::formatter<std::string>::format("Unknown", ctx);
        }
    }
};
