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

#include <map>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/genai/tokenizer.hpp>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"

namespace ovms {
using ParametersValues_t = std::map<std::string, std::string>;
struct Functool {
    std::string name;
    ParametersValues_t parameters;
    void clear() {
        name.clear();
        parameters.clear();
    }
    std::string parametersToJson();
};
struct Qwen3CoderToolParserImpl {
    enum class State {
        Content,              // (C) here we expect either tools start tag or end of content
        InsideToolCall,       // (ITC) here we expect function start tag
        InsideFunctionName,   // (IFN) here we expect xml end tag
        InsideFunction,       // (IF) here we expect parameter start tag or function end tag
        InsideParameterName,  // (IPN) here we expect xml end tag
        InsideParameter,      // (IP) here we expect parameter end tag
        AfterFunction         // (AF) here we expect tool end tag
    };
    // Example request
    // <tool_call>
    // <function=GetWeather>
    // <parameter=Location>Gdansk, Pomorskie</parameter>
    // </function>
    // </tool_call>

    // STATE DEMARKATION
    /*
Content
<tool_call>
InsideToolCall
<function=InsideFunctionName>
InsideFunction
(<parameter=InsideParameterName>InsideParameter</parameter>InsideFunction)*
</function>AfterFunction
</tool_call>Content

State transitions
              /<<<<<<<<<<\
C->ITC->IFN->IF->IPN->IP->AF->C
              \>>>>>>>>>>/
*/
    Qwen3CoderToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap);
    /*
     * Return all tool calls found in agreggated content so far
     * that were not returned before
     */
    std::optional<ToolCalls> parseChunk(const std::string& chunk);
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
    const bool removeNewlineAroundParameters = true;
    State currentState = State::Content;
    Functool currentFunction;
    std::string currentParameterName;
    std::string streamContent;  // content accumulated from stream chunks
    // current position in content
    size_t lastProcessedPosition{0};
    // members required for unary for removing tool calls from content
    struct ToolCallPositions {
        std::stack<size_t> begin;
        std::stack<size_t> end;
    };
    ToolCallPositions toolCallPositions;
    /*
     * process streamContent from lastProcessedPosition until change of state
     * return true if state changed, false otherwise
     * false means no more state changes possible with current content
     */
    bool parseUntilStateChange(ToolCalls& toolCalls);
};

class Qwen3CoderToolParser : public BaseOutputParser {
public:
    static const std::string TOOL_START_TAG;
    static const std::string TOOL_END_TAG;
    static const std::string FUNCTION_NAME_TAG;
    static const std::string FUNCTION_END_TAG;
    static const std::string PARAMETER_NAME_TAG;
    static const std::string PARAMETER_END_TAG;
    static const std::string XML_TAG_END;

private:
    const ToolsSchemas_t& toolSchemas;  // we need to keep reference as this is not filled in OpenAIApiHandler during ToolParser creation, NOTE that its const here but it can change outside
    ToolsParameterTypeMap_t toolsParametersTypes;
    bool filledParametersTypesMap{false};
    // for streaming parsing we need to keep parser as a member
    Qwen3CoderToolParserImpl streamParser;
    int toolCallIndex{-1};
    ToolCalls currentToolCalls;
    rapidjson::Document currentJson;
    std::set<int> returnedFirstDeltas;
    std::set<int> returnedCompleteDeltas;

public:
    Qwen3CoderToolParser() = delete;
    explicit Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas);

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::string& getParsingStartTag() const override {
        return TOOL_START_TAG;
    }
    const std::unordered_set<std::string>& getSpecialParsingStartTags() const override {
        static const std::unordered_set<std::string> specialParsingStartTags = {};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        static const std::string EMPTY_STRING = "";
        return EMPTY_STRING;
    }

private:
    std::optional<rapidjson::Document> sendFirstDeltaIfNeeded(const std::string& currentFunctionName);
    std::optional<rapidjson::Document> sendFullDelta(std::optional<ToolCalls>& toolCallsOpt);
    void lazyFillInitToolParametersTypesMap();
};
}  // namespace ovms
template <>
struct fmt::formatter<ovms::Qwen3CoderToolParserImpl::State> : fmt::formatter<std::string> {
    auto format(const ovms::Qwen3CoderToolParserImpl::State& state, fmt::format_context& ctx) const {
        std::unordered_map<ovms::Qwen3CoderToolParserImpl::State, std::string> stateMap = {
            {ovms::Qwen3CoderToolParserImpl::State::Content, "Content"},
            {ovms::Qwen3CoderToolParserImpl::State::InsideToolCall, "InsideToolCall"},
            {ovms::Qwen3CoderToolParserImpl::State::InsideFunctionName, "InsideFunctionName"},
            {ovms::Qwen3CoderToolParserImpl::State::InsideFunction, "InsideFunction"},
            {ovms::Qwen3CoderToolParserImpl::State::InsideParameterName, "InsideParameterName"},
            {ovms::Qwen3CoderToolParserImpl::State::InsideParameter, "InsideParameter"},
            {ovms::Qwen3CoderToolParserImpl::State::AfterFunction, "AfterFunction"}};
        auto it = stateMap.find(state);
        if (it != stateMap.end()) {
            return fmt::formatter<std::string>::format(it->second, ctx);
        } else {
            return fmt::formatter<std::string>::format("Unknown", ctx);
        }
    }
};
