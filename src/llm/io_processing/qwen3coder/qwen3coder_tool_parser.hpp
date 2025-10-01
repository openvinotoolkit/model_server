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
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

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
struct Functool {
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    void clear() {
        name.clear();
        parameters.clear();
    }
};
struct Parser {
    enum class State {
        Content,              // (C) here we expect either tools start tag or end of content
        InsideToolCall,       // (ITC) here we expect function start tag
        InsideFunctionName,   // (IFN) here we expect parameter start tag
        InsideFunction,       // (IF) here we expect parameter start tag
        InsideParameterName,  // (IPN) here we expect parameter end tag
        InsideParameter,      // (IP) here we expect parameter end tag
        AfterFunction         // (AF) here we expect either next parameter or function & tools end
                              /*
                      /<<<<<<<\
        C->ITC->IFN->IF->IPN->IP->AF->C
                      \>>>>>>>/
*/
    };
    std::string& content;
    std::string streamContent;  // content accumulated from stream chunks
    // string iterator to current position in content
    size_t currentPosition{0};
    size_t lastStreamProcessedPosition{0};
    State currentState = State::Content;
    Functool currentFunction;
    std::string currentParameterName;
    std::stack<size_t> toolsBeginStack;
    std::stack<size_t> toolsEndStack;
    const ToolsParameterTypeMap_t& toolsParametersTypeMap;
    bool removeNewlineAroundParameters = true;
    Status removeToolCallsFromContentIfNeeded(std::string& outContent);
    /*
     * Returns true if step was successful, false if we reached the end or error occurred
     */
    bool streamStepImpl(ToolCalls& toolCalls);
    std::optional<ToolCalls> streamStep(const std::string& chunk);
    Parser(std::string& content, const ToolsParameterTypeMap_t& toolsParametersTypeMap);
};
static std::string NULL_STRING_CONTENT = "";

class Qwen3CoderToolParser : public BaseOutputParser {
public:
    static const std::string toolsStartTag;
    static const std::string toolsEndTag;
    static const std::string toolPrefixTag;
    static const std::string toolEndTag;
    static const std::string parameterPrefixTag;
    static const std::string parameterEndTag;
    static const std::string tagEnd;

private:
    const ToolsSchemas_t& toolSchemas;             // we need to keep reference as this is not filled in OpenAIApiHandler during ToolParser creation, NOTE that its const here but it can change outside
    ToolsParameterTypeMap_t toolsParametersTypes;  // FIXME do it once per request
    bool filledParametersTypesMap{false};
    // streaming
    Parser streamParser;
    int toolCallIndex{-1};
    ToolCalls currentToolCalls;
    rapidjson::Document currentJson;
    std::set<int> returnedFirstDeltas;
    std::set<int> returnedCompleteDeltas;
    // streaming off

public:
    Qwen3CoderToolParser() = delete;
    explicit Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas);

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::string& getParsingStartTag() const override {
        return toolsStartTag;  // FIXME CHECK
    }
    const std::unordered_set<std::string>& getSpecialParsingStartTags() const override {
        static const std::unordered_set<std::string> specialParsingStartTags = {toolsStartTag};  // FIXME CHECK
        return specialParsingStartTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        return toolsEndTag;  // FIXME CHECK
    }

private:
    void lazyFillInitToolParamatersTypsMap();
};
}  // namespace ovms
template <>
struct fmt::formatter<ovms::Parser::State> : fmt::formatter<std::string> {
    auto format(const ovms::Parser::State& state, fmt::format_context& ctx) const {
        // use unordered_map
        std::unordered_map<ovms::Parser::State, std::string> stateMap = {
            {ovms::Parser::State::Content, "Content"},
            {ovms::Parser::State::InsideToolCall, "InsideToolCall"},
            {ovms::Parser::State::InsideFunctionName, "InsideFunctionName"},
            {ovms::Parser::State::InsideFunction, "InsideFunction"},
            {ovms::Parser::State::InsideParameterName, "InsideParameterName"},
            {ovms::Parser::State::InsideParameter, "InsideParameter"},
            {ovms::Parser::State::AfterFunction, "AfterFunction"}};
        auto it = stateMap.find(state);
        if (it != stateMap.end()) {
            return fmt::formatter<std::string>::format(it->second, ctx);
        } else {
            return fmt::formatter<std::string>::format("Unknown", ctx);
        }
    }
};
