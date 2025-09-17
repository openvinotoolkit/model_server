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
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../base_output_parser.hpp"

namespace ovms {
struct Functool {
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    //    std::unordered_set<std::string> usedParameterNames;
    void clear() {
        name.clear();
        parameters.clear();
        //    usedParameterNames.clear();
    }
};
struct Parser {
    enum State {
        Content,              // (C) here we expect either tools start tag or end of content
        InsideToolCall,       // (ITC) here we expect function start tag
        InsideFunctionName,   // (IFN) here we expect parameter start tag
        InsideFunction,       // (IF) here we expect parameter start tag
        InsideParameterName,  // (IPN) here we expect parameter end tag
        InsideParameter,      // (IP) here we expect parameter end tag
        AfterParameter,       // (AP) here we expect either next parameter or function & tools end
        ErrorEnd,             // (EE) we reached the end with error
        End                   // (E) we reached the end successfully
        // C->ITC->IFN->IF->IPN->IP->AP->(IPN|E)
        // all states beside C could reach EE
    };
    std::string& content;
    // string iterator to current position in content
    size_t currentPosition{0};
    State currentState = State::Content;
    Functool currentFunction;
    std::string currentParameterName;
    std::stack<size_t> toolsBeginStack;
    std::stack<size_t> toolsEndStack;
    bool removeNewlineAroundParameters = true;
    void removeToolCallsFromContent();
    /*
     * Returns true if step was successful, false if we reached the end or error occurred
     */
    bool step(ToolCalls& toolCalls);
    Parser(std::string& content) :
        content(content) {}
};

class Qwen3CoderToolParser : public BaseOutputParser {
public:
    static const std::string toolsStartTag;
    static const std::string toolsEndTag;
    static const std::string toolPrefixTag;
    static const std::string toolEndTag;
    static const std::string parameterPrefixTag;
    static const std::string parameterEndTag;
    static const std::string tagEnd;

protected:
    bool stripNewline = false;

    // ";" is used as a separator between tool calls in the response
    const std::string separator = ";";

    // Streaming required members
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    int toolCallIndex = -1;  // Index to track the current tool call being processed, -1 means we are not processing any tool call yet
    // Storing last two chunks of arguments to return delta with delay.
    // We do this to properly close arguments when tool call end tag is received.
    // With support for more models this could be moved to the base class.
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};
    int escapeLevel = 0;

public:
    Qwen3CoderToolParser() = delete;
    explicit Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

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
};
}  // namespace ovms
