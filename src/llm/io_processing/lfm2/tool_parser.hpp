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
#include <string>
#include <vector>
#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {
class Lfm2ToolParser : public BaseOutputParser {
protected:
    static const std::string TOOL_CALL_START_TAG;
    static const std::string TOOL_CALL_END_TAG;
    static const std::string TOOL_RESPONSE_START_TAG;
    static const std::string TOOL_RESPONSE_END_TAG;

    static const std::string TOOL_LIST_START_INDICATOR;
    static const std::string TOOL_LIST_END_INDICATOR;
    static const std::string TOOL_ARGS_START_INDICATOR;
    static const std::string TOOL_ARGS_END_INDICATOR;
    static const std::string TOOL_SEPARATOR_STR;

    static constexpr size_t MAX_TOOL_CALLS = 100;
    static constexpr size_t MAX_TOOLS_PER_CALL = 100;

    enum class State {
        Content,             // Content -> ToolCallStarted (on TOOL_CALL_START_TAG)
        ToolCallStarted,     // ToolCallStarted -> ToolCallParameters (on TOOL_ARGS_START_INDICATOR, emits name)
        ToolCallParameters,  // ToolCallParameters -> ToolCallEnded (on TOOL_ARGS_END_INDICATOR, emits args)
        ToolCallEnded,       // ToolCallEnded -> ToolCallStarted (on separator) | AfterToolCall (on end tag/list end)
        AfterToolCall        // AfterToolCall -> Content
    };

public:
    struct Argument {
        std::string name;
        std::string value;
    };
    Lfm2ToolParser() = delete;
    explicit Lfm2ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags = {TOOL_CALL_START_TAG};
        return parsingStartTags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }

    const std::string& getParsingEndTag() const override {
        return TOOL_CALL_END_TAG;
    }

    static std::string normalizeArgStr(const std::string& arg);

private:
    void writeArgumentToWriter(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);

    Argument parseSingleArgument(const std::string& argumentStr);
    std::vector<Argument> parseArguments(const std::string& argumentsStr);

    bool parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall);
    bool parseNewContent();
    bool parseInContentState();
    bool parseInToolCallState();
    bool parseToolCallParametersState();
    bool parseInToolCallEndedState();

    rapidjson::Document wrapDeltaContent(const std::string& content);
    rapidjson::Document wrapDeltaArgs(const std::string& argsStr, int toolCallIndex);

    std::string streamingContent;
    size_t streamingPosition{0};
    State currentState{State::Content};
    ToolCall toolCall;
    int toolCallIndex{-1};
};
}  // namespace ovms
