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
#include <utility>
#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {
class Gemma4ToolParser : public BaseOutputParser {
protected:
    static const std::string TOOL_CALL_START_TAG;
    static const std::string TOOL_CALL_END_TAG;
    static const std::string TOOL_CALL_NAME_PREFIX;

    static const std::string TOOL_ARGS_START_INDICATOR;
    static const std::string TOOL_ARGS_END_INDICATOR;
    static const std::string TOOL_ARGS_STRING_INDICATOR;
    static const std::string TOOL_ARGS_SEPARATOR_STR;
    static const std::string TURN_END_TAG;

    static const int64_t botTokenId;
    static const int64_t eotTokenId;
    static const int64_t reasoningTokenId;
    static const int64_t reasoningEndTokenId;

    enum class State {
        Content,             // Content -> ToolCallStarted (on TOOL_CALL_START_TAG)
        ToolCallStarted,     // ToolCallStarted -> ToolCallParameters (on TOOL_ARGS_START_INDICATOR, emits name)
        ToolCallParameters,  // ToolCallParameters -> ToolCallEnded (on TOOL_ARGS_END_INDICATOR, emits args)
        ToolCallEnded,       // ToolCallEnded -> ToolCallStarted (on TOOL_CALL_NAME_PREFIX) | AfterToolCall (on end tag)
        AfterToolCall        // AfterToolCall -> Content
    };

public:
    Gemma4ToolParser() = delete;
    explicit Gemma4ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags = {TOOL_CALL_START_TAG};
        return parsingStartTags;
    }

    const std::vector<std::string>& getSpecialTagsToErase() const override {
        static const std::vector<std::string> tagsToErase = {TURN_END_TAG};
        return tagsToErase;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }

    const std::string& getParsingEndTag() const override {
        return TOOL_CALL_END_TAG;
    }

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }

    static std::string normalizeArgStr(const std::string& arg);
    static std::string parseArrayParameter(const std::string& argumentStr);
    static std::string parseObjectParameter(const std::string& argumentStr);

private:
    void writeArgumentToWriter(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);

    std::pair<std::string, std::string> parseSingleArgument(const std::string& argumentStr);
    std::vector<std::pair<std::string, std::string>> parseArguments(const std::string& argumentsStr);

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
