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

extern const std::string TOOL_LIST_START_INDICATOR;
extern const std::string TOOL_LIST_END_INDICATOR;
extern const std::string TOOL_ARGS_START_INDICATOR;
extern const std::string TOOL_ARGS_END_INDICATOR;
extern const std::string TOOL_SEPARATOR_STR;
extern const std::string EOS_TOKEN_STR;
extern const int TOOL_CALL_INDEX_START;

struct Argument {
    std::string name;
    std::string value;
};

enum class State {
    Content,
    ToolCallStarted,
    ToolCallParameters,
    ToolCallEnded,
    AfterToolCall
};

struct TagIds {
    std::string toolCallStartTag;
    std::string toolCallEndTag;
    int64_t toolCallStartTokenId;
    int64_t toolCallEndTokenId;
    std::optional<int64_t> reasoningStartTokenId = std::nullopt;
    std::optional<int64_t> reasoningEndTokenId = std::nullopt;
};

std::string parseArrayParameter(std::string argumentStr);
std::string parseObjectParameter(std::string argumentStr);
std::string normalizeArgStr(const std::string& arg);
void writeArgumentToWriter(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);
Argument parseSingleArgument(const std::string& argumentStr);
std::vector<Argument> parseArguments(const std::string& argumentsStr);
bool parseInContentState(const std::string& streamingContent, size_t& streamingPosition, State& currentState, const TagIds& tagIds);
bool parseInToolCallState(const std::string& streamingContent, ToolCall& toolCall, size_t& streamingPosition, State& currentState);
bool parseInToolCallParametersState(const std::string& streamingContent, ToolCall& toolCall, size_t& streamingPosition, State& currentState);
bool parseInToolCallEndedState(const std::string& streamingContent, size_t& streamingPosition, State& currentState, const std::string& toolCallEndTag);
rapidjson::Document wrapDeltaContent(const std::string& content);
rapidjson::Document wrapDeltaArgs(const std::string& argsStr, int toolCallIndex);
void cutEOSFromContent(std::string& content);
bool parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall);
void parseUnaryResponse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens, ov::genai::Tokenizer& tokenizer, const TagIds& tagIds);

}  // namespace ovms
