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
#include "lfm2_tool_parser.hpp"
#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "src/port/rapidjson_document.hpp"
#include "rapidjson/error/en.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace ovms {

namespace {

// LFM2.5 assigns token ID 124905 to <|tool_call_start|>; LFM2 uses 10.
// (Token-ID resolution happens automatically via tokenIdStartTags.)

// Tool-call format delimiters shared by LFM2 and LFM2.5.
const std::string TOOL_LIST_START_INDICATOR = "[";
const std::string TOOL_LIST_END_INDICATOR = "]";
const std::string TOOL_ARGS_START_INDICATOR = "(";
const std::string TOOL_ARGS_END_INDICATOR = ")";
const std::string TOOL_SEPARATOR_STR = ", ";
// EOS token emitted by the LFM2.5 chat template after tool-call blocks.
const std::string EOS_TOKEN_STR = "<|im_end|>";

struct Argument {
    std::string name;
    std::string value;
};

// ---------------------------------------------------------------------------
// Argument-value normalisation helpers
// ---------------------------------------------------------------------------

std::string parseArrayParameter(std::string argumentStr) {
    int quoteDepth = 0;
    for (size_t i = 1; i < argumentStr.size() - 1; ++i) {
        if (argumentStr[i] != '\'')
            continue;
        bool isLastElement = (i == argumentStr.size() - 2);
        bool isFollowedByComma = !isLastElement && argumentStr[i + 1] == ',';
        if (quoteDepth == 0) {
            argumentStr[i] = '"';
            quoteDepth++;
        } else if (quoteDepth > 0 && (isFollowedByComma || isLastElement)) {
            argumentStr[i] = '"';
            quoteDepth--;
        }
    }
    return argumentStr;
}

std::string parseObjectParameter(std::string argumentStr) {
    int quoteDepth = 0;
    for (size_t i = 1; i < argumentStr.size() - 1; ++i) {
        if (argumentStr[i] != '\'')
            continue;
        bool isLastElement = (i == argumentStr.size() - 2);
        bool isFollowedByComma = !isLastElement && argumentStr[i + 1] == ',';
        bool isFollowedByColon = !isLastElement && argumentStr[i + 1] == ':';
        if (quoteDepth == 0) {
            argumentStr[i] = '"';
            quoteDepth++;
        } else if (quoteDepth > 0 && (isFollowedByComma || isLastElement || isFollowedByColon)) {
            argumentStr[i] = '"';
            quoteDepth--;
        }
    }
    return argumentStr;
}

std::string normalizeArgStr(const std::string& arg) {
    if (arg.empty())
        return arg;

    std::string normalized = arg;
    trim(normalized);
    std::string lower = normalized;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "true" || lower == "false" || lower == "null")
        return lower;

    const char first = normalized.front();
    const char last = normalized.back();
    if (first == '{' && last == '}') {
        normalized = parseObjectParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an object, replaced single quotes: {}", normalized);
    }
    if (first == '[' && last == ']') {
        normalized = parseArrayParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an array, normalised quotes: {}", normalized);
    }
    if (first == '\'' && last == '\'') {
        normalized[0] = '"';
        normalized[normalized.size() - 1] = '"';
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument enclosed in single quotes, replaced with double quotes: {}", normalized);
    }

    rapidjson::Document tempDoc;
    rapidjson::Value finalValue;
    tempDoc.Parse(normalized.c_str());
    if (tempDoc.HasParseError()) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument not valid JSON ({}), treating as string: {}",
            rapidjson::GetParseError_En(tempDoc.GetParseError()), normalized);
        if (first == '"' && last == '"')
            normalized = normalized.substr(1, normalized.size() - 2);
        finalValue.SetString(normalized.c_str(), static_cast<rapidjson::SizeType>(normalized.size()), tempDoc.GetAllocator());
    } else {
        finalValue.CopyFrom(tempDoc, tempDoc.GetAllocator());
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    finalValue.Accept(writer);
    return buffer.GetString();
}

void writeArgumentToWriter(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer) {
    std::string normalized = normalizeArgStr(arg);
    rapidjson::Document doc;
    doc.Parse(normalized.c_str());
    rapidjson::Value& argumentDoc = doc;
    writeArgumentOfAnyType(argumentDoc, writer);
}

Argument parseSingleArgument(const std::string& argumentStr) {
    Argument argument;
    size_t equalPos = argumentStr.find('=');
    if (equalPos != std::string::npos) {
        argument.name = argumentStr.substr(0, equalPos);
        argument.value = argumentStr.substr(equalPos + 1);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed argument - name: {}, value: {}", argument.name, argument.value);
    } else {
        argument.name = argumentStr;
        argument.value = "";
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument '{}' has no '='; value set to empty", argumentStr);
    }
    return argument;
}

std::vector<Argument> parseArguments(const std::string& argumentsStr) {
    std::vector<Argument> parsedArgs;
    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = findInStringRespectingSpecialChars(argumentsStr, TOOL_SEPARATOR_STR, argPos);
        if (commaPos == std::string::npos) {
            parsedArgs.push_back(parseSingleArgument(argumentsStr.substr(argPos)));
            break;
        }
        parsedArgs.push_back(parseSingleArgument(argumentsStr.substr(argPos, commaPos - argPos)));
        argPos = commaPos + TOOL_SEPARATOR_STR.length();
    }
    return parsedArgs;
}

// ---------------------------------------------------------------------------
// State-machine step functions
// ---------------------------------------------------------------------------

bool parseInContentState(const std::string& streamingContent, size_t& streamingPosition,
    Lfm2ParseState& currentState,
    const std::string& startTag, const std::string& endTag) {
    size_t startTagPos = streamingContent.find(startTag, streamingPosition);
    size_t endTagPos = streamingContent.find(endTag, streamingPosition);
    if (endTagPos != std::string::npos && startTagPos == std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected stray end tag at position: {}", endTagPos);
        streamingPosition = endTagPos + endTag.length();
        return false;
    }
    if (startTagPos != std::string::npos) {
        if (startTagPos > streamingPosition) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Content before tool-call start tag at position: {}", startTagPos);
            return true;
        }
        currentState = Lfm2ParseState::ToolCallStarted;
        streamingPosition = startTagPos + startTag.length();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected tool-call start at position: {}", startTagPos);
        return false;
    }
    return true;
}

bool parseInToolCallState(const std::string& streamingContent, ToolCall& toolCall,
    size_t& streamingPosition, Lfm2ParseState& currentState) {
    size_t toolListStartPos = streamingContent.find(TOOL_LIST_START_INDICATOR, streamingPosition);
    size_t argsPos = streamingContent.find(TOOL_ARGS_START_INDICATOR, streamingPosition);

    if (toolListStartPos != std::string::npos) {
        streamingPosition = toolListStartPos + TOOL_LIST_START_INDICATOR.length();
    } else if (argsPos != std::string::npos) {
        size_t bracketAnyPos = streamingContent.find(TOOL_LIST_START_INDICATOR);
        if (bracketAnyPos == std::string::npos || bracketAnyPos >= argsPos)
            return false;
    }

    if (argsPos == std::string::npos)
        return false;

    std::string toolName = streamingContent.substr(streamingPosition, argsPos - streamingPosition);
    trim(toolName);
    toolCall = ToolCall{generateRandomId(), toolName, ""};
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);
    streamingPosition = argsPos + TOOL_ARGS_START_INDICATOR.length();
    currentState = Lfm2ParseState::ToolCallParameters;
    return true;
}

bool parseInToolCallParametersState(const std::string& streamingContent, ToolCall& toolCall,
    size_t& streamingPosition, Lfm2ParseState& currentState) {
    size_t pos = findInStringRespectingSpecialChars(streamingContent, TOOL_ARGS_END_INDICATOR, streamingPosition);
    if (pos == std::string::npos)
        return false;

    std::string argumentsStr = streamingContent.substr(streamingPosition, pos - streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed arguments string: {}", argumentsStr);
    std::vector<Argument> arguments = parseArguments(argumentsStr);

    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
    argsWriter.StartObject();
    for (const Argument& argument : arguments) {
        argsWriter.Key(argument.name.c_str());
        writeArgumentToWriter(argument.value, argsWriter);
    }
    argsWriter.EndObject();
    toolCall.arguments = sb.GetString();
    currentState = Lfm2ParseState::ToolCallEnded;
    streamingPosition = pos + TOOL_ARGS_END_INDICATOR.length();
    return true;
}

bool parseInToolCallEndedState(const std::string& streamingContent, size_t& streamingPosition,
    Lfm2ParseState& currentState, const std::string& endTag) {
    size_t listEndPos = streamingContent.find(TOOL_LIST_END_INDICATOR, streamingPosition);
    size_t separatorPos = streamingContent.find(TOOL_SEPARATOR_STR, streamingPosition);
    size_t endTagPos = streamingContent.find(endTag, streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "ToolCallEnded: content from pos {}: {}",
        streamingPosition, streamingContent.substr(streamingPosition));
    if (listEndPos == std::string::npos && separatorPos == std::string::npos && endTagPos == std::string::npos)
        return false;
    if (separatorPos != std::string::npos && separatorPos < listEndPos) {
        streamingPosition = separatorPos + TOOL_SEPARATOR_STR.length();
        currentState = Lfm2ParseState::ToolCallStarted;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Tool-call separator at {}, expecting next call", separatorPos);
    } else if (endTagPos != std::string::npos) {
        streamingPosition = endTagPos + endTag.length();
        currentState = Lfm2ParseState::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "End tag at {}", endTagPos);
    } else {
        streamingPosition = listEndPos + TOOL_LIST_END_INDICATOR.length();
        currentState = Lfm2ParseState::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "End of tool list at {}", listEndPos);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Delta-wrapping helpers
// ---------------------------------------------------------------------------

ContentDelta wrapDeltaContent(const std::string& content) {
    return ContentDelta{content};
}

ToolCallDelta wrapDeltaArgs(const std::string& argsStr, int toolCallIndex) {
    return ToolCallDelta{toolCallIndex, std::nullopt, std::nullopt, argsStr};
}

void cutEOSFromContent(std::string& content) {
    size_t pos = content.find(EOS_TOKEN_STR);
    if (pos != std::string::npos)
        content = content.substr(0, pos);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Lfm2ToolParser implementation
// ---------------------------------------------------------------------------

bool Lfm2ToolParser::parseNewContent() {
    const std::string& startTag = parsingConfig.startTags[0];
    const std::string& endTag = parsingConfig.endTag;
    switch (this->currentState) {
    case Lfm2ParseState::Content:
        return parseInContentState(this->streamingContent, this->streamingPosition,
            this->currentState, startTag, endTag);
    case Lfm2ParseState::ToolCallStarted: {
        auto ok = parseInToolCallState(this->streamingContent, this->toolCall,
            this->streamingPosition, this->currentState);
        if (ok)
            this->toolCallIndex++;
        return ok;
    }
    case Lfm2ParseState::ToolCallParameters:
        return parseInToolCallParametersState(this->streamingContent, this->toolCall,
            this->streamingPosition, this->currentState);
    case Lfm2ParseState::ToolCallEnded:
        return parseInToolCallEndedState(this->streamingContent, this->streamingPosition,
            this->currentState, endTag);
    case Lfm2ParseState::AfterToolCall:
        break;
    }
    return false;
}

std::optional<Delta> Lfm2ToolParser::parseChunk(const std::string& chunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason finishReason) {
    // Empty chunks may arrive from the two-step streamer end() (NONE + empty STOP).
    // Skip them unless we have buffered state that still needs to be flushed.
    const bool hasPendingState = (this->currentState == Lfm2ParseState::ToolCallParameters) ||
                                 (this->currentState == Lfm2ParseState::ToolCallEnded);
    if (chunk.empty() && !hasPendingState)
        return std::nullopt;

    this->streamingContent += chunk;

    if (parseNewContent()) {
        if (this->currentState == Lfm2ParseState::ToolCallParameters) {
            return ToolCallDelta{this->toolCallIndex, generateRandomId(), this->toolCall.name, ""};
        }
        if (this->currentState == Lfm2ParseState::ToolCallEnded) {
            return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
        }
        if (this->currentState == Lfm2ParseState::Content) {
            const std::string& startTag = parsingConfig.startTags[0];
            size_t contentEnd = this->streamingContent.find(startTag, this->streamingPosition);
            std::string content = (contentEnd != std::string::npos)
                                      ? this->streamingContent.substr(this->streamingPosition, contentEnd - this->streamingPosition)
                                      : this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();
            cutEOSFromContent(content);
            if (!content.empty())
                return wrapDeltaContent(content);
        }
        if (this->currentState == Lfm2ParseState::AfterToolCall) {
            this->currentState = Lfm2ParseState::Content;
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        if ((this->currentState == Lfm2ParseState::ToolCallParameters ||
                this->currentState == Lfm2ParseState::ToolCallEnded) &&
            !this->toolCall.arguments.empty()) {
            return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
        }
        if (this->currentState == Lfm2ParseState::Content &&
            this->streamingPosition < this->streamingContent.size()) {
            auto content = this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();
            cutEOSFromContent(content);
            if (!content.empty())
                return wrapDeltaContent(content);
        }
    }

    return std::nullopt;
}

}  // namespace ovms
