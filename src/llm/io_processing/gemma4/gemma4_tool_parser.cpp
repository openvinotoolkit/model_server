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
#include "gemma4_tool_parser.hpp"
#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "rapidjson/error/en.h"
#include <algorithm>
#include <cctype>
#include <utility>

namespace ovms {

const std::string Gemma4ToolParser::TOOL_CALL_START_TAG = "<|tool_call>";
const std::string Gemma4ToolParser::TOOL_CALL_END_TAG = "<tool_call|>";
const std::string Gemma4ToolParser::TOOL_CALL_NAME_PREFIX = "call:";

const std::string Gemma4ToolParser::TOOL_ARGS_START_INDICATOR = "{";
const std::string Gemma4ToolParser::TOOL_ARGS_END_INDICATOR = "}";
const std::string Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR = "<|\"|>";
const std::string Gemma4ToolParser::TOOL_ARGS_SEPARATOR_STR = ",";

const std::string Gemma4ToolParser::TURN_END_TAG = "<turn|>";
const std::string Gemma4ToolParser::TOOL_RESPONSE_START_TAG = "<|tool_response>";

const int64_t Gemma4ToolParser::botTokenId = 48;  // <|tool_call>
const int64_t Gemma4ToolParser::eotTokenId = 49;  // <tool_call|>

const int64_t Gemma4ToolParser::reasoningTokenId = 100;     // <|channel>
const int64_t Gemma4ToolParser::reasoningEndTokenId = 101;  // <channel|>

std::string Gemma4ToolParser::parseArrayParameter(const std::string& argumentStr) {
    std::string body = argumentStr.substr(1, argumentStr.size() - 2);
    trim(body);
    if (body.empty()) {
        return "[]";
    }

    std::string parsedArray = "[";
    bool firstElement = true;
    for (const std::string& element : splitRespectingSpecialChars(body, TOOL_ARGS_SEPARATOR_STR, maskDelimitedStringValues(body, TOOL_ARGS_STRING_INDICATOR))) {
        if (!firstElement) {
            parsedArray += ",";
        }
        parsedArray += normalizeArgStr(element);
        firstElement = false;
    }
    parsedArray += "]";
    return parsedArray;
}

std::string Gemma4ToolParser::parseObjectParameter(const std::string& argumentStr) {
    std::string body = argumentStr.substr(1, argumentStr.size() - 2);
    trim(body);
    if (body.empty()) {
        return "{}";
    }

    std::string parsedObject = "{";
    bool firstMember = true;
    for (const std::string& member : splitRespectingSpecialChars(body, TOOL_ARGS_SEPARATOR_STR, maskDelimitedStringValues(body, TOOL_ARGS_STRING_INDICATOR))) {
        const std::string maskedMember = maskDelimitedStringValues(member, TOOL_ARGS_STRING_INDICATOR);
        size_t keyEndPos = findInStringRespectingSpecialChars(maskedMember, ":", 0);
        if (keyEndPos == std::string::npos) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Object member does not contain a key separator, leaving argument unchanged. Member: {}", member);
            return argumentStr;
        }
        std::string key = member.substr(0, keyEndPos);
        trim(key);
        if (isWrappedByDelimiter(key, TOOL_ARGS_STRING_INDICATOR)) {
            key = key.substr(TOOL_ARGS_STRING_INDICATOR.size(), key.size() - 2 * TOOL_ARGS_STRING_INDICATOR.size());
        }
        if (!firstMember) {
            parsedObject += ",";
        }
        parsedObject += escapeAsJsonString(key) + ":" + normalizeArgStr(member.substr(keyEndPos + 1));
        firstMember = false;
    }
    parsedObject += "}";
    return parsedObject;
}

std::string Gemma4ToolParser::normalizeArgStr(const std::string& arg) {
    std::string normalized = arg;
    trim(normalized);
    if (normalized.empty()) {
        return "\"\"";
    }

    std::string lower = normalized;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "true" || lower == "false" || lower == "null") {
        return lower;
    }

    // Build valid JSON out of the Gemma4 specific syntax before handing it over to rapidjson.
    if (isWrappedByDelimiter(normalized, TOOL_ARGS_STRING_INDICATOR)) {
        normalized = escapeAsJsonString(normalized.substr(TOOL_ARGS_STRING_INDICATOR.size(), normalized.size() - 2 * TOOL_ARGS_STRING_INDICATOR.size()));
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is a string, converted it to correct JSON format. Modified string: {}", normalized);
    } else if (normalized.front() == '{' && normalized.back() == '}') {
        normalized = parseObjectParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an object, converted it to correct JSON format. Modified string: {}", normalized);
    } else if (normalized.front() == '[' && normalized.back() == ']') {
        normalized = parseArrayParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an array, converted it to correct JSON format. Modified string: {}", normalized);
    }

    rapidjson::Document tempDoc;
    tempDoc.Parse(normalized.c_str());
    if (!tempDoc.HasParseError()) {
        return normalized;
    }

    auto errorCode = tempDoc.GetParseError();
    auto errorMessage = rapidjson::GetParseError_En(errorCode);
    size_t errorOffset = tempDoc.GetErrorOffset();
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Failed to parse argument string as JSON, falling back to string value. Argument string: {}, Error: {} Offset: {}", normalized, errorMessage, errorOffset);

    return escapeAsJsonString(arg);
}

void Gemma4ToolParser::writeArgumentToWriter(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer) {
    std::string normalized = normalizeArgStr(arg);

    rapidjson::Document doc;
    doc.Parse(normalized.c_str());

    rapidjson::Value& argumentDoc = doc;
    writeArgumentOfAnyType(argumentDoc, writer);
}

std::pair<std::string, std::string> Gemma4ToolParser::parseSingleArgument(const std::string& argumentStr) {
    std::pair<std::string, std::string> argument;

    size_t colonPos = argumentStr.find(':');
    if (colonPos != std::string::npos) {
        argument.first = argumentStr.substr(0, colonPos);
        std::string value = argumentStr.substr(colonPos + 1);
        argument.second = value;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed argument - name: {}, value: {}", argument.first, argument.second);
    } else {
        argument.first = argumentStr;
        argument.second = "";
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument string: {} does not contain ':', setting name as entire string and value as empty", argumentStr);
    }
    trim(argument.first);
    
    return argument;
}

std::vector<std::pair<std::string, std::string>> Gemma4ToolParser::parseArguments(const std::string& argumentsStr) {
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> parsedArgs;

    const std::string maskedArgumentsStr = maskDelimitedStringValues(argumentsStr, TOOL_ARGS_STRING_INDICATOR);
    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = findInStringRespectingSpecialChars(maskedArgumentsStr, TOOL_ARGS_SEPARATOR_STR, argPos);
        if (commaPos == std::string::npos) {
            auto remainingStr = argumentsStr.substr(argPos);
            args.push_back(remainingStr);
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "No more commas found, adding remaining argument string: {}", remainingStr);
            break;
        }
        std::string argStr = argumentsStr.substr(argPos, commaPos - argPos);
        args.push_back(argStr);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed argument string: {}", argStr);
        argPos = commaPos + TOOL_ARGS_SEPARATOR_STR.length();
    }

    for (const std::string& arg : args) {
        parsedArgs.push_back(parseSingleArgument(arg));
    }
    return parsedArgs;
}

bool Gemma4ToolParser::parseInContentState() {
    size_t toolCallStartTagPos = this->streamingContent.find(TOOL_CALL_START_TAG, this->streamingPosition);
    if (toolCallStartTagPos != std::string::npos) {
        if (toolCallStartTagPos > this->streamingPosition) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Content found before tool call start tag at position: {}", toolCallStartTagPos);
            return true;
        }
        this->streamingPosition = toolCallStartTagPos + TOOL_CALL_START_TAG.length();
        this->currentState = State::ToolCallStarted;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected start of tool call at position: {}", toolCallStartTagPos);
        return false;
    }

    return true;
}

bool Gemma4ToolParser::parseInToolCallState() {
    size_t argsPos = this->streamingContent.find(TOOL_ARGS_START_INDICATOR, this->streamingPosition);
    if (argsPos == std::string::npos) {
        return false;
    }

    size_t toolNameStart = this->streamingContent.find(TOOL_CALL_NAME_PREFIX, this->streamingPosition);
    if (toolNameStart != std::string::npos && toolNameStart < argsPos) {
        toolNameStart += TOOL_CALL_NAME_PREFIX.length();
    } else {
        toolNameStart = this->streamingPosition;
    }

    std::string toolName = this->streamingContent.substr(toolNameStart, argsPos - toolNameStart);
    trim(toolName);
    this->toolCall = ToolCall{generateRandomId(), toolName, ""};
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);
    this->streamingPosition = argsPos + TOOL_ARGS_START_INDICATOR.length();
    this->currentState = State::ToolCallParameters;
    this->toolCallIndex++;
    return true;
}

bool Gemma4ToolParser::parseToolCallParametersState() {
    if (this->streamingContent.back() == TOOL_ARGS_END_INDICATOR.back()) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Tool arguments end indicator found at the end of streaming content, attempting to parse arguments: {}", this->streamingContent.substr(this->streamingPosition));
    }
    const std::string maskedStreamingContent = maskDelimitedStringValues(this->streamingContent, TOOL_ARGS_STRING_INDICATOR);
    size_t pos = findInStringRespectingSpecialChars(maskedStreamingContent, TOOL_ARGS_END_INDICATOR, this->streamingPosition);
    if (pos == std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Tool arguments end indicator not found in streaming content starting from position: {}", this->streamingPosition);
        return false;
    }
    std::string argumentsStr = this->streamingContent.substr(this->streamingPosition, pos - this->streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed arguments string: {}", argumentsStr);
    std::vector<std::pair<std::string, std::string>> arguments = parseArguments(argumentsStr);

    rapidjson::Document argsDoc(rapidjson::kObjectType);
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
    argsWriter.StartObject();

    for (const std::pair<std::string, std::string>& argument : arguments) {
        argsWriter.Key(argument.first.c_str());
        writeArgumentToWriter(argument.second, argsWriter);
    }

    argsWriter.EndObject();
    this->toolCall.arguments = sb.GetString();
    this->currentState = State::ToolCallEnded;
    this->streamingPosition = pos + TOOL_ARGS_END_INDICATOR.length();

    return true;
}

bool Gemma4ToolParser::parseInToolCallEndedState() {
    size_t nextToolCallPos = this->streamingContent.find(TOOL_CALL_NAME_PREFIX, this->streamingPosition);
    size_t toolCallEndTagPos = this->streamingContent.find(TOOL_CALL_END_TAG, this->streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Current state: ToolCallEnded. Streaming content from current position: {}", this->streamingContent.substr(this->streamingPosition));
    if (nextToolCallPos != std::string::npos && nextToolCallPos < toolCallEndTagPos) {
        this->streamingPosition = nextToolCallPos;
        this->currentState = State::ToolCallStarted;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected next tool call at position: {}", nextToolCallPos);
    } else if (toolCallEndTagPos != std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected end of tool call at position: {}", toolCallEndTagPos);
        this->streamingPosition = toolCallEndTagPos + TOOL_CALL_END_TAG.length();
        this->currentState = State::AfterToolCall;
    } else {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Waiting for more data in ToolCallEnded state; no complete next tool call prefix or end tag found from position: {}", this->streamingPosition);
        return false;
    }
    return true;
}

bool Gemma4ToolParser::parseNewContent() {
    switch (this->currentState) {
    case State::Content: {
        return parseInContentState();
    }
    case State::ToolCallStarted: {
        return parseInToolCallState();
    }
    case State::ToolCallParameters: {
        return parseToolCallParametersState();
    }
    case State::ToolCallEnded: {
        return parseInToolCallEndedState();
    }
    case State::AfterToolCall:
        break;
    }
    return false;
}

std::optional<Delta> Gemma4ToolParser::wrapDeltaContent(const std::string& content) {
    if (content.empty())
        return std::nullopt;
    return ContentDelta{content};
}

ToolCallDelta Gemma4ToolParser::wrapDeltaArgs(const std::string& argsStr, int toolCallIndex) {
    return ToolCallDelta{toolCallIndex, std::nullopt, std::nullopt, argsStr};
}

std::optional<Delta> Gemma4ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (!chunk.empty()) {
        this->streamingContent += chunk;
    }

    if (parseNewContent()) {
        if (this->currentState == State::ToolCallParameters) {
            return ToolCallDelta{toolCallIndex, generateRandomId(), this->toolCall.name, ""};
        }
        if (this->currentState == State::ToolCallEnded) {
            auto delta = wrapDeltaArgs(this->toolCall.arguments, toolCallIndex);
            this->toolCall = ToolCall{};
            return delta;
        }
        if (this->currentState == State::Content) {
            size_t contentEnd = this->streamingContent.find(TOOL_CALL_START_TAG, this->streamingPosition);
            std::string content;
            if (contentEnd != std::string::npos) {
                content = this->streamingContent.substr(this->streamingPosition, contentEnd - this->streamingPosition);
            } else {
                content = this->streamingContent.substr(this->streamingPosition);
            }
            this->streamingPosition += content.size();

            // Structural/stop markers must never reach the client, on any chunk, not just the final flush.
            for (const std::string& tagToErase : {TURN_END_TAG, TOOL_RESPONSE_START_TAG}) {
                size_t tagPos = content.find(tagToErase);
                while (tagPos != std::string::npos) {
                    content.erase(tagPos, tagToErase.length());
                    tagPos = content.find(tagToErase, tagPos);
                }
            }

            return wrapDeltaContent(content);
        }
        if (this->currentState == State::AfterToolCall) {
            this->currentState = State::Content;
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        // Unary/STOP flush can arrive after a chunk that only advanced one state
        // (e.g. parsed the tool name but not yet the immediately following "}").
        // Give the state machine one last chance to consume already-buffered data
        // before deciding whether an arguments delta exists.
        if (this->currentState == State::ToolCallParameters) {
            parseToolCallParametersState();
        }

        if ((this->currentState == State::ToolCallParameters || this->currentState == State::ToolCallEnded) && !this->toolCall.arguments.empty()) {
            return wrapDeltaArgs(this->toolCall.arguments, toolCallIndex);
        }

        if (this->currentState == State::Content && this->streamingPosition < this->streamingContent.size()) {
            auto content = this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();

            for (const std::string& tagToErase : {TURN_END_TAG, TOOL_RESPONSE_START_TAG}) {
                size_t tagPos = content.find(tagToErase);
                while (tagPos != std::string::npos) {
                    content.erase(tagPos, tagToErase.length());
                    tagPos = content.find(tagToErase, tagPos);
                }
            }

            return wrapDeltaContent(content);
        }
    }

    return std::nullopt;
}

bool Gemma4ToolParser::parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall) {
    size_t argsPos = toolStr.find(TOOL_ARGS_START_INDICATOR);
    if (argsPos != std::string::npos) {
        std::string toolNameWithPrefix = toolStr.substr(0, argsPos);
        if (toolNameWithPrefix.find(TOOL_CALL_NAME_PREFIX) != 0) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Tool name does not start with expected prefix '{}'. Tool string: {}", TOOL_CALL_NAME_PREFIX, toolStr);
            return false;
        }
        std::string toolName = toolNameWithPrefix.substr(TOOL_CALL_NAME_PREFIX.length());
        trim(toolName);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);

        int argsStrLen = toolStr.length() - argsPos - TOOL_ARGS_START_INDICATOR.length() - TOOL_ARGS_END_INDICATOR.length();
        std::string argsStr = toolStr.substr(argsPos + TOOL_ARGS_START_INDICATOR.length(), argsStrLen);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed args string: {}", argsStr);
        std::vector<std::pair<std::string, std::string>> arguments = parseArguments(argsStr);

        toolCall.name = toolName;
        rapidjson::Document argsDoc(rapidjson::kObjectType);
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
        argsWriter.StartObject();
        for (const std::pair<std::string, std::string>& argument : arguments) {
            argsWriter.Key(argument.first.c_str());
            writeArgumentToWriter(argument.second, argsWriter);
        }
        argsWriter.EndObject();
        toolCall.arguments = sb.GetString();
        toolCall.id = generateRandomId();
        return true;
    }
    return false;
}

}  // namespace ovms
