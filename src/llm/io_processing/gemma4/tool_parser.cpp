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
#include "tool_parser.hpp"
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

const int64_t Gemma4ToolParser::botTokenId = 48;
const int64_t Gemma4ToolParser::eotTokenId = 49;

std::string Gemma4ToolParser::parseArrayParameter(std::string argumentStr) {
    size_t pos = 1;
    std::string parsedArguments = "[";

    while (pos != std::string::npos) {
        size_t stringStartPos = argumentStr.find(TOOL_ARGS_STRING_INDICATOR, pos);
        if (stringStartPos == std::string::npos) {
            break;
        }
        stringStartPos += TOOL_ARGS_STRING_INDICATOR.size();
        size_t stringEndPos = argumentStr.find(TOOL_ARGS_STRING_INDICATOR, stringStartPos);
        if (stringEndPos == std::string::npos) {
            break;
        }

        std::string originalStr = argumentStr.substr(stringStartPos, stringEndPos - stringStartPos);
        size_t quotePos = 0;
        while ((quotePos = originalStr.find('\"', quotePos)) != std::string::npos) {
            originalStr.insert(quotePos, "\\");
            quotePos += 2;
        }
        parsedArguments += "\"" + originalStr + "\",";

        pos = stringEndPos + TOOL_ARGS_STRING_INDICATOR.size() + 1;
    }

    parsedArguments.back() = ']';

    return parsedArguments;
}

std::string Gemma4ToolParser::parseObjectParameter(std::string argumentStr) {
    size_t pos = 1;
    std::vector<std::pair<std::string, std::string>> keyValuePairs;

    while (pos != std::string::npos) {
        std::string key, value;
        bool isStringValue = false;
        size_t keyEndPos = argumentStr.find(':', pos);
        if (keyEndPos == std::string::npos) {
            break;
        }
        key = argumentStr.substr(pos, keyEndPos - pos);
        size_t valueStartPos = keyEndPos + 1;
        size_t valueEndPos;
        if (argumentStr.substr(valueStartPos, TOOL_ARGS_STRING_INDICATOR.size()) == TOOL_ARGS_STRING_INDICATOR) {
            valueStartPos = valueStartPos + TOOL_ARGS_STRING_INDICATOR.size();
            valueEndPos = argumentStr.find(TOOL_ARGS_STRING_INDICATOR, valueStartPos);
            isStringValue = true;
        } else {
            valueEndPos = argumentStr.find(',', valueStartPos);
        }

        if (valueEndPos == std::string::npos) {
            valueEndPos = argumentStr.size() - 1;
        }
        value = argumentStr.substr(valueStartPos, valueEndPos - valueStartPos);
        if (isStringValue) {
            value = "\"" + value + "\"";
        }
        keyValuePairs.emplace_back(key, value);
        if (valueEndPos == argumentStr.size() - 1) {
            break;
        } else if (isStringValue) {
            pos = valueEndPos + TOOL_ARGS_STRING_INDICATOR.size() + 1;
        } else {
            pos = valueEndPos + 1;
        }
    }

    if (keyValuePairs.empty()) {
        return argumentStr;
    }

    std::string parsedObject = "{";
    for (const auto& [key, value] : keyValuePairs) {
        parsedObject += "\"" + key + "\":" + value + ",";
    }
    parsedObject.back() = '}';
    return parsedObject;
}

std::string Gemma4ToolParser::normalizeArgStr(const std::string& arg) {
    if (arg.empty()) {
        return arg;
    }

    std::string normalized = arg;
    trim(normalized);
    std::string lower = normalized;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "true" || lower == "false" || lower == "null") {
        return lower;
    }

    const char first = normalized.front();
    const char last = normalized.back();
    if (first == '{' && last == '}') {
        normalized = parseObjectParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument contains is an object, changed it to correct JSON format. Modified string: {}", normalized);
    }

    if (first == '[' && last == ']' && normalized.find(TOOL_ARGS_STRING_INDICATOR) != std::string::npos) {
        normalized = parseArrayParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an array, normalized quotes for JSON parsing. Modified string: {}", normalized);
    }

    if (normalized.substr(0, TOOL_ARGS_STRING_INDICATOR.size()) == TOOL_ARGS_STRING_INDICATOR &&
        normalized.substr(normalized.size() - TOOL_ARGS_STRING_INDICATOR.size(), TOOL_ARGS_STRING_INDICATOR.size()) == TOOL_ARGS_STRING_INDICATOR) {
        normalized = "\"" + normalized.substr(TOOL_ARGS_STRING_INDICATOR.size(), normalized.size() - 2 * TOOL_ARGS_STRING_INDICATOR.size()) + "\"";
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is enclosed in string indicators, removed them for JSON parsing. Modified string: {}", normalized);
    }

    rapidjson::Document tempDoc;
    rapidjson::Value finalValue;
    tempDoc.Parse(normalized.c_str());
    if (tempDoc.HasParseError()) {
        auto errorCode = tempDoc.GetParseError();
        auto errorMessage = rapidjson::GetParseError_En(errorCode);
        size_t errorOffset = tempDoc.GetErrorOffset();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Failed to parse argument string as JSON. Argument string: {}, Error: {} Offset: {}", normalized, errorMessage, errorOffset);

        if (normalized.front() == '\"' && normalized.back() == '\"') {
            normalized = normalized.substr(1, normalized.size() - 2);
        }
        finalValue.SetString(normalized.c_str(), static_cast<rapidjson::SizeType>(normalized.size()), tempDoc.GetAllocator());
    } else {
        finalValue.CopyFrom(tempDoc, tempDoc.GetAllocator());
    }

    {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        finalValue.Accept(writer);
        normalized = buffer.GetString();
    }

    return normalized;
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
    return argument;
}

std::vector<std::pair<std::string, std::string>> Gemma4ToolParser::parseArguments(const std::string& argumentsStr) {
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> parsedArgs;

    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = findInStringRespectingSpecialChars(argumentsStr, TOOL_ARGS_SEPARATOR_STR, argPos);
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
    size_t pos = findInStringRespectingSpecialChars(this->streamingContent, TOOL_ARGS_END_INDICATOR, this->streamingPosition);
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
        this->streamingPosition = toolCallEndTagPos + TOOL_CALL_END_TAG.length();
        this->currentState = State::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected end of tool call at position: {}, returning to content state", toolCallEndTagPos);
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

rapidjson::Document Gemma4ToolParser::wrapDeltaContent(const std::string& content) {
    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Value deltaObj(rapidjson::kObjectType);
    deltaObj.AddMember("content", rapidjson::Value(content.c_str(), doc.GetAllocator()), doc.GetAllocator());
    doc.AddMember("delta", deltaObj, doc.GetAllocator());
    return doc;
}

rapidjson::Document Gemma4ToolParser::wrapDeltaArgs(const std::string& argsStr, int toolCallIndex) {
    rapidjson::Document doc(rapidjson::kObjectType);
    doc.AddMember("arguments", rapidjson::Value(argsStr.c_str(), doc.GetAllocator()), doc.GetAllocator());

    return BaseOutputParser::wrapDelta(doc, toolCallIndex);
}

std::optional<rapidjson::Document> Gemma4ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        return std::nullopt;
    }

    this->streamingContent += chunk;

    if (parseNewContent()) {
        if (this->currentState == State::ToolCallParameters) {
            return BaseOutputParser::wrapFirstDelta(this->toolCall.name, toolCallIndex);
        }
        if (this->currentState == State::ToolCallEnded) {
            return wrapDeltaArgs(this->toolCall.arguments, toolCallIndex);
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
            if (!content.empty()) {
                return wrapDeltaContent(content);
            }
        }
        if (this->currentState == State::AfterToolCall) {
            this->currentState = State::Content;
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        if ((this->currentState == State::ToolCallParameters || this->currentState == State::ToolCallEnded) && !this->toolCall.arguments.empty()) {
            return wrapDeltaArgs(this->toolCall.arguments, toolCallIndex);
        }

        if (this->currentState == State::Content && this->streamingPosition < this->streamingContent.size()) {
            auto content = this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();

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

void Gemma4ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;
    std::vector<std::pair<size_t, size_t>> toolCallPositions;
    size_t pos = 0;

    while (pos != std::string::npos) {
        size_t start, end;
        auto it = std::find(generatedTokens.begin() + pos, generatedTokens.end(), botTokenId);
        if (it != generatedTokens.end()) {
            start = std::distance(generatedTokens.begin(), it);
        } else {
            break;
        }
        auto itArgs = std::find(generatedTokens.begin() + start, generatedTokens.end(), eotTokenId);
        if (itArgs != generatedTokens.end()) {
            end = std::distance(generatedTokens.begin(), itArgs);
        } else {
            break;
        }

        std::string toolCallStr = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + start + 1, generatedTokens.begin() + end + 1), ov::AnyMap{ov::genai::skip_special_tokens(false)});
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool list string: {}", toolCallStr);

        while (!toolCallStr.empty()) {
            size_t nextToolPos = toolCallStr.find(TOOL_CALL_NAME_PREFIX, TOOL_CALL_NAME_PREFIX.length());
            size_t toolEndPos;
            if (nextToolPos == std::string::npos) {
                toolEndPos = toolCallStr.rfind(TOOL_ARGS_END_INDICATOR);
            } else {
                toolEndPos = nextToolPos - 1;
            }
            std::string singleTool;
            if (toolEndPos != std::string::npos) {
                singleTool = toolCallStr.substr(0, toolEndPos + TOOL_ARGS_END_INDICATOR.length());
                if (toolEndPos + TOOL_ARGS_END_INDICATOR.length() < toolCallStr.length()) {
                    toolCallStr = toolCallStr.substr(toolEndPos + TOOL_ARGS_END_INDICATOR.length());
                } else {
                    toolCallStr.clear();
                }
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed single tool string {}", singleTool);
            } else {
                break;
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "No more tool strings found in the decoded string: {}", toolCallStr);
            }

            if (!singleTool.empty()) {
                tools.push_back(singleTool);
            }
        }

        pos = end;
        toolCallPositions.emplace_back(start, end);
    }

    for (const std::string& tool : tools) {
        ToolCall toolCall;
        auto wasToolCallParsed = parseSingleToolCall(tool, toolCall);
        if (wasToolCallParsed) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool call - name: {}, args: {}", toolCall.name, toolCall.arguments);
            parsedOutput.toolCalls.push_back(toolCall);
        } else {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Failed to parse tool call from string: {}", tool);
        }
    }
    std::vector<int64_t> contentWithoutToolCalls = generatedTokens;
    for (auto it = toolCallPositions.rbegin(); it != toolCallPositions.rend(); ++it) {
        contentWithoutToolCalls.erase(contentWithoutToolCalls.begin() + it->first, contentWithoutToolCalls.begin() + it->second + 1);
    }
    parsedOutput.content = tokenizer.decode(contentWithoutToolCalls, ov::AnyMap{ov::genai::skip_special_tokens(true)});
}
}  // namespace ovms