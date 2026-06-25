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
#include "lfm2_utils.hpp"
#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "rapidjson/error/en.h"
#include <algorithm>
#include <cctype>
#include <utility>

namespace ovms {
const std::string TOOL_LIST_START_INDICATOR = "[";
const std::string TOOL_LIST_END_INDICATOR = "]";
const std::string TOOL_ARGS_START_INDICATOR = "(";
const std::string TOOL_ARGS_END_INDICATOR = ")";
const std::string TOOL_SEPARATOR_STR = ", ";
const std::string EOS_TOKEN_STR = "<|im_end|>";

const int TOOL_CALL_INDEX_START = -1;



std::string parseArrayParameter(std::string argumentStr) {
    int quoteDepth = 0;

    for (size_t i = 1; i < argumentStr.size() - 1; ++i) {
        if (argumentStr[i] != '\'') {
            continue;
        }

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
        if (argumentStr[i] != '\'') {
            continue;
        }

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
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument contains is an object, replaced single quotes with double quotes for JSON parsing. Modified string: {}", normalized);
    }

    if (first == '[' && last == ']') {
        normalized = parseArrayParameter(normalized);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is an array, normalized quotes for JSON parsing. Modified string: {}", normalized);
    }

    if ((first == '\'' && last == '\'')) {
        normalized[0] = '"';
        normalized[normalized.size() - 1] = '"';
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument is enclosed in quotes, replaced outer quotes with double quotes for JSON parsing. Modified string: {}", normalized);
    }

    rapidjson::Document tempDoc;
    rapidjson::Value finalValue;
    tempDoc.Parse(normalized.c_str());
    if (tempDoc.HasParseError()) {
        auto errorCode = tempDoc.GetParseError();
        auto errorMessage = rapidjson::GetParseError_En(errorCode);
        size_t errorOffset = tempDoc.GetErrorOffset();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Failed to parse argument string as JSON. Argument string: {}, Error: {} Offset: {}", normalized, errorMessage, errorOffset);

        if (first == '\"' && last == '\"') {
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
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Argument string: {} does not contain '=', setting name as entire string and value as empty", argumentStr);
    }
    return argument;
}

std::vector<Argument> parseArguments(const std::string& argumentsStr) {
    std::vector<std::string> args;
    std::vector<Argument> parsedArgs;

    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = findInStringRespectingSpecialChars(argumentsStr, TOOL_SEPARATOR_STR, argPos);
        if (commaPos == std::string::npos) {
            auto remainingStr = argumentsStr.substr(argPos);
            args.push_back(remainingStr);
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "No more commas found, adding remaining argument string: {}", remainingStr);
            break;
        }
        auto argStr = argumentsStr.substr(argPos, commaPos - argPos);
        args.push_back(argStr);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed argument string: {}", argStr);
        argPos = commaPos + TOOL_SEPARATOR_STR.length();
    }

    for (const std::string& arg : args) {
        parsedArgs.push_back(parseSingleArgument(arg));
    }
    return parsedArgs;
}

bool parseInContentState(const std::string& streamingContent, size_t& streamingPosition, State& currentState, const std::string& toolCallStartTag, const std::string& toolCallEndTag) {
    size_t toolCallStartTagPos = streamingContent.find(toolCallStartTag, streamingPosition);
    size_t toolCallEndTagPos = streamingContent.find(toolCallEndTag, streamingPosition);
    if (toolCallEndTagPos != std::string::npos && toolCallStartTagPos == std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected end of tool call at position: {}", toolCallEndTagPos);
        streamingPosition = toolCallEndTagPos + toolCallEndTag.length();
        return false;
    }
    if (toolCallStartTagPos != std::string::npos) {
        if (toolCallStartTagPos > streamingPosition) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Content found before tool call start tag at position: {}", toolCallStartTagPos);
            return true;
        }
        currentState = State::ToolCallStarted;
        streamingPosition = toolCallStartTagPos + toolCallStartTag.length();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected start of tool call at position: {}", toolCallStartTagPos);
        return false;
    }

    return true;
}
bool parseInToolCallState(const std::string& streamingContent, ToolCall& toolCall, size_t& streamingPosition, State& currentState) {
    size_t toolListStartPos = streamingContent.find(TOOL_LIST_START_INDICATOR, streamingPosition);
    size_t argsPos = streamingContent.find(TOOL_ARGS_START_INDICATOR, streamingPosition);

    if (toolListStartPos != std::string::npos) {
        streamingPosition = toolListStartPos + TOOL_LIST_START_INDICATOR.length();
    }

    if (argsPos == std::string::npos) {
        return false;
    }

    std::string toolName = streamingContent.substr(streamingPosition, argsPos - streamingPosition);
    trim(toolName);
    toolCall = ToolCall{generateRandomId(), toolName, ""};
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);
    streamingPosition = argsPos + TOOL_ARGS_START_INDICATOR.length();
    currentState = State::ToolCallParameters;
    return true;
}

bool parseToolCallParametersState(const std::string& streamingContent, ToolCall& toolCall, size_t& streamingPosition, State& currentState) {
    size_t pos = findInStringRespectingSpecialChars(streamingContent, TOOL_ARGS_END_INDICATOR, streamingPosition);
    if (pos == std::string::npos) {
        return false;
    }
    std::string argumentsStr = streamingContent.substr(streamingPosition, pos - streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed arguments string: {}", argumentsStr);
    std::vector<Argument> arguments = parseArguments(argumentsStr);

    rapidjson::Document argsDoc(rapidjson::kObjectType);
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
    argsWriter.StartObject();

    for (const Argument& argument : arguments) {
        argsWriter.Key(argument.name.c_str());
        writeArgumentToWriter(argument.value, argsWriter);
    }

    argsWriter.EndObject();
    toolCall.arguments = sb.GetString();
    currentState = State::ToolCallEnded;
    streamingPosition = pos + TOOL_ARGS_END_INDICATOR.length();

    return true;
}

bool parseInToolCallEndedState(const std::string& streamingContent, size_t& streamingPosition, State& currentState, const std::string& toolCallEndTag) {
    size_t pos = streamingContent.find(TOOL_LIST_END_INDICATOR, streamingPosition);
    size_t toolSeparatorPos = streamingContent.find(TOOL_SEPARATOR_STR, streamingPosition);
    size_t toolCallEndTagPos = streamingContent.find(toolCallEndTag, streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Current state: ToolCallEnded. Streaming content from current position: {}", streamingContent.substr(streamingPosition));
    if (pos == std::string::npos && toolSeparatorPos == std::string::npos && toolCallEndTagPos == std::string::npos) {
        return false;
    } else if (toolSeparatorPos != std::string::npos && toolSeparatorPos < pos) {
        streamingPosition = toolSeparatorPos + TOOL_SEPARATOR_STR.length();
        currentState = State::ToolCallStarted;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected separator between tool calls at position: {}, expecting another tool call to start", toolSeparatorPos);
    } else if (toolCallEndTagPos != std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected end of tool call at position: {}", toolCallEndTagPos);
        streamingPosition = toolCallEndTagPos + toolCallEndTag.length();
        currentState = State::AfterToolCall;
    } else {
        streamingPosition = pos + TOOL_LIST_END_INDICATOR.length();
        currentState = State::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected end of tool list at position: {}, returning to content state", pos);
    }
    return true;
}

rapidjson::Document wrapDeltaContent(const std::string& content) {
    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Value deltaObj(rapidjson::kObjectType);
    deltaObj.AddMember("content", rapidjson::Value(content.c_str(), doc.GetAllocator()), doc.GetAllocator());
    doc.AddMember("delta", deltaObj, doc.GetAllocator());
    return doc;
}

rapidjson::Document wrapDeltaArgs(const std::string& argsStr, int toolCallIndex) {
    rapidjson::Document doc(rapidjson::kObjectType);
    doc.AddMember("arguments", rapidjson::Value(argsStr.c_str(), doc.GetAllocator()), doc.GetAllocator());

    return BaseOutputParser::wrapDelta(doc, toolCallIndex);
}

void cutEOSFromContent(std::string& content) {
    size_t eosPos = content.find(EOS_TOKEN_STR);
    if (eosPos != std::string::npos) {
        content = content.substr(0, eosPos);
    }
}

bool parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall) {
    size_t argsPos = toolStr.find(TOOL_ARGS_START_INDICATOR);
    if (argsPos != std::string::npos) {
        std::string toolName = toolStr.substr(0, argsPos);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);

        int argsStrLen = toolStr.length() - argsPos - TOOL_ARGS_START_INDICATOR.length() - TOOL_ARGS_END_INDICATOR.length();
        std::string argsStr = toolStr.substr(argsPos + TOOL_ARGS_START_INDICATOR.length(), argsStrLen);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed args string: {}", argsStr);
        std::vector<Argument> arguments = parseArguments(argsStr);

        toolCall.name = toolName;
        rapidjson::Document argsDoc(rapidjson::kObjectType);
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
        argsWriter.StartObject();
        for (const Argument& argument : arguments) {
            argsWriter.Key(argument.name.c_str());
            writeArgumentToWriter(argument.value, argsWriter);
        }
        argsWriter.EndObject();
        toolCall.arguments = sb.GetString();
        toolCall.id = generateRandomId();
        return true;
    }
    return false;
}

void parseUnaryResponse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens, ov::genai::Tokenizer& tokenizer, const int64_t botTokenId, const int64_t eotTokenId, const std::optional<int64_t> reasoningEndTokenId) {
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

        std::string toolListStr = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + start + 1, generatedTokens.begin() + end), ov::AnyMap{ov::genai::skip_special_tokens(false)});
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool list string: {}", toolListStr);
        toolListStr = toolListStr.substr(TOOL_LIST_START_INDICATOR.length(), toolListStr.length() - TOOL_LIST_START_INDICATOR.length() - TOOL_LIST_END_INDICATOR.length());

        while (!toolListStr.empty()) {
            size_t toolEndPos = findInStringRespectingSpecialChars(toolListStr, TOOL_ARGS_END_INDICATOR, 0);
            std::string singleTool;
            if (toolEndPos != std::string::npos) {
                singleTool = toolListStr.substr(0, toolEndPos + TOOL_ARGS_END_INDICATOR.length());
                if (toolEndPos + TOOL_ARGS_END_INDICATOR.length() < toolListStr.length()) {
                    toolListStr = toolListStr.substr(toolEndPos + TOOL_ARGS_END_INDICATOR.length() + TOOL_SEPARATOR_STR.length());
                } else {
                    toolListStr.clear();
                }
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed single tool string {}", singleTool);
            } else {
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "No more tool calls found in tool list string: {}", toolListStr);
                break;
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
    if (reasoningEndTokenId.has_value()) {
        auto reasoningEndIt = std::find(contentWithoutToolCalls.begin(), contentWithoutToolCalls.end(), reasoningEndTokenId.value());
        if (reasoningEndIt != contentWithoutToolCalls.end()) {
            contentWithoutToolCalls.erase(reasoningEndIt, contentWithoutToolCalls.end());
        }
    }

    parsedOutput.content = tokenizer.decode(contentWithoutToolCalls, ov::AnyMap{ov::genai::skip_special_tokens(true)});
}
} // namespace ovms
