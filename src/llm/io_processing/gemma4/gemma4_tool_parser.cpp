//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************

#include "gemma4_tool_parser.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

namespace ovms {

const std::string Gemma4ToolParser::TOOL_CALL_START_TAG = "<|tool_call>";
const std::string Gemma4ToolParser::TOOL_CALL_END_TAG = "<tool_call|>";
const std::string Gemma4ToolParser::TOOL_CALL_NAME_PREFIX = "call:";
const std::string Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR = "<|\"|>";
const std::string Gemma4ToolParser::TURN_END_TAG = "<turn|>";
const std::string Gemma4ToolParser::TOOL_RESPONSE_START_TAG = "<|tool_response>";

namespace {

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

void trimLocal(std::string& value) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
}

bool saneToolName(const std::string& name) {
    if (name.empty())
        return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-' || c == '.';
    });
}

class NativeValueParser {
    const std::string& input;
    size_t pos{0};
    JsonWriter& writer;

    bool startsWith(const std::string& marker) const {
        return pos + marker.size() <= input.size() && input.compare(pos, marker.size(), marker) == 0;
    }

    void skipWs() {
        while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos])))
            ++pos;
    }

    bool writeJsonToken(const std::string& token) {
        rapidjson::Document doc;
        doc.Parse(token.c_str());
        if (doc.HasParseError())
            return false;
        return doc.Accept(writer);
    }

    bool parseDelimitedString() {
        if (!startsWith(Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR))
            return false;
        pos += Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR.size();
        const size_t end = input.find(Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR, pos);
        if (end == std::string::npos)
            return false;
        writer.String(input.data() + pos, static_cast<rapidjson::SizeType>(end - pos));
        pos = end + Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR.size();
        return true;
    }

    bool parseJsonString() {
        if (pos >= input.size() || input[pos] != '"')
            return false;
        const size_t start = pos++;
        bool escaped = false;
        while (pos < input.size()) {
            const char c = input[pos++];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (c == '\\') {
                escaped = true;
                continue;
            }
            if (c == '"')
                return writeJsonToken(input.substr(start, pos - start));
        }
        return false;
    }

    bool parseKey(std::string& key) {
        skipWs();
        if (startsWith(Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR)) {
            pos += Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR.size();
            const size_t end = input.find(Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR, pos);
            if (end == std::string::npos)
                return false;
            key = input.substr(pos, end - pos);
            pos = end + Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR.size();
            return true;
        }
        if (pos < input.size() && input[pos] == '"') {
            const size_t start = pos++;
            bool escaped = false;
            while (pos < input.size()) {
                const char c = input[pos++];
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (c == '\\') {
                    escaped = true;
                    continue;
                }
                if (c == '"') {
                    rapidjson::Document keyDoc;
                    const std::string token = input.substr(start, pos - start);
                    keyDoc.Parse(token.c_str());
                    if (keyDoc.HasParseError() || !keyDoc.IsString())
                        return false;
                    key.assign(keyDoc.GetString(), keyDoc.GetStringLength());
                    return true;
                }
            }
            return false;
        }
        const size_t start = pos;
        while (pos < input.size() && input[pos] != ':')
            ++pos;
        if (pos == input.size())
            return false;
        key = input.substr(start, pos - start);
        trimLocal(key);
        return !key.empty();
    }

    bool parseObject() {
        if (pos >= input.size() || input[pos] != '{')
            return false;
        ++pos;
        writer.StartObject();
        skipWs();
        if (pos < input.size() && input[pos] == '}') {
            ++pos;
            writer.EndObject();
            return true;
        }
        while (pos < input.size()) {
            std::string key;
            if (!parseKey(key))
                return false;
            skipWs();
            if (pos >= input.size() || input[pos] != ':')
                return false;
            ++pos;
            writer.Key(key.c_str(), static_cast<rapidjson::SizeType>(key.size()));
            if (!parseValue())
                return false;
            skipWs();
            if (pos < input.size() && input[pos] == ',') {
                ++pos;
                skipWs();
                continue;
            }
            if (pos < input.size() && input[pos] == '}') {
                ++pos;
                writer.EndObject();
                return true;
            }
            return false;
        }
        return false;
    }

    bool parseArray() {
        if (pos >= input.size() || input[pos] != '[')
            return false;
        ++pos;
        writer.StartArray();
        skipWs();
        if (pos < input.size() && input[pos] == ']') {
            ++pos;
            writer.EndArray();
            return true;
        }
        while (pos < input.size()) {
            if (!parseValue())
                return false;
            skipWs();
            if (pos < input.size() && input[pos] == ',') {
                ++pos;
                skipWs();
                continue;
            }
            if (pos < input.size() && input[pos] == ']') {
                ++pos;
                writer.EndArray();
                return true;
            }
            return false;
        }
        return false;
    }

    bool parseBareScalar() {
        const size_t start = pos;
        while (pos < input.size()) {
            const char c = input[pos];
            if (c == ',' || c == '}' || c == ']' || c == ')')
                break;
            ++pos;
        }
        std::string token = input.substr(start, pos - start);
        trimLocal(token);
        if (token.empty())
            return false;
        if (writeJsonToken(token))
            return true;
        writer.String(token.c_str(), static_cast<rapidjson::SizeType>(token.size()));
        return true;
    }

public:
    NativeValueParser(const std::string& input, JsonWriter& writer) : input(input), writer(writer) {}

    bool parseValue() {
        skipWs();
        if (pos >= input.size())
            return false;
        if (startsWith(Gemma4ToolParser::TOOL_ARGS_STRING_INDICATOR))
            return parseDelimitedString();
        if (input[pos] == '"')
            return parseJsonString();
        if (input[pos] == '{')
            return parseObject();
        if (input[pos] == '[')
            return parseArray();
        return parseBareScalar();
    }

    bool parseArgumentsBody() {
        writer.StartObject();
        skipWs();
        if (pos == input.size()) {
            writer.EndObject();
            return true;
        }
        while (pos < input.size()) {
            std::string key;
            if (!parseKey(key))
                return false;
            skipWs();
            if (pos >= input.size() || input[pos] != ':')
                return false;
            ++pos;
            writer.Key(key.c_str(), static_cast<rapidjson::SizeType>(key.size()));
            if (!parseValue())
                return false;
            skipWs();
            if (pos == input.size()) {
                writer.EndObject();
                return true;
            }
            if (input[pos] != ',')
                return false;
            ++pos;
            skipWs();
            if (pos == input.size())
                return false;
        }
        return false;
    }

    bool parseSingleValueFully() {
        if (!parseValue())
            return false;
        skipWs();
        return pos == input.size();
    }
};

std::optional<std::string> normalizeSingleNativeValue(const std::string& arg) {
    std::string value = arg;
    trimLocal(value);
    if (value.empty())
        return std::nullopt;

    rapidjson::Document doc;
    doc.Parse(value.c_str());
    if (!doc.HasParseError()) {
        rapidjson::StringBuffer buffer;
        JsonWriter writer(buffer);
        doc.Accept(writer);
        return std::string(buffer.GetString(), buffer.GetSize());
    }

    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    NativeValueParser parser(value, writer);
    if (!parser.parseSingleValueFully())
        return std::nullopt;
    return std::string(buffer.GetString(), buffer.GetSize());
}

}  // namespace

std::optional<std::string> Gemma4ToolParser::parseNativeArgumentsBody(const std::string& argumentsBody) {
    const std::string jsonCandidate = "{" + argumentsBody + "}";
    rapidjson::Document doc;
    doc.Parse(jsonCandidate.c_str());
    if (!doc.HasParseError() && doc.IsObject()) {
        rapidjson::StringBuffer buffer;
        JsonWriter writer(buffer);
        doc.Accept(writer);
        return std::string(buffer.GetString(), buffer.GetSize());
    }

    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    NativeValueParser parser(argumentsBody, writer);
    if (!parser.parseArgumentsBody())
        return std::nullopt;
    return std::string(buffer.GetString(), buffer.GetSize());
}

std::optional<size_t> Gemma4ToolParser::findMatchingContainerEnd(const std::string& text, size_t openPos, char openChar, char closeChar) {
    if (openPos >= text.size() || text[openPos] != openChar)
        return std::nullopt;

    std::vector<char> expectedClosers{closeChar};
    size_t i = openPos + 1;
    while (i < text.size()) {
        if (text.compare(i, TOOL_CALL_END_TAG.size(), TOOL_CALL_END_TAG) == 0)
            return std::nullopt;

        if (text.compare(i, TOOL_ARGS_STRING_INDICATOR.size(), TOOL_ARGS_STRING_INDICATOR) == 0) {
            const size_t valueStart = i + TOOL_ARGS_STRING_INDICATOR.size();
            const size_t valueEnd = text.find(TOOL_ARGS_STRING_INDICATOR, valueStart);
            if (valueEnd == std::string::npos)
                return std::nullopt;
            i = valueEnd + TOOL_ARGS_STRING_INDICATOR.size();
            continue;
        }

        if (text[i] == '"') {
            ++i;
            bool escaped = false;
            while (i < text.size()) {
                const char c = text[i++];
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (c == '\\') {
                    escaped = true;
                    continue;
                }
                if (c == '"')
                    break;
            }
            continue;
        }

        switch (text[i]) {
        case '{': expectedClosers.push_back('}'); break;
        case '[': expectedClosers.push_back(']'); break;
        case '(': expectedClosers.push_back(')'); break;
        case '}':
        case ']':
        case ')':
            if (expectedClosers.empty() || expectedClosers.back() != text[i])
                return std::nullopt;
            expectedClosers.pop_back();
            if (expectedClosers.empty())
                return i;
            break;
        default: break;
        }
        ++i;
    }
    return std::nullopt;
}

std::string Gemma4ToolParser::normalizeToolName(std::string rawName) {
    trim(rawName);
    if (rawName.rfind(TOOL_CALL_NAME_PREFIX, 0) == 0)
        rawName.erase(0, TOOL_CALL_NAME_PREFIX.size());
    trim(rawName);
    if (!rawName.empty() && rawName.front() == ':')
        rawName.erase(rawName.begin());
    trim(rawName);
    return rawName;
}

std::string Gemma4ToolParser::normalizeArgStr(const std::string& arg) {
    auto normalized = normalizeSingleNativeValue(arg);
    return normalized.value_or(arg);
}

std::string Gemma4ToolParser::parseArrayParameter(const std::string& argumentStr) {
    return normalizeArgStr(argumentStr);
}

std::string Gemma4ToolParser::parseObjectParameter(const std::string& argumentStr) {
    return normalizeArgStr(argumentStr);
}

bool Gemma4ToolParser::parseInContentState() {
    const size_t toolCallStartTagPos = streamingContent.find(TOOL_CALL_START_TAG, streamingPosition);
    if (toolCallStartTagPos != std::string::npos) {
        if (toolCallStartTagPos > streamingPosition)
            return true;
        streamingPosition = toolCallStartTagPos + TOOL_CALL_START_TAG.length();
        currentState = State::ToolCallStarted;
        currentCallValid = true;
        return false;
    }

    // The generic OutputParser may route the preamble-only `call:` variant here
    // after a reasoning end boundary. Do not search for it later in arbitrary
    // content: accept it only exactly at the current phase entry position.
    if (streamingContent.compare(streamingPosition, TOOL_CALL_NAME_PREFIX.size(), TOOL_CALL_NAME_PREFIX) == 0) {
        currentState = State::ToolCallStarted;
        currentCallValid = true;
        return false;
    }
    return true;
}

bool Gemma4ToolParser::parseInToolCallState() {
    const size_t endTagPos = streamingContent.find(TOOL_CALL_END_TAG, streamingPosition);
    const size_t bracePos = streamingContent.find('{', streamingPosition);
    const size_t parenPos = streamingContent.find('(', streamingPosition);

    size_t argsPos = std::string::npos;
    if (bracePos != std::string::npos)
        argsPos = bracePos;
    if (parenPos != std::string::npos && (argsPos == std::string::npos || parenPos < argsPos))
        argsPos = parenPos;

    if (endTagPos != std::string::npos && (argsPos == std::string::npos || endTagPos < argsPos)) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Gemma4 tool call ended before an argument container; dropping malformed call");
        streamingPosition = endTagPos + TOOL_CALL_END_TAG.size();
        currentState = State::AfterToolCall;
        currentCallValid = false;
        toolCall = {};
        return true;
    }
    if (argsPos == std::string::npos)
        return false;

    std::string toolName = normalizeToolName(streamingContent.substr(streamingPosition, argsPos - streamingPosition));
    currentCallValid = saneToolName(toolName) && toolNameAllowed(toolName);
    if (!currentCallValid)
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Gemma4 parser refusing malformed or unavailable tool name: '{}'", toolName);

    currentArgsOpen = streamingContent[argsPos];
    currentArgsClose = currentArgsOpen == '(' ? ')' : '}';
    streamingPosition = argsPos + 1;
    currentState = State::ToolCallParameters;

    if (currentCallValid) {
        toolCall = ToolCall{generateRandomId(), toolName, ""};
        ++toolCallIndex;
    } else {
        toolCall = {};
    }
    return true;
}

bool Gemma4ToolParser::parseToolCallParametersState() {
    if (streamingPosition == 0)
        return false;
    const size_t openPos = streamingPosition - 1;
    auto closePos = findMatchingContainerEnd(streamingContent, openPos, currentArgsOpen, currentArgsClose);
    if (!closePos.has_value()) {
        const size_t endTagPos = streamingContent.find(TOOL_CALL_END_TAG, streamingPosition);
        if (endTagPos != std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Gemma4 malformed tool arguments bounded by <tool_call|>; dropping current call");
            streamingPosition = endTagPos + TOOL_CALL_END_TAG.size();
            currentState = State::AfterToolCall;
            currentCallValid = false;
            toolCall = {};
            return true;
        }
        return false;
    }

    const std::string argumentsBody = streamingContent.substr(streamingPosition, closePos.value() - streamingPosition);
    if (currentCallValid) {
        auto parsedArguments = parseNativeArgumentsBody(argumentsBody);
        if (parsedArguments.has_value()) {
            toolCall.arguments = std::move(parsedArguments.value());
        } else {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Gemma4 native argument parse failed; refusing executable tool call '{}'.", toolCall.name);
            currentCallValid = false;
            toolCall = {};
        }
    }

    streamingPosition = closePos.value() + 1;
    currentState = State::ToolCallEnded;
    return true;
}

bool Gemma4ToolParser::parseInToolCallEndedState() {
    const size_t endTagPos = streamingContent.find(TOOL_CALL_END_TAG, streamingPosition);
    const size_t nextCallPos = streamingContent.find(TOOL_CALL_NAME_PREFIX, streamingPosition);

    if (nextCallPos != std::string::npos && (endTagPos == std::string::npos || nextCallPos < endTagPos)) {
        streamingPosition = nextCallPos;
        currentState = State::ToolCallStarted;
        currentCallValid = true;
        return true;
    }
    if (endTagPos != std::string::npos) {
        streamingPosition = endTagPos + TOOL_CALL_END_TAG.length();
        currentState = State::AfterToolCall;
        return true;
    }
    return false;
}

bool Gemma4ToolParser::parseNewContent() {
    switch (currentState) {
    case State::Content: return parseInContentState();
    case State::ToolCallStarted: return parseInToolCallState();
    case State::ToolCallParameters: return parseToolCallParametersState();
    case State::ToolCallEnded: return parseInToolCallEndedState();
    case State::AfterToolCall: break;
    }
    return false;
}

std::optional<Delta> Gemma4ToolParser::wrapDeltaContent(const std::string& content) {
    if (content.empty())
        return std::nullopt;
    return ContentDelta{content};
}

ToolCallDelta Gemma4ToolParser::wrapDeltaArgs(const std::string& argsStr, int index) {
    return ToolCallDelta{index, std::nullopt, std::nullopt, argsStr};
}

std::optional<Delta> Gemma4ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (!chunk.empty())
        streamingContent += chunk;

    if (parseNewContent()) {
        if (currentState == State::ToolCallParameters && currentCallValid)
            return ToolCallDelta{toolCallIndex, toolCall.id, toolCall.name, ""};
        if (currentState == State::ToolCallEnded) {
            if (currentCallValid && !toolCall.arguments.empty()) {
                auto delta = wrapDeltaArgs(toolCall.arguments, toolCallIndex);
                toolCall = {};
                return delta;
            }
            toolCall = {};
            return std::nullopt;
        }
        if (currentState == State::Content) {
            const size_t contentEnd = streamingContent.find(TOOL_CALL_START_TAG, streamingPosition);
            std::string content = contentEnd == std::string::npos
                ? streamingContent.substr(streamingPosition)
                : streamingContent.substr(streamingPosition, contentEnd - streamingPosition);
            streamingPosition += content.size();
            for (const std::string& tagToErase : {TURN_END_TAG, TOOL_RESPONSE_START_TAG}) {
                size_t tagPos = content.find(tagToErase);
                while (tagPos != std::string::npos) {
                    content.erase(tagPos, tagToErase.length());
                    tagPos = content.find(tagToErase, tagPos);
                }
            }
            return wrapDeltaContent(content);
        }
        if (currentState == State::AfterToolCall)
            currentState = State::Content;
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        if (currentState == State::ToolCallParameters)
            parseToolCallParametersState();
        if (currentState == State::ToolCallEnded && currentCallValid && !toolCall.arguments.empty()) {
            auto delta = wrapDeltaArgs(toolCall.arguments, toolCallIndex);
            toolCall = {};
            return delta;
        }
        if (currentState == State::Content && streamingPosition < streamingContent.size()) {
            auto content = streamingContent.substr(streamingPosition);
            streamingPosition += content.size();
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

}  // namespace ovms
