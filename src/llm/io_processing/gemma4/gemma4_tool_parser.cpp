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
#include <unordered_set>
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

// Tool arguments are JSON text, not machine arithmetic. Preserve number tokens
// without routing large integers/decimals through uint64_t or double.
class NumberPreservingWriter : public JsonWriter {
public:
    explicit NumberPreservingWriter(rapidjson::StringBuffer& buffer) : JsonWriter(buffer) {}
    bool RawNumber(const char* value, rapidjson::SizeType length, bool) {
        return RawValue(value, length, rapidjson::kNumberType);
    }
};

std::optional<std::string> normalizeJsonLosslessly(const std::string& input) {
    rapidjson::StringStream stream(input.c_str());
    rapidjson::Reader reader;
    rapidjson::StringBuffer buffer;
    NumberPreservingWriter writer(buffer);
    if (!reader.Parse<rapidjson::kParseNumbersAsStringsFlag>(stream, writer) || stream.Tell() != input.size())
        return std::nullopt;
    return std::string(buffer.GetString(), buffer.GetSize());
}

bool isValidJsonNumber(const std::string& token) {
    size_t pos = 0;
    if (pos < token.size() && token[pos] == '-')
        ++pos;
    if (pos == token.size())
        return false;
    if (token[pos] == '0') {
        ++pos;
    } else {
        if (!std::isdigit(static_cast<unsigned char>(token[pos])))
            return false;
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos])))
            ++pos;
    }
    if (pos < token.size() && token[pos] == '.') {
        ++pos;
        const size_t digits = pos;
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos])))
            ++pos;
        if (pos == digits)
            return false;
    }
    if (pos < token.size() && (token[pos] == 'e' || token[pos] == 'E')) {
        ++pos;
        if (pos < token.size() && (token[pos] == '+' || token[pos] == '-'))
            ++pos;
        const size_t digits = pos;
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos])))
            ++pos;
        if (pos == digits)
            return false;
    }
    return pos == token.size();
}

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

// Recover only the observed native leak shape where `call:` starts a logical line
// (optionally indented). This deliberately rejects prose such as
// `Documentation example: call:foo{...}` and quoted examples. The candidate is
// still validated again by the normal argument parser before it becomes executable.
std::optional<size_t> findRecoverableBareCall(
    const std::string& content,
    size_t from,
    const std::unordered_set<std::string>& allowedToolNames,
    bool enforceToolRegistry) {
    size_t candidate = content.find(Gemma4ToolParser::TOOL_CALL_NAME_PREFIX, from);
    while (candidate != std::string::npos) {
        bool lineBoundary = candidate == from;
        if (!lineBoundary) {
            const size_t lineStartPos = content.rfind('\n', candidate - 1);
            const size_t lineStart = lineStartPos == std::string::npos ? from : lineStartPos + 1;
            lineBoundary = lineStart >= from;
            for (size_t i = lineStart; lineBoundary && i < candidate; ++i) {
                const char c = content[i];
                if (c != ' ' && c != '\t' && c != '\r')
                    lineBoundary = false;
            }
        }
        if (!lineBoundary) {
            candidate = content.find(Gemma4ToolParser::TOOL_CALL_NAME_PREFIX, candidate + Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.size());
            continue;
        }

        const size_t nameStart = candidate + Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.size();
        const size_t bracePos = content.find('{', nameStart);
        const size_t parenPos = content.find('(', nameStart);
        size_t argsPos = std::string::npos;
        if (bracePos != std::string::npos)
            argsPos = bracePos;
        if (parenPos != std::string::npos && (argsPos == std::string::npos || parenPos < argsPos))
            argsPos = parenPos;
        if (argsPos == std::string::npos) {
            if (!enforceToolRegistry)
                return candidate;

            std::string partialName = content.substr(nameStart);
            trimLocal(partialName);
            if (partialName.empty())
                return candidate;

            const bool couldBecomeAllowed = saneToolName(partialName) && std::any_of(
                allowedToolNames.begin(), allowedToolNames.end(), [&](const std::string& allowedName) {
                    return allowedName.rfind(partialName, 0) == 0;
                });
            if (couldBecomeAllowed)
                return candidate;  // hold only a viable streaming tool-name prefix

            candidate = content.find(Gemma4ToolParser::TOOL_CALL_NAME_PREFIX, candidate + Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.size());
            continue;
        }

        std::string name = content.substr(nameStart, argsPos - nameStart);
        trimLocal(name);
        const bool allowed = saneToolName(name) && (!enforceToolRegistry || allowedToolNames.count(name) != 0);
        if (allowed)
            return candidate;

        candidate = content.find(Gemma4ToolParser::TOOL_CALL_NAME_PREFIX, candidate + Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.size());
    }
    return std::nullopt;
}

// Whether a trailing line fragment could still become a recoverable bare
// `call:` boundary with more streaming input. Only spaces plus a proper
// prefix of "call:" (no completed colon) are holdable here; a completed
// "call:" (with or without a tool name) is handled by findRecoverableBareCall
// as a complete split point, and anything else on the line already rules out a
// line-start bare call.
bool isHoldableBareCallPrefix(const std::string& lineText) {
    if (lineText.empty())
        return false;
    size_t i = 0;
    while (i < lineText.size() && (lineText[i] == ' ' || lineText[i] == '\t' || lineText[i] == '\r'))
        ++i;
    const std::string rest = lineText.substr(i);
    if (rest.empty())
        return true;  // spaces only after a newline: may become "  call:" next chunk
    if (rest.size() >= Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.size())
        return false;  // completed "call:" or longer: split logic owns it, do not hold here
    return Gemma4ToolParser::TOOL_CALL_NAME_PREFIX.compare(0, rest.size(), rest) == 0;
}

// Earliest position from which trailing bytes must be held because they may
// still grow into a line-start bare `call:` boundary. Returns npos when the
// buffered suffix is safe to emit as ordinary content.
size_t bareCallHoldStart(const std::string& content, size_t from) {
    if (content.size() <= from)
        return std::string::npos;
    const size_t nl = content.rfind('\n');
    size_t lineStartFull = (nl == std::string::npos) ? 0 : nl + 1;
    if (lineStartFull < from) {
        for (size_t i = lineStartFull; i < from; ++i) {
            const char c = content[i];
            if (c != ' ' && c != '\t' && c != '\r')
                return std::string::npos;  // line already has prose: no bare call possible
        }
        if (isHoldableBareCallPrefix(content.substr(from)))
            return from;
        return std::string::npos;
    }
    if (isHoldableBareCallPrefix(content.substr(lineStartFull)))
        return lineStartFull;
    return std::string::npos;
}

// Earliest position of a trailing partial "<|tool_call>" start tag that must
// be held until the next chunk proves or disproves the boundary.
size_t startTagHoldStart(const std::string& content, size_t from) {
    const std::string& tag = Gemma4ToolParser::TOOL_CALL_START_TAG;
    if (content.size() <= from || tag.size() <= 1)
        return std::string::npos;
    const size_t avail = content.size() - from;
    size_t maxLen = std::min(avail, tag.size() - 1);
    for (size_t len = maxLen; len > 0; --len) {
        if (content.compare(content.size() - len, len, tag, 0, len) == 0)
            return content.size() - len;
    }
    return std::string::npos;
}

bool anchoredToolCallMayStartAt(const std::string& content, size_t pos) {
    const std::string& tag = Gemma4ToolParser::TOOL_CALL_START_TAG;
    const std::string& prefix = Gemma4ToolParser::TOOL_CALL_NAME_PREFIX;
    if (pos + tag.size() > content.size() || content.compare(pos, tag.size(), tag) != 0)
        return false;
    const size_t afterTag = pos + tag.size();
    const size_t suffixSize = content.size() - afterTag;
    if (suffixSize == 0)
        return true;  // full tag at tail; next chunk decides whether it is a call
    if (content[afterTag] == ':')
        return true;
    if (suffixSize < prefix.size())
        return prefix.compare(0, suffixSize, content, afterTag, suffixSize) == 0;
    return content.compare(afterTag, prefix.size(), prefix) == 0;
}

std::optional<size_t> findAnchoredToolCallStart(const std::string& content, size_t from) {
    size_t pos = content.find(Gemma4ToolParser::TOOL_CALL_START_TAG, from);
    while (pos != std::string::npos) {
        if (anchoredToolCallMayStartAt(content, pos))
            return pos;
        pos = content.find(Gemma4ToolParser::TOOL_CALL_START_TAG, pos + Gemma4ToolParser::TOOL_CALL_START_TAG.size());
    }
    return std::nullopt;
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
        auto normalized = normalizeJsonLosslessly(token);
        if (!normalized)
            return false;
        if (!token.empty() && (std::isdigit(static_cast<unsigned char>(token.front())) || token.front() == '-')) {
            // Validate numeric syntax through RapidJSON while preserving the original
            // lexical token. This keeps large/high-precision values lossless without
            // allowing malformed forms such as `1.` or `1e` into OpenAI JSON.
            if (*normalized != token)
                return false;
            return writer.RawValue(token.data(), static_cast<rapidjson::SizeType>(token.size()), rapidjson::kNumberType);
        }
        // This path accepts a quoted string or a bare scalar only.
        const auto type = normalized->front() == '"' ? rapidjson::kStringType :
            normalized->front() == 't' ? rapidjson::kTrueType :
            normalized->front() == 'f' ? rapidjson::kFalseType :
            normalized->front() == 'n' ? rapidjson::kNullType : rapidjson::kNumberType;
        return writer.RawValue(normalized->data(), normalized->size(), type);
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
        const bool numericCandidate = std::isdigit(static_cast<unsigned char>(token.front())) || token.front() == '-';
        if (numericCandidate) {
            if (!isValidJsonNumber(token))
                return false;
            return writer.RawValue(token.data(), static_cast<rapidjson::SizeType>(token.size()), rapidjson::kNumberType);
        }
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
    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    NativeValueParser parser(argumentsBody, writer);
    if (!parser.parseArgumentsBody())
        return std::nullopt;
    return std::string(buffer.GetString(), buffer.GetSize());
}

std::optional<size_t> Gemma4ToolParser::findMatchingContainerEnd(const std::string& text, size_t openPos, char openChar, char closeChar, size_t& malformedEndTag) {
    malformedEndTag = std::string::npos;
    if (openPos >= text.size() || text[openPos] != openChar)
        return std::nullopt;

    std::vector<char> expectedClosers{closeChar};
    bool malformed = false;
    size_t i = openPos + 1;
    while (i < text.size()) {
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

        if (text.compare(i, TOOL_CALL_END_TAG.size(), TOOL_CALL_END_TAG) == 0) {
            malformedEndTag = i;
            return std::nullopt;
        }

        switch (text[i]) {
        case '{': expectedClosers.push_back('}'); break;
        case '[': expectedClosers.push_back(']'); break;
        case '(': expectedClosers.push_back(')'); break;
        case '}':
        case ']':
        case ')':
            if (expectedClosers.empty() || expectedClosers.back() != text[i]) {
                malformed = true;
                break;
            }
            expectedClosers.pop_back();
            if (expectedClosers.empty() && !malformed)
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
    const auto toolCallStartTagPos = findAnchoredToolCallStart(streamingContent, streamingPosition);
    if (toolCallStartTagPos.has_value()) {
        if (toolCallStartTagPos.value() > streamingPosition)
            return true;
        const size_t namePrefixPos = toolCallStartTagPos.value() + TOOL_CALL_START_TAG.length();
        const size_t suffixSize = streamingContent.size() - namePrefixPos;
        if (suffixSize < TOOL_CALL_NAME_PREFIX.size() &&
            TOOL_CALL_NAME_PREFIX.compare(0, suffixSize, streamingContent, namePrefixPos, suffixSize) == 0)
            return false;
        const bool colonVariant = suffixSize > 0 && streamingContent[namePrefixPos] == ':';
        if (!colonVariant && streamingContent.compare(namePrefixPos, TOOL_CALL_NAME_PREFIX.size(), TOOL_CALL_NAME_PREFIX) != 0)
            return true;
        currentCallStartPos = toolCallStartTagPos.value();
        currentCallBare = false;
        streamingPosition = namePrefixPos + (colonVariant ? 1 : TOOL_CALL_NAME_PREFIX.size());
        currentState = State::ToolCallStarted;
        currentCallValid = true;
        return false;
    }

    const auto bareCallPos = findRecoverableBareCall(streamingContent, streamingPosition, allowedToolNames, enforceToolRegistry);
    if (bareCallPos.has_value()) {
        if (bareCallPos.value() > streamingPosition)
            return true;
        currentCallStartPos = bareCallPos.value();
        currentCallBare = true;
        streamingPosition = bareCallPos.value() + TOOL_CALL_NAME_PREFIX.size();
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
        if (currentCallBare) {
            // Bare `call:` ended by "<tool_call|>" before any argument container:
            // it was never an anchored call, so keep the bytes as prose.
            streamingPosition = currentCallStartPos;
            currentState = State::Content;
            currentCallBare = false;
            toolCall = {};
            currentCallValid = false;
            return false;
        }
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Gemma4 tool call ended before an argument container; dropping malformed call");
        streamingPosition = endTagPos + TOOL_CALL_END_TAG.size();
        currentState = State::AfterToolCall;
        currentCallValid = false;
        currentCallBare = false;
        toolCall = {};
        return true;
    }
    if (argsPos == std::string::npos)
        return false;

    std::string toolName = normalizeToolName(streamingContent.substr(streamingPosition, argsPos - streamingPosition));
    currentCallValid = saneToolName(toolName) && toolNameAllowed(toolName);
    if (!currentCallValid)
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Gemma4 parser refusing malformed or unavailable tool name: '{}'", toolName);

    if (!currentCallValid && currentCallBare) {
        // A bare line-start `call:` without an anchored "<|tool_call>" marker
        // is ordinary prose unless it names an available tool. Rewind so the
        // bytes re-emit as content instead of being dropped as a refused call.
        // findRecoverableBareCall will skip this now-delimited invalid name on
        // the next pass, so the rewind terminates.
        streamingPosition = currentCallStartPos;
        currentState = State::Content;
        currentCallBare = false;
        toolCall = {};
        return false;
    }
    currentCallBare = false;

    currentArgsOpen = streamingContent[argsPos];
    currentArgsClose = currentArgsOpen == '(' ? ')' : '}';
    streamingPosition = argsPos + 1;
    currentState = State::ToolCallParameters;

    if (currentCallValid) {
        toolCall = ToolCall{generateRandomId(), toolName, ""};
    } else {
        toolCall = {};
    }
    return true;
}

bool Gemma4ToolParser::parseToolCallParametersState() {
    if (streamingPosition == 0)
        return false;
    const size_t openPos = streamingPosition - 1;
    size_t endTagPos = std::string::npos;
    auto closePos = findMatchingContainerEnd(streamingContent, openPos, currentArgsOpen, currentArgsClose, endTagPos);
    if (!closePos.has_value()) {
        if (endTagPos != std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Gemma4 malformed tool arguments bounded by <tool_call|>; dropping current call");
            streamingPosition = endTagPos + TOOL_CALL_END_TAG.size();
            currentState = State::AfterToolCall;
            currentCallValid = false;
            currentCallBare = false;
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
        // A chained call after "<|tool_call>" framing stays anchored (drop on
        // invalid); a bare line-start `call:` rewinds to content on invalid.
        currentCallBare = true;
        currentCallStartPos = nextCallPos;
        const size_t tagPos = streamingContent.rfind(TOOL_CALL_START_TAG, nextCallPos);
        if (tagPos != std::string::npos && tagPos >= streamingPosition &&
            tagPos + TOOL_CALL_START_TAG.size() <= nextCallPos) {
            bool gapClean = true;
            for (size_t i = tagPos + TOOL_CALL_START_TAG.size(); i < nextCallPos; ++i) {
                const char c = streamingContent[i];
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                    gapClean = false;
                    break;
                }
            }
            if (gapClean) {
                currentCallBare = false;
                currentCallStartPos = tagPos;
            }
        }
        streamingPosition = nextCallPos + TOOL_CALL_NAME_PREFIX.size();
        currentState = State::ToolCallStarted;
        currentCallValid = true;
        return true;
    }
    if (endTagPos != std::string::npos) {
        streamingPosition = endTagPos + TOOL_CALL_END_TAG.length();
        currentState = State::AfterToolCall;
        currentCallBare = false;
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
    // Emitted deltas own their strings. Only the unconsumed suffix belongs to
    // this parser; it is not conversation memory. Preserve the argument opener
    // while its container is incomplete (the scanner starts one byte before pos).
    if (streamingPosition >= 4096) {
        const size_t keep = currentState == State::ToolCallParameters ? 1 : 0;
        streamingContent.erase(0, streamingPosition - keep);
        streamingPosition = keep;
    }
    if (!chunk.empty())
        streamingContent += chunk;

    for (;;) {
        const State stateBefore = currentState;
        const size_t positionBefore = streamingPosition;
        const bool ready = parseNewContent();

        if (currentState == State::ToolCallEnded) {
            if (currentCallValid && !toolCall.arguments.empty()) {
                // An emitted header cannot be retracted from SSE or the unary
                // accumulator. Publish the complete call only after validation.
                auto delta = ToolCallDelta{++toolCallIndex, toolCall.id, toolCall.name, toolCall.arguments};
                toolCall = {};
                currentCallValid = false;
                return delta;
            }
            // Nothing to emit and parseInToolCallEndedState found no following
            // boundary (no next call, no end tag). Waiting here must not spin:
            // the previous loop re-entered this block forever when the final
            // STOP flush arrived in ToolCallEnded with an empty call.
            toolCall = {};
            currentCallValid = false;
            break;
        }

        if (ready && currentState == State::Content) {
            const auto anchoredStart = findAnchoredToolCallStart(streamingContent, streamingPosition);
            size_t contentEnd = anchoredStart.value_or(std::string::npos);
            const auto bareCallPos = findRecoverableBareCall(streamingContent, streamingPosition, allowedToolNames, enforceToolRegistry);
            if (bareCallPos.has_value() && (contentEnd == std::string::npos || bareCallPos.value() < contentEnd))
                contentEnd = bareCallPos.value();
            if (contentEnd == std::string::npos && finishReason == ov::genai::GenerationFinishReason::NONE) {
                // No complete boundary yet: hold a trailing fragment that may still
                // grow into "<|tool_call>" or a line-start bare "call:" split
                // across streamer chunks (e.g. "call" + ":" under DELAY_N_TOKENS).
                // Emitting it now as prose would make the boundary unrecoverable.
                const size_t tagHold = startTagHoldStart(streamingContent, streamingPosition);
                const size_t bareHold = bareCallHoldStart(streamingContent, streamingPosition);
                size_t holdStart = std::string::npos;
                if (tagHold != std::string::npos)
                    holdStart = tagHold;
                if (bareHold != std::string::npos && (holdStart == std::string::npos || bareHold < holdStart))
                    holdStart = bareHold;
                if (holdStart != std::string::npos)
                    contentEnd = holdStart;
            }
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

        if (currentState == State::AfterToolCall) {
            currentState = State::Content;
            continue;
        }

        if (finishReason != ov::genai::GenerationFinishReason::NONE && currentState == State::ToolCallParameters) {
            if (parseToolCallParametersState())
                continue;
        }

        if (ready || currentState != stateBefore || streamingPosition != positionBefore)
            continue;
        break;
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        if (currentState == State::ToolCallParameters)
            parseToolCallParametersState();
        if (currentState == State::ToolCallEnded && currentCallValid && !toolCall.arguments.empty()) {
            auto delta = ToolCallDelta{++toolCallIndex, toolCall.id, toolCall.name, toolCall.arguments};
            toolCall = {};
            currentCallValid = false;
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
