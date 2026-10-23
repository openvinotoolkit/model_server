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
#include <rapidjson/reader.h>

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

namespace {
const std::string NATIVE_STRING_MARKER = "<|\"|>";
using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

// A generated tool call should be far smaller than the surrounding HTTP/model
// limits. These local caps prevent a malformed candidate from retaining
// unbounded streaming state while leaving ample room for legitimate arguments.
constexpr size_t MAX_NATIVE_TOOL_CANDIDATE_BYTES = 64 * 1024;
constexpr size_t MAX_NATIVE_TOOL_CONTAINER_DEPTH = 64;

class NumberPreservingWriter : public JsonWriter {
public:
    explicit NumberPreservingWriter(rapidjson::StringBuffer& buffer) :
        JsonWriter(buffer) {}
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

void trimLocal(std::string& value) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
}

bool saneToolName(const std::string& name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-' || c == '.';
    });
}

bool isLogicalBoundary(const std::string& text, size_t pos) {
    if (pos == 0)
        return true;  // The generic router only enters on a boundary-validated preamble.
    const size_t newline = text.rfind('\n', pos - 1);
    if (newline == std::string::npos)
        return false;
    for (size_t i = newline + 1; i < pos; ++i) {
        const char c = text[i];
        if (c != ' ' && c != '\t' && c != '\r')
            return false;
    }
    return true;
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
    bool writeJsonScalar(const std::string& token) {
        auto normalized = normalizeJsonLosslessly(token);
        if (!normalized || normalized->empty())
            return false;
        const char first = normalized->front();
        rapidjson::Type type = rapidjson::kNumberType;
        if (first == '"')
            type = rapidjson::kStringType;
        else if (first == 't')
            type = rapidjson::kTrueType;
        else if (first == 'f')
            type = rapidjson::kFalseType;
        else if (first == 'n')
            type = rapidjson::kNullType;
        else if (!(first == '-' || std::isdigit(static_cast<unsigned char>(first))))
            return false;
        return writer.RawValue(normalized->data(), static_cast<rapidjson::SizeType>(normalized->size()), type);
    }
    bool parseDelimitedString() {
        if (!startsWith(NATIVE_STRING_MARKER))
            return false;
        pos += NATIVE_STRING_MARKER.size();
        const size_t end = input.find(NATIVE_STRING_MARKER, pos);
        if (end == std::string::npos)
            return false;
        const std::string raw = input.substr(pos, end - pos);
        const std::string encoded = escapeAsJsonString(raw);
        if (!writer.RawValue(encoded.data(), static_cast<rapidjson::SizeType>(encoded.size()), rapidjson::kStringType))
            return false;
        pos = end + NATIVE_STRING_MARKER.size();
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
                return writeJsonScalar(input.substr(start, pos - start));
        }
        return false;
    }
    bool parseKey(std::string& key) {
        skipWs();
        if (startsWith(NATIVE_STRING_MARKER)) {
            pos += NATIVE_STRING_MARKER.size();
            const size_t end = input.find(NATIVE_STRING_MARKER, pos);
            if (end == std::string::npos)
                return false;
            key = input.substr(pos, end - pos);
            pos = end + NATIVE_STRING_MARKER.size();
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
                    const std::string token = input.substr(start, pos - start);
                    rapidjson::Document doc;
                    doc.Parse(token.c_str());
                    if (doc.HasParseError() || !doc.IsString())
                        return false;
                    key.assign(doc.GetString(), doc.GetStringLength());
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
        return !token.empty() && writeJsonScalar(token);
    }

public:
    NativeValueParser(const std::string& input, JsonWriter& writer) :
        input(input), writer(writer) {}
    bool parseValue() {
        skipWs();
        if (pos >= input.size())
            return false;
        if (startsWith(NATIVE_STRING_MARKER))
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

std::optional<std::string> Gemma4ToolParser::parseNativeArgumentsBody(const std::string& body) {
    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    NativeValueParser parser(body, writer);
    if (!parser.parseArgumentsBody())
        return std::nullopt;
    return std::string(buffer.GetString(), buffer.GetSize());
}

std::optional<size_t> Gemma4ToolParser::findMatchingContainerEnd(const std::string& text,
    size_t openPos, char openChar, char closeChar, size_t& malformedEndTag,
    bool& candidateLimitExceeded) {
    malformedEndTag = std::string::npos;
    candidateLimitExceeded = false;
    if (openPos >= text.size() || text[openPos] != openChar)
        return std::nullopt;
    std::vector<char> closers{closeChar};
    bool malformed = false;
    size_t i = openPos + 1;
    while (i < text.size()) {
        if (i - openPos > MAX_NATIVE_TOOL_CANDIDATE_BYTES) {
            candidateLimitExceeded = true;
            malformedEndTag = text.find(TOOL_CALL_END_TAG, i);
            return std::nullopt;
        }
        if (text.compare(i, TOOL_ARGS_STRING_INDICATOR.size(), TOOL_ARGS_STRING_INDICATOR) == 0) {
            const size_t end = text.find(TOOL_ARGS_STRING_INDICATOR, i + TOOL_ARGS_STRING_INDICATOR.size());
            if (end == std::string::npos) {
                if (text.size() - openPos > MAX_NATIVE_TOOL_CANDIDATE_BYTES) {
                    candidateLimitExceeded = true;
                    malformedEndTag = text.find(TOOL_CALL_END_TAG, i + TOOL_ARGS_STRING_INDICATOR.size());
                    return std::nullopt;
                }
                // An end tag before the closer is malformed only once the next
                // envelope start is already buffered. A split closer may still arrive.
                const size_t envelopeEnd = text.find(TOOL_CALL_END_TAG, i + TOOL_ARGS_STRING_INDICATOR.size());
                if (envelopeEnd != std::string::npos &&
                    text.find(TOOL_CALL_START_TAG + TOOL_CALL_NAME_PREFIX, envelopeEnd + TOOL_CALL_END_TAG.size()) != std::string::npos)
                    malformedEndTag = envelopeEnd;
                return std::nullopt;
            }
            if (end - openPos > MAX_NATIVE_TOOL_CANDIDATE_BYTES) {
                candidateLimitExceeded = true;
                malformedEndTag = text.find(TOOL_CALL_END_TAG, end);
                return std::nullopt;
            }
            i = end + TOOL_ARGS_STRING_INDICATOR.size();
            continue;
        }
        if (text[i] == '"') {
            ++i;
            bool escaped = false;
            bool closed = false;
            while (i < text.size()) {
                if (i - openPos > MAX_NATIVE_TOOL_CANDIDATE_BYTES) {
                    candidateLimitExceeded = true;
                    malformedEndTag = text.find(TOOL_CALL_END_TAG, i);
                    return std::nullopt;
                }
                if (!escaped && text.compare(i, TOOL_CALL_END_TAG.size(), TOOL_CALL_END_TAG) == 0) {
                    const size_t after = i + TOOL_CALL_END_TAG.size();
                    const size_t nextStart = text.find(TOOL_CALL_START_TAG + TOOL_CALL_NAME_PREFIX, after);
                    const size_t limit = nextStart == std::string::npos ? text.size() : nextStart;
                    bool quoteBeforeNext = false;
                    bool lookEscaped = false;
                    for (size_t j = after; j < limit; ++j) {
                        if (lookEscaped) {
                            lookEscaped = false;
                            continue;
                        }
                        if (text[j] == '\\') {
                            lookEscaped = true;
                            continue;
                        }
                        if (text[j] == '"') {
                            quoteBeforeNext = true;
                            break;
                        }
                    }
                    if (!quoteBeforeNext) {
                        if (nextStart != std::string::npos)
                            malformedEndTag = i;
                        return std::nullopt;
                    }
                }
                const char c = text[i++];
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (c == '\\') {
                    escaped = true;
                    continue;
                }
                if (c == '"') {
                    closed = true;
                    break;
                }
            }
            if (!closed)
                return std::nullopt;
            continue;
        }
        if (text.compare(i, TOOL_CALL_END_TAG.size(), TOOL_CALL_END_TAG) == 0) {
            malformedEndTag = i;
            return std::nullopt;
        }
        switch (text[i]) {
        case '{':
        case '[':
        case '(':
            if (closers.size() >= MAX_NATIVE_TOOL_CONTAINER_DEPTH) {
                candidateLimitExceeded = true;
                malformedEndTag = text.find(TOOL_CALL_END_TAG, i);
                return std::nullopt;
            }
            if (text[i] == '{')
                closers.push_back('}');
            else if (text[i] == '[')
                closers.push_back(']');
            else
                closers.push_back(')');
            break;
        case '}':
        case ']':
        case ')':
            if (closers.empty() || closers.back() != text[i]) {
                malformed = true;
                break;
            }
            closers.pop_back();
            if (closers.empty() && !malformed)
                return i;
            break;
        default:
            break;
        }
        ++i;
    }
    return std::nullopt;
}

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

    return escapeAsJsonString(normalized);
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
    if (toolCallStartTagPos == std::string::npos && enforceToolRegistry) {
        for (const auto& tag : parsingConfig.preambleStartTags) {
            const size_t pos = streamingContent.find(tag, streamingPosition);
            if (pos != std::string::npos && isLogicalBoundary(streamingContent, pos) &&
                (toolCallStartTagPos == std::string::npos || pos < toolCallStartTagPos))
                toolCallStartTagPos = pos;
        }
        if (toolCallStartTagPos == streamingPosition) {
            currentState = State::ToolCallStarted;
            return false;
        }
    }
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
    if (enforceToolRegistry) {
        const auto paren = streamingContent.find('(', streamingPosition);
        if (paren != std::string::npos && (argsPos == std::string::npos || paren < argsPos))
            argsPos = paren;
    }
    if (argsPos == std::string::npos) {
        if (enforceToolRegistry && streamingContent.size() - streamingPosition > MAX_NATIVE_TOOL_CANDIDATE_BYTES) {
            rejectedCandidate = true;
            currentState = State::ToolCallParameters;
            return parseToolCallParametersState();
        }
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
    if (enforceToolRegistry) {
        toolName = streamingContent.substr(streamingPosition, argsPos - streamingPosition);
        trim(toolName);
        if (toolName.rfind(TOOL_CALL_NAME_PREFIX, 0) == 0)
            toolName.erase(0, TOOL_CALL_NAME_PREFIX.size());
        else if (!toolName.empty() && toolName.front() == ':')
            toolName.erase(toolName.begin());
        trim(toolName);
    }
    rejectedCandidate = enforceToolRegistry && (!saneToolName(toolName) || allowedToolNames.count(toolName) == 0);
    this->toolCall = ToolCall{generateRandomId(), toolName, ""};
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);
    this->streamingPosition = argsPos + TOOL_ARGS_START_INDICATOR.length();
    argsOpen = streamingContent[argsPos];
    this->currentState = State::ToolCallParameters;
    return true;
}

bool Gemma4ToolParser::parseToolCallParametersState() {
    if (enforceToolRegistry) {
        const size_t nextStart = streamingContent.find(TOOL_CALL_START_TAG, streamingPosition);
        const size_t envelopeEnd = streamingContent.find(TOOL_CALL_END_TAG, streamingPosition);
        if (rejectedCandidate) {
            if (nextStart != std::string::npos && (envelopeEnd == std::string::npos || nextStart < envelopeEnd)) {
                streamingPosition = nextStart;
                toolCall = {};
                rejectedCandidate = false;
                currentState = State::Content;
                return true;
            }
            if (envelopeEnd == std::string::npos) {
                // Retain only the suffix needed to recognize a split end marker.
                const size_t keep = TOOL_CALL_END_TAG.size() - 1;
                if (streamingContent.size() > keep) {
                    streamingContent.erase(0, streamingContent.size() - keep);
                    streamingPosition = 0;
                }
                return false;
            }
            streamingPosition = envelopeEnd + TOOL_CALL_END_TAG.size();
            toolCall = {};
            currentState = State::ToolCallEnded;
            return true;
        }
        size_t malformedEnd = std::string::npos;
        bool overLimit = false;
        auto close = findMatchingContainerEnd(streamingContent, streamingPosition - 1, argsOpen, argsOpen == '(' ? ')' : '}', malformedEnd, overLimit);
        if (!close) {
            if (malformedEnd != std::string::npos) {
                streamingPosition = malformedEnd + TOOL_CALL_END_TAG.size();
                toolCall = {};
                currentState = State::ToolCallEnded;
                return true;
            }
            if (overLimit) {
                rejectedCandidate = true;
                return parseToolCallParametersState();
            }
            return false;
        }
        const auto arguments = parseNativeArgumentsBody(streamingContent.substr(streamingPosition, *close - streamingPosition));
        if (arguments)
            toolCall.arguments = *arguments;
        else
            toolCall = {};
        streamingPosition = *close + 1;
        currentState = State::ToolCallEnded;
        return true;
    }
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
        if (enforceToolRegistry) {
            auto gap = streamingContent.substr(streamingPosition, nextToolCallPos - streamingPosition);
            trim(gap);
            if (!gap.empty()) {
                rejectedCandidate = true;
                currentState = State::ToolCallParameters;
                return parseToolCallParametersState();
            }
        }
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

ToolCallDelta Gemma4ToolParser::wrapCompletedToolCall() {
    ++toolCallIndex;
    std::string id = this->toolCall.id.empty() ? generateRandomId() : this->toolCall.id;
    return ToolCallDelta{this->toolCallIndex, std::move(id), this->toolCall.name, this->toolCall.arguments};
}

std::optional<Delta> Gemma4ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (!chunk.empty()) {
        this->streamingContent += chunk;
    }

    // Advance through silent transitions (start tag, opening `{`) in this chunk so a
    // complete call can be published at the unmasked `}` without waiting for AfterToolCall.
    // parseChunk still returns at most one delta; a later call in the same buffer is left
    // for the next parseChunk.
    for (;;) {
        const State stateBefore = this->currentState;
        const bool progressed = parseNewContent();
        if (this->currentState == State::ToolCallStarted) {
            if (stateBefore == State::ToolCallStarted && !progressed) {
                break;
            }
            continue;
        }
        if (this->currentState == State::ToolCallParameters) {
            if (!progressed) {
                break;
            }
            continue;
        }
        if (!progressed) {
            break;
        }
        if (this->currentState == State::ToolCallEnded) {
            if (this->toolCall.arguments.empty())
                continue;
            auto delta = wrapCompletedToolCall();
            this->toolCall = ToolCall{};
            return delta;
        }
        if (this->currentState == State::Content) {
            size_t contentEnd = this->streamingContent.find(TOOL_CALL_START_TAG, this->streamingPosition);
            if (enforceToolRegistry) {
                for (const auto& tag : parsingConfig.preambleStartTags) {
                    const size_t pos = streamingContent.find(tag, streamingPosition);
                    if (pos != std::string::npos && isLogicalBoundary(streamingContent, pos) &&
                        (contentEnd == std::string::npos || pos < contentEnd))
                        contentEnd = pos;
                }
            }
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
            continue;
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        // STOP may arrive with a complete `}` already buffered but unpublished if the
        // last NONE chunk only reached ToolCallParameters. Publish once, then clear so
        // a missing end tag cannot re-emit arguments.
        if (this->currentState == State::ToolCallParameters) {
            parseToolCallParametersState();
        }

        if ((this->currentState == State::ToolCallParameters || this->currentState == State::ToolCallEnded) && !this->toolCall.arguments.empty()) {
            auto delta = wrapCompletedToolCall();
            this->toolCall = ToolCall{};
            return delta;
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
