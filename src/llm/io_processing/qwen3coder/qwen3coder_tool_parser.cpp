//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <openvino/genai/tokenizer.hpp>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <stack>
#include <vector>

#include "rapidjson/error/en.h"

#include "src/llm/io_processing/utils.hpp"
#include "src/logging.hpp"
#include "src/utils/rapidjson_utils.hpp"
#include "qwen3coder_tool_parser.hpp"
#include "tool_schema_profile.hpp"

namespace ovms {
const std::string Qwen3CoderToolParser::TOOL_START_TAG = "<tool_call>";
const std::string Qwen3CoderToolParser::FUNCTION_NAME_TAG = "<function=";
const std::string Qwen3CoderToolParser::XML_TAG_END = ">";
const std::string Qwen3CoderToolParser::PARAMETER_NAME_TAG = "<parameter=";
const std::string Qwen3CoderToolParser::PARAMETER_END_TAG = "</parameter>";
const std::string Qwen3CoderToolParser::FUNCTION_END_TAG = "</function>";
const std::string Qwen3CoderToolParser::TOOL_END_TAG = "</tool_call>";

Status Qwen3CoderToolParserImpl::removeToolCallsFromContentIfNeeded(std::string& outContent) {
    // Generation can be truncated mid-tool-call (max_tokens hit, or eos suppressed) so an opening
    // "<tool_call>"/"<function=" is recorded with no matching close. That leaves begin with more
    // entries than end. The unterminated call is always the most recent one (top of the begin
    // stack), so drop it -- erasing from its start to end-of-content -- rather than bailing and
    // leaving every (including completed) block in the content returned to the user.
    while (toolCallPositions.begin.size() > toolCallPositions.end.size()) {
        auto posBegin = toolCallPositions.begin.top();
        toolCallPositions.begin.pop();
        if (posBegin <= outContent.size()) {
            SPDLOG_TRACE("Removing unterminated tool call from outContent begin:{} to end", posBegin);
            outContent.erase(posBegin);
        }
    }
    if (toolCallPositions.begin.size() != toolCallPositions.end.size()) {
        SPDLOG_DEBUG("Mismatched tool tags, begin: {}, end: {}", toolCallPositions.begin.size(), toolCallPositions.end.size());
        return Status(StatusCode::INTERNAL_ERROR, "Mismatched tool tags");
    }
    while (!toolCallPositions.begin.empty() && !toolCallPositions.end.empty()) {
        auto posBegin = toolCallPositions.begin.top();
        auto posEnd = toolCallPositions.end.top();
        SPDLOG_TRACE("Removing tool call from outContent begin:{}, end:{}, removing:{}", posBegin, posEnd, outContent.substr(posBegin, posEnd - posBegin));

        outContent.erase(posBegin, posEnd - posBegin);
        toolCallPositions.begin.pop();
        toolCallPositions.end.pop();
    }
    return StatusCode::OK;
}
#define DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(TAG)                         \
    auto pos = this->streamContent.find(TAG, this->getLastProcessedPosition()); \
    if (pos == std::string::npos) {                                             \
        SPDLOG_TRACE("Did not find: {}", TAG);                                  \
        break;                                                                  \
    }

void Qwen3CoderToolParserImpl::addParameterToCurrentFunctionDoc(std::string& parameterValueAsString) {
    if (this->removeNewlineAroundParameters)
        trimNewline(parameterValueAsString);
    // now we have parameter value in string format. We need to use toolsSchemas to determine if it is string, number, bool, array or object
    auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
    auto& currentFunctionArgsDoc = this->currentFunction.argumentsAsDocument;
    auto& allocator = currentFunctionArgsDoc.GetAllocator();
    auto& key = this->currentParameterName;
    if (key.find('\0') != std::string::npos || parameterValueAsString.find('\0') != std::string::npos) {
        currentFunction.invalidTypedArgument = true;
        return;
    }
    if (currentFunctionArgsDoc.HasMember(key.c_str())) {
        // Duplicate XML arguments have no unambiguous executable meaning.
        currentFunction.invalidTypedArgument = true;
        return;
    }
    rapidjson::Value keyVal(key.c_str(), allocator);
    ParameterType expectedType = ParameterType::UNKNOWN;
    if (paramIt != this->toolsParametersTypeMap.end()) {
        const auto p = paramIt->second.find(currentParameterName);
        if (p != paramIt->second.end())
            expectedType = p->second;
    }
    if (expectedType == ParameterType::STRING) {
        // Gemma encodes delimited argument bytes once. A JSON parse of the same
        // text turns \t into a tab and \\ into one backslash. A string parameter
        // is those bytes. A clean wrapping quote pair is only a delimiter.
        std::string literal = parameterValueAsString;
        trimSurroundingQuotes(literal);
        rapidjson::Value literalValue;
        literalValue.SetString(literal.c_str(), static_cast<rapidjson::SizeType>(literal.size()), allocator);
        currentFunctionArgsDoc.AddMember(keyVal, literalValue, allocator);
        return;
    }
    const bool losslessTypedValue = expectedType == ParameterType::NUMBER ||
                                    expectedType == ParameterType::ARRAY || expectedType == ParameterType::OBJECT;
    if (losslessTypedValue || expectedType == ParameterType::UNKNOWN) {
        const auto exact = normalizeJsonLosslessly(parameterValueAsString);
        const bool spellingOk = exact.has_value() && !exact->empty() &&
            ((expectedType == ParameterType::NUMBER && ((*exact)[0] == '-' || std::isdigit(static_cast<unsigned char>((*exact)[0])))) ||
             (expectedType == ParameterType::ARRAY && (*exact)[0] == '[') ||
             (expectedType == ParameterType::OBJECT && (*exact)[0] == '{') ||
             (expectedType == ParameterType::UNKNOWN &&
                 ((*exact)[0] == '-' || std::isdigit(static_cast<unsigned char>((*exact)[0])) || (*exact)[0] == '[' || (*exact)[0] == '{')));
        if (losslessTypedValue && !spellingOk) {
            SPDLOG_WARN("Dropping Qwen tool call with invalid typed argument: function={} parameter={}",
                currentFunction.name, currentParameterName);
            currentFunction.invalidTypedArgument = true;
            return;
        }
        if (spellingOk) {
            // The SAX pass has validated the complete value. Keep only a member
            // placeholder in the DOM: publication uses exactJsonMembers, so no
            // numeric token needs to be converted through a DOM number type.
            rapidjson::Value placeholder;
            currentFunctionArgsDoc.AddMember(keyVal, placeholder, allocator);
            currentFunction.exactJsonMembers.emplace(key, *exact);
            return;
        }
    }
    rapidjson::Document temp;
    if (paramIt != this->toolsParametersTypeMap.end()) {
        auto paramJt = paramIt->second.find(currentParameterName);
        if (paramJt != paramIt->second.end() && (paramJt->second == ParameterType::BOOLEAN)) {
            normalizeBooleanString(parameterValueAsString);
        }
    }
    temp.Parse(parameterValueAsString.c_str());
    // Known non-string JSON types must never silently degrade to strings: an
    // integer argument containing a half-written number is not an executable call.
    const bool typedValue = expectedType == ParameterType::NUMBER ||
                            expectedType == ParameterType::BOOLEAN ||
                            expectedType == ParameterType::ARRAY ||
                            expectedType == ParameterType::OBJECT;
    if (typedValue) {
        const bool matches = !temp.HasParseError() &&
            ((expectedType == ParameterType::NUMBER && temp.IsNumber()) ||
             (expectedType == ParameterType::BOOLEAN && temp.IsBool()) ||
             (expectedType == ParameterType::ARRAY && temp.IsArray()) ||
             (expectedType == ParameterType::OBJECT && temp.IsObject()));
        if (!matches) {
            SPDLOG_WARN("Dropping Qwen tool call with invalid typed argument: function={} parameter={}",
                currentFunction.name, currentParameterName);
            currentFunction.invalidTypedArgument = true;
            return;
        }
    }
    if (temp.HasParseError()) {
        rapidjson::ParseErrorCode errorCode = temp.GetParseError();
        size_t errorOffset = temp.GetErrorOffset();
        // If error occurred during parsing, we will insert parameter as string
        SPDLOG_TRACE("RapidJSON can not parse parameter: {} with value: {}; error at offset: {}; code: {}; falling back to inserting value as string", this->currentParameterName, parameterValueAsString, errorOffset, rapidjson::GetParseError_En(errorCode));
        // since we are unable to parse with rapidjson generated parameter will be inserted as is as string
        rapidjson::Value v;
        v.SetString(parameterValueAsString.c_str(), static_cast<rapidjson::SizeType>(parameterValueAsString.size()), allocator);
        if (!currentFunctionArgsDoc.HasMember(keyVal)) {
            currentFunctionArgsDoc.AddMember(keyVal, v, allocator);
        } else {
            SPDLOG_DEBUG("Parameter: {} already exists in document", key);
        }
    } else {
        rapidjson::Value valueCopy;
        valueCopy.CopyFrom(temp, allocator);
        // Add or overwrite key in the main document
        if (!currentFunctionArgsDoc.HasMember(keyVal)) {
            SPDLOG_TRACE("Will add key:{} val:{} type:{}", key, parameterValueAsString, jsonTypeOf(valueCopy));
            currentFunctionArgsDoc.AddMember(keyVal, valueCopy, allocator);
        } else {
            SPDLOG_DEBUG("Parameter: {} already exists in document.", key);
        }
    }
}

bool Qwen3CoderToolParserImpl::parseUntilStateChange(ToolCalls_t& toolCalls) {
    SPDLOG_TRACE("State: {}", this->currentState);
    auto previousState = this->currentState;
    if (currentState != State::Content) {
        const auto find = [&](const std::string& tag) { return streamContent.find(tag, lastProcessedPosition); };
        size_t expected = std::string::npos;
        switch (currentState) {
        case State::InsideToolCall: expected = find(Qwen3CoderToolParser::FUNCTION_NAME_TAG); break;
        case State::InsideFunctionName:
        case State::InsideParameterName: expected = find(Qwen3CoderToolParser::XML_TAG_END); break;
        case State::InsideFunction:
            expected = std::min(find(Qwen3CoderToolParser::PARAMETER_NAME_TAG), find(Qwen3CoderToolParser::FUNCTION_END_TAG));
            break;
        case State::InsideParameter: expected = find(Qwen3CoderToolParser::PARAMETER_END_TAG); break;
        case State::AfterFunction: expected = find(Qwen3CoderToolParser::TOOL_END_TAG); break;
        case State::Content: break;
        }
        const auto close = find(Qwen3CoderToolParser::TOOL_END_TAG);
        const auto open = find(Qwen3CoderToolParser::TOOL_START_TAG);
        // Transport markers are reserved boundaries, including inside malformed
        // names/values. No state may borrow a later envelope's delimiter.
        if ((close != std::string::npos && close < expected) ||
            (open != std::string::npos && open < expected)) {
            lastProcessedPosition = close < open ? close + Qwen3CoderToolParser::TOOL_END_TAG.size() : open;
            toolCallPositions.end.push(lastProcessedPosition);
            currentFunction.clear();
            currentParameterName.clear();
            currentState = State::Content;
            return true;
        }
    }
    switch (this->currentState) {
    case State::Content: {
        // normally we expect <tool_call> tag but we observed that sometimes model generates <function=...> directly
        // so we will check for both tags and handle accordingly
        auto posTool = this->streamContent.find(Qwen3CoderToolParser::TOOL_START_TAG, this->getLastProcessedPosition());
        auto posFunc = this->streamContent.find(Qwen3CoderToolParser::FUNCTION_NAME_TAG, this->getLastProcessedPosition());
        if (posFunc == std::string::npos && posTool == std::string::npos) {
            SPDLOG_TRACE("Did not find: {} or {}", Qwen3CoderToolParser::TOOL_START_TAG, Qwen3CoderToolParser::FUNCTION_NAME_TAG);
        } else if (posTool < posFunc) {
            // found <tool_call> first
            this->lastProcessedPosition = posTool + Qwen3CoderToolParser::TOOL_START_TAG.length();
            this->currentState = State::InsideToolCall;
            this->toolCallPositions.begin.push(posTool);
        } else {
            // found <function=...> first, we will assume <tool_call> is missing
            SPDLOG_DEBUG("Did not find: {}, assuming it should exist", Qwen3CoderToolParser::TOOL_START_TAG);
            this->lastProcessedPosition = posFunc + Qwen3CoderToolParser::FUNCTION_NAME_TAG.length();
            this->currentState = State::InsideFunctionName;
            this->toolCallPositions.begin.push(posFunc);
        }
        break;
    }
    case State::InsideToolCall: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::FUNCTION_NAME_TAG);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::FUNCTION_NAME_TAG.length();
        this->currentState = State::InsideFunctionName;
        break;
    }
    case State::InsideFunctionName: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::XML_TAG_END);
        this->currentFunction.name = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        // Some models quote the tag attribute (<function="name">); the quotes are delimiters, not part of the name.
        trimSurroundingQuotes(this->currentFunction.name);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::XML_TAG_END.length();
        this->currentState = State::InsideFunction;
        break;
    }
    case State::InsideFunction: {
        auto funcEnd = streamContent.find(Qwen3CoderToolParser::FUNCTION_END_TAG, this->lastProcessedPosition);
        auto paramStart = streamContent.find(Qwen3CoderToolParser::PARAMETER_NAME_TAG, this->lastProcessedPosition);
        if (funcEnd == std::string::npos && paramStart == std::string::npos) {
        } else if (paramStart < funcEnd) {  // next parameter
            this->lastProcessedPosition = paramStart + Qwen3CoderToolParser::PARAMETER_NAME_TAG.length();
            this->currentState = State::InsideParameterName;
        } else {  // end of function
            this->lastProcessedPosition = funcEnd + Qwen3CoderToolParser::FUNCTION_END_TAG.length();
            this->currentState = State::AfterFunction;
        }
        break;
    }
    case State::InsideParameterName: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::XML_TAG_END);
        this->currentParameterName = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        // Some models quote the tag attribute (<parameter="name">); the quotes are delimiters, not part of the name.
        trimSurroundingQuotes(this->currentParameterName);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::XML_TAG_END.length();
        this->currentState = State::InsideParameter;
        break;
    }
    case State::InsideParameter: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::PARAMETER_END_TAG);
        std::string parameterValueAsString(streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition));
        addParameterToCurrentFunctionDoc(parameterValueAsString);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::PARAMETER_END_TAG.length();
        this->currentState = State::InsideFunction;
        break;
    }
    case State::AfterFunction: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::TOOL_END_TAG);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::TOOL_END_TAG.length();
        this->currentState = State::Content;
        std::string argumentsAsString;
        {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            auto& doc = this->currentFunction.argumentsAsDocument;
            auto& exact = this->currentFunction.exactJsonMembers;
            writer.StartObject();
            for (auto member = doc.MemberBegin(); member != doc.MemberEnd(); ++member) {
                writer.Key(member->name.GetString(), member->name.GetStringLength());
                const auto spelled = exact.find(member->name.GetString());
                if (spelled != exact.end()) {
                    const std::string& raw = spelled->second;
                    rapidjson::Type type = rapidjson::kNumberType;
                    if (!raw.empty() && raw.front() == '{')
                        type = rapidjson::kObjectType;
                    else if (!raw.empty() && raw.front() == '[')
                        type = rapidjson::kArrayType;
                    writer.RawValue(raw.data(), static_cast<rapidjson::SizeType>(raw.size()), type);
                } else {
                    member->value.Accept(writer);
                }
            }
            writer.EndObject();
            argumentsAsString = buffer.GetString();
        }
        if (!this->currentFunction.invalidTypedArgument) {
            ToolCall toolCall{generateRandomId(), this->currentFunction.name, argumentsAsString};
            SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
            toolCalls.emplace_back(std::move(toolCall));
        }
        this->currentFunction.clear();
        this->toolCallPositions.end.push(this->lastProcessedPosition);
        break;
    }
    }
    return previousState != this->currentState;
}
std::optional<ToolCalls_t> Qwen3CoderToolParserImpl::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        return std::nullopt;
    }
    ToolCalls_t toolCalls;
    this->streamContent += chunk;
    while (parseUntilStateChange(toolCalls)) {
    }
    // ASSUMPTION
    // in streaming we will only have either one tool call or one function name at a time
    // while underlying parser can handle multiple tool calls in one go, we will not use that here
    // as there's no way to send multiple tool calls at once in streaming
    if (!toolCalls.empty()) {
        return std::move(toolCalls);
    }
    return std::nullopt;
}

std::optional<ToolCalls_t> Qwen3CoderToolParserImpl::finalizeOnGenerationEnd() {
    // The closing </function> is the only reliable evidence that the model has
    // finished all argument values.  STOP alone does not establish the boundary:
    // an open <parameter> may still contain a truncated path, command or JSON.
    if (currentState != State::AfterFunction) {
        resetParsingState();
        return std::nullopt;
    }
    // The complete function and its parameters are already parsed; only the
    // outer transport envelope is missing.  Recover exactly that closing tag.
    streamContent += Qwen3CoderToolParser::TOOL_END_TAG;
    ToolCalls_t recovered;
    while (parseUntilStateChange(recovered)) {
    }
    // Keep toolCallPositions for unary cleanup, as on the normal completion path.
    resetParsingState();
    if (recovered.empty()) {
        return std::nullopt;
    }
    return recovered;
}

Qwen3CoderToolParser::Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas,
    std::optional<OutputParsingConfig> configOverride) :
    BaseOutputParser(tokenizer, [&]() {
        if (configOverride.has_value())
            return std::move(*configOverride);
        OutputParsingConfig cfg;
        cfg.startTags = {TOOL_START_TAG, FUNCTION_NAME_TAG};
        return cfg;
    }()),
    toolSchemas(toolSchemas),
    toolsParametersTypes(createToolsParametersTypesMap(toolSchemas)),
    streamParser(this->toolsParametersTypes) {
}

std::optional<std::string> Qwen3CoderToolParserImpl::getCurrentFunctionName() const {
    if (this->currentFunction.name.empty()) {
        return std::nullopt;
    }
    return this->currentFunction.name;
}
bool Qwen3CoderToolParser::isRegisteredToolName(const std::string& name) const {
    return toolSchemas.find(name) != toolSchemas.end();
}

bool Qwen3CoderToolParser::hasValidArguments(const ToolCall& toolCall) const {
    const auto it = toolSchemas.find(toolCall.name);
    return it != toolSchemas.end() &&
           validateQwenXmlToolArguments(it->second.stringRepr, toolCall.arguments);
}

void Qwen3CoderToolParser::queueFullDeltas(const ToolCalls_t& toolCalls) {
    // The low-level parser may complete more than one envelope at once.
    // Queue name/id and full arguments separately for each in original order.
    for (const auto& toolCall : toolCalls) {
        if (!isRegisteredToolName(toolCall.name)) {
            SPDLOG_WARN("Ignoring completed Qwen call to undeclared tool: {}", toolCall.name);
            continue;
        }
        if (!hasValidArguments(toolCall)) {
            SPDLOG_WARN("Ignoring Qwen tool call with invalid schema or arguments: {}", toolCall.name);
            continue;
        }
        // OpenAI SSE has no retraction event. Allocate a public identity only
        // after validation, then queue its header and arguments atomically.
        const int index = ++toolCallIndex;
        std::string id = toolCall.id;
        while (!publishedCallIds.insert(id).second)
            id = generateRandomId();
        pendingDeltas.emplace_back(ToolCallDelta{index, std::move(id), toolCall.name, ""});
        SPDLOG_DEBUG("Complete Qwen delta: index={} arguments={}", index, toolCall.arguments);
        pendingDeltas.emplace_back(ToolCallDelta{index, std::nullopt, std::nullopt, toolCall.arguments});
    }
}

std::optional<Delta> Qwen3CoderToolParser::takeQueuedDelta() {
    if (pendingDeltas.empty()) {
        return std::nullopt;
    }
    Delta result = std::move(pendingDeltas.front());
    pendingDeltas.pop_front();
    return result;
}

std::optional<Delta> Qwen3CoderToolParser::drainPendingDelta() {
    return takeQueuedDelta();
}

std::optional<Delta> Qwen3CoderToolParser::parseChunk(const std::string& newChunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_DEBUG("Chunk: '{}', finishReason: {}", newChunk, static_cast<int>(finishReason));
    if (!newChunk.empty()) {
        if (auto completed = streamParser.parseChunk(newChunk)) {
            queueFullDeltas(*completed);
        }
    }

    // Only natural STOP may justify bounded recovery; LENGTH, cancellation and other
    // finish reasons must never turn an incomplete argument into an executable call.
    if (finishReason == ov::genai::GenerationFinishReason::STOP) {
        if (auto recovered = streamParser.finalizeOnGenerationEnd()) {
            queueFullDeltas(*recovered);
        }
    }

    return takeQueuedDelta();
}
Qwen3CoderToolParserImpl::Qwen3CoderToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap) :
    toolsParametersTypeMap(toolsParametersTypeMap) {}
}  // namespace ovms
