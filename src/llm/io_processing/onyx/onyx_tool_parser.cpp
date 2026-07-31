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

#include <openvino/genai/tokenizer.hpp>
#include <algorithm>
#include <stack>
#include <string>
#include <vector>

#include "rapidjson/error/en.h"

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/utils.hpp"
#include "src/logging.hpp"
#include "src/utils/rapidjson_utils.hpp"
#include "src/llm/io_processing/onyx/onyx_tool_parser.hpp"

namespace ovms {

const std::string OnyxToolParser::TOOL_START_TAG = "<atem:function_calls>";
const std::string OnyxToolParser::TOOL_END_TAG = "</atem:function_calls>";
const std::string OnyxToolParser::FUNCTION_NAME_TAG = "<atem:invoke name=\"";
const std::string OnyxToolParser::FUNCTION_END_TAG = "</atem:invoke>";
const std::string OnyxToolParser::PARAMETER_NAME_TAG = "<atem:parameter name=\"";
const std::string OnyxToolParser::PARAMETER_END_TAG = "</atem:parameter>";
const std::string OnyxToolParser::NAME_ATTR_END_TAG = "\">";

// Static empty map the default-constructed impl binds its const-ref member to (used by the
// direct-impl unit tests, which pass plain-string values that need no schema typing).
static const ToolsParameterTypeMap_t EMPTY_TOOLS_PARAMETER_TYPE_MAP{};

OnyxToolParserImpl::OnyxToolParserImpl() :
    toolsParametersTypeMap(EMPTY_TOOLS_PARAMETER_TYPE_MAP) {}

OnyxToolParserImpl::OnyxToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap) :
    toolsParametersTypeMap(toolsParametersTypeMap) {}

// parseToolSchema / createToolsParametersTypesMap now live in base_output_parser
// (shared with Qwen3CoderToolParser, Minicpm5ToolParser, ...); trimNewline,
// jsonTypeOf and enforceStringValue come from io_processing/utils. This parser
// reuses them instead of keeping its own copies.

void OnyxToolParserImpl::addParameterToCurrentFunctionDoc(std::string& parameterValueAsString) {
    if (this->removeNewlineAroundParameters)
        trimNewline(parameterValueAsString);
    // Serialize the untyped ATEM value into JSON using the tool schema to decide the type.
    auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
    auto& currentFunctionArgsDoc = this->currentFunction.argumentsAsDocument;
    auto& allocator = currentFunctionArgsDoc.GetAllocator();
    auto& key = this->currentParameterName;
    rapidjson::Value keyVal(key.c_str(), allocator);
    rapidjson::Document temp;
    if (paramIt != this->toolsParametersTypeMap.end()) {
        auto paramJt = paramIt->second.find(currentParameterName);
        if (paramJt != paramIt->second.end() && (paramJt->second == ParameterType::BOOLEAN)) {
            if (parameterValueAsString == "True" || parameterValueAsString == "TRUE") {
                parameterValueAsString = "true";
            } else if (parameterValueAsString == "False" || parameterValueAsString == "FALSE") {
                parameterValueAsString = "false";
            }
        }
    }
    temp.Parse(parameterValueAsString.c_str());
    if (temp.HasParseError()) {
        // Not valid JSON -> insert as a string value.
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
        if (paramIt != this->toolsParametersTypeMap.end()) {
            auto paramJt = paramIt->second.find(currentParameterName);
            if (paramJt != paramIt->second.end() && (paramJt->second == ParameterType::STRING)) {
                enforceStringValue(valueCopy, allocator);
            }
        }
        if (!currentFunctionArgsDoc.HasMember(keyVal)) {
            SPDLOG_TRACE("Will add key:{} val:{} type:{}", key, parameterValueAsString, jsonTypeOf(valueCopy));
            currentFunctionArgsDoc.AddMember(keyVal, valueCopy, allocator);
        } else {
            SPDLOG_DEBUG("Parameter: {} already exists in document.", key);
        }
    }
}

#define DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(TAG)                         \
    auto pos = this->streamContent.find(TAG, this->getLastProcessedPosition()); \
    if (pos == std::string::npos) {                                             \
        SPDLOG_TRACE("Did not find: {}", TAG);                                  \
        break;                                                                  \
    }

bool OnyxToolParserImpl::parseUntilStateChange(ToolCalls_t& toolCalls) {
    SPDLOG_TRACE("State: {}", this->currentState);
    auto previousState = this->currentState;
    switch (this->currentState) {
    case State::Content: {
        // Normally "<atem:function_calls>" precedes "<atem:invoke name=", but tolerate a
        // missing wrapper (mirrors qwen3coder's <tool_call>/<function=> handling).
        auto posTool = this->streamContent.find(OnyxToolParser::TOOL_START_TAG, this->getLastProcessedPosition());
        auto posFunc = this->streamContent.find(OnyxToolParser::FUNCTION_NAME_TAG, this->getLastProcessedPosition());
        if (posFunc == std::string::npos && posTool == std::string::npos) {
            SPDLOG_TRACE("Did not find: {} or {}", OnyxToolParser::TOOL_START_TAG, OnyxToolParser::FUNCTION_NAME_TAG);
        } else if (posTool < posFunc) {
            this->lastProcessedPosition = posTool + OnyxToolParser::TOOL_START_TAG.length();
            this->currentState = State::InsideToolCall;
            this->toolCallPositions.begin.push(posTool);
        } else {
            SPDLOG_DEBUG("Did not find: {}, assuming it should exist", OnyxToolParser::TOOL_START_TAG);
            this->lastProcessedPosition = posFunc + OnyxToolParser::FUNCTION_NAME_TAG.length();
            this->currentState = State::InsideFunctionName;
            this->toolCallPositions.begin.push(posFunc);
        }
        break;
    }
    case State::InsideToolCall: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(OnyxToolParser::FUNCTION_NAME_TAG);
        this->lastProcessedPosition = pos + OnyxToolParser::FUNCTION_NAME_TAG.length();
        this->currentState = State::InsideFunctionName;
        break;
    }
    case State::InsideFunctionName: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(OnyxToolParser::NAME_ATTR_END_TAG);
        this->currentFunction.name = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        this->lastProcessedPosition = pos + OnyxToolParser::NAME_ATTR_END_TAG.length();
        this->currentState = State::InsideFunction;
        break;
    }
    case State::InsideFunction: {
        auto funcEnd = streamContent.find(OnyxToolParser::FUNCTION_END_TAG, this->lastProcessedPosition);
        auto paramStart = streamContent.find(OnyxToolParser::PARAMETER_NAME_TAG, this->lastProcessedPosition);
        if (funcEnd == std::string::npos && paramStart == std::string::npos) {
        } else if (paramStart < funcEnd) {  // next parameter
            this->lastProcessedPosition = paramStart + OnyxToolParser::PARAMETER_NAME_TAG.length();
            this->currentState = State::InsideParameterName;
        } else {  // end of function
            this->lastProcessedPosition = funcEnd + OnyxToolParser::FUNCTION_END_TAG.length();
            this->currentState = State::AfterFunction;
        }
        break;
    }
    case State::InsideParameterName: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(OnyxToolParser::NAME_ATTR_END_TAG);
        this->currentParameterName = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        this->lastProcessedPosition = pos + OnyxToolParser::NAME_ATTR_END_TAG.length();
        this->currentState = State::InsideParameter;
        break;
    }
    case State::InsideParameter: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(OnyxToolParser::PARAMETER_END_TAG);
        std::string parameterValueAsString(streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition));
        addParameterToCurrentFunctionDoc(parameterValueAsString);
        this->lastProcessedPosition = pos + OnyxToolParser::PARAMETER_END_TAG.length();
        this->currentState = State::InsideFunction;
        break;
    }
    case State::AfterFunction: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(OnyxToolParser::TOOL_END_TAG);
        this->lastProcessedPosition = pos + OnyxToolParser::TOOL_END_TAG.length();
        this->currentState = State::Content;
        std::string argumentsAsString;
        {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            this->currentFunction.argumentsAsDocument.Accept(writer);
            argumentsAsString = buffer.GetString();
        }
        ToolCall toolCall{generateRandomId(), this->currentFunction.name, argumentsAsString};
        SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
        toolCalls.emplace_back(std::move(toolCall));
        this->currentFunction.clear();
        this->toolCallPositions.end.push(this->lastProcessedPosition);
        break;
    }
    }
    return previousState != this->currentState;
}

std::optional<ToolCalls_t> OnyxToolParserImpl::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        return std::nullopt;
    }
    ToolCalls_t toolCalls;
    this->streamContent += chunk;
    while (parseUntilStateChange(toolCalls)) {
    }
    if (!toolCalls.empty()) {
        return std::move(toolCalls);
    }
    return std::nullopt;
}

std::optional<std::string> OnyxToolParserImpl::getCurrentFunctionName() const {
    if (this->currentFunction.name.empty()) {
        return std::nullopt;
    }
    return this->currentFunction.name;
}

Status OnyxToolParserImpl::removeToolCallsFromContentIfNeeded(std::string& outContent) {
    // Generation can be truncated mid-tool-call (max_tokens hit, or eos suppressed) so an opening
    // "<atem:function_calls>" is recorded with no matching "</atem:function_calls>" close. That
    // leaves begin with more entries than end. The unterminated call is always the most recent one
    // (top of the begin stack), so drop it -- erasing from its start to end-of-content -- rather
    // than bailing and leaving every (including completed) block in the content returned to the user.
    while (toolCallPositions.begin.size() > toolCallPositions.end.size()) {
        auto posBegin = toolCallPositions.begin.top();
        toolCallPositions.begin.pop();
        if (posBegin <= outContent.size()) {
            SPDLOG_TRACE("Removing unterminated tool call from outContent begin:{} to end, removing:{}", posBegin, outContent.substr(posBegin));
            outContent.erase(posBegin);
        }
    }
    if (toolCallPositions.begin.size() != toolCallPositions.end.size()) {
        // Unexpected shape (more closes than opens) -- leave content untouched to avoid corrupting it.
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

OnyxToolParser::OnyxToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas) :
    BaseOutputParser(tokenizer),
    toolSchemas(toolSchemas),
    streamParser(this->toolsParametersTypes) {
}

void OnyxToolParser::lazyFillInitToolParametersTypesMap() {
    if (this->filledParametersTypesMap) {
        return;
    }
    this->toolsParametersTypes = createToolsParametersTypesMap(this->toolSchemas);
    this->filledParametersTypesMap = true;
    SPDLOG_DEBUG("OnyxToolParser created with {} tools", this->toolsParametersTypes.size());
}

void OnyxToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // Unary is the single-shot edge case of streaming: drive the same streamParser with the
    // whole content as one chunk (mirrors Qwen3CoderToolParser::parse()).
    this->lazyFillInitToolParametersTypesMap();
    auto toolCallsOpt = this->streamParser.parseChunk(parsedOutput.content);
    if (!toolCallsOpt.has_value()) {
        SPDLOG_DEBUG("Parsing ended, no tool calls found");
        return;
    }
    parsedOutput.toolCalls = std::move(toolCallsOpt.value());
    for (const auto& toolCall : parsedOutput.toolCalls) {
        SPDLOG_DEBUG("Unary | Onyx Tool | id: [{}], name: [{}], arguments: [{}]", toolCall.id, toolCall.name, toolCall.arguments);
    }
    auto status = this->streamParser.removeToolCallsFromContentIfNeeded(parsedOutput.content);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to remove tool calls from content: {}", status.string());
    }
}

std::optional<rapidjson::Document> OnyxToolParser::sendFirstDeltaIfNeeded(const std::string& functionName) {
    if (this->returnedFirstDeltas.size() == (this->returnedCompleteDeltas.size() + 1)) {
        // already sent the first delta for the function currently being read
        return std::nullopt;
    }
    int currentToolCallIndex = ++this->toolCallIndex;
    rapidjson::Document doc = wrapFirstDelta(functionName, currentToolCallIndex);
    this->returnedFirstDeltas.insert(currentToolCallIndex);
    return doc;
}

std::optional<rapidjson::Document> OnyxToolParser::sendFullDelta(const ToolCalls_t& toolCalls) {
    // ASSUMPTION (mirrors Qwen3CoderToolParser): in streaming we only ever complete one tool
    // call per parseChunk() -- there is no way to send multiple tool calls in one delta.
    if (toolCalls.size() != 1) {
        SPDLOG_ERROR("For streaming we expected one tool call, got: {}", toolCalls.size());
        throw std::runtime_error("For streaming we expected one tool call");
    }
    const auto& toolCall = toolCalls[0];
    this->returnedCompleteDeltas.insert(this->toolCallIndex);
    rapidjson::Document argumentsWrapper;
    argumentsWrapper.SetObject();
    rapidjson::Value argumentsValue(toolCall.arguments.c_str(), static_cast<rapidjson::SizeType>(toolCall.arguments.size()), argumentsWrapper.GetAllocator());
    argumentsWrapper.AddMember("arguments", argumentsValue, argumentsWrapper.GetAllocator());
    return wrapDelta(argumentsWrapper, this->toolCallIndex);
}

std::optional<rapidjson::Document> OnyxToolParser::parseChunk(const std::string& newChunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason /*finishReason*/) {
    // streamParser returns assembled toolCalls once a call closes ("</atem:function_calls>");
    // until then, if the function name is already known, send its first delta once.
    this->lazyFillInitToolParametersTypesMap();
    if (newChunk.empty()) {
        return std::nullopt;
    }
    auto toolCallsOpt = this->streamParser.parseChunk(newChunk);
    if (toolCallsOpt.has_value()) {
        return this->sendFullDelta(toolCallsOpt.value());
    }
    auto functionNameOpt = this->streamParser.getCurrentFunctionName();
    if (functionNameOpt.has_value()) {
        return this->sendFirstDeltaIfNeeded(functionNameOpt.value());
    }
    return std::nullopt;
}
}  // namespace ovms
