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
#include <string>
#include <stack>
#include <vector>

#include "rapidjson/error/en.h"

#include "src/llm/io_processing/utils.hpp"
#include "src/logging.hpp"
#include "src/utils/rapidjson_utils.hpp"
#include "qwen3coder_tool_parser.hpp"

namespace ovms {
const std::string Qwen3CoderToolParser::TOOL_START_TAG = "<tool_call>";
const std::string Qwen3CoderToolParser::FUNCTION_NAME_TAG = "<function=";
const std::string Qwen3CoderToolParser::XML_TAG_END = ">";
const std::string Qwen3CoderToolParser::PARAMETER_NAME_TAG = "<parameter=";
const std::string Qwen3CoderToolParser::PARAMETER_END_TAG = "</parameter>";
const std::string Qwen3CoderToolParser::FUNCTION_END_TAG = "</function>";
const std::string Qwen3CoderToolParser::TOOL_END_TAG = "</tool_call>";

Status Qwen3CoderToolParserImpl::removeToolCallsFromContentIfNeeded(std::string& outContent) {
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
    rapidjson::Value keyVal(key.c_str(), allocator);
    rapidjson::Document temp;
    if (paramIt != this->toolsParametersTypeMap.end()) {
        auto paramJt = paramIt->second.find(currentParameterName);
        if (paramJt != paramIt->second.end() && (paramJt->second == ParameterType::BOOLEAN)) {
            normalizeBooleanString(parameterValueAsString);
        }
    }
    temp.Parse(parameterValueAsString.c_str());
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
        if (paramIt != this->toolsParametersTypeMap.end()) {
            auto paramJt = paramIt->second.find(currentParameterName);
            if (paramJt != paramIt->second.end() && (paramJt->second == ParameterType::STRING)) {
                enforceStringValue(valueCopy, allocator);
            }
        }
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

void Qwen3CoderToolParser::lazyFillInitToolParametersTypesMap() {
    if (this->filledParametersTypesMap) {
        return;
    }
    SPDLOG_DEBUG("Filling tools parameters types map");
    this->toolsParametersTypes = createToolsParametersTypesMap(this->toolSchemas);
    this->filledParametersTypesMap = true;
    SPDLOG_DEBUG("Qwen3CoderToolParser created with {} tools", this->toolsParametersTypes.size());
}

Qwen3CoderToolParser::Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas,
                                             std::optional<ParsingConfig> configOverride) :
    BaseOutputParser(tokenizer, [&]() {
        if (configOverride.has_value()) return std::move(*configOverride);
        ParsingConfig cfg;
        cfg.startTags = {TOOL_START_TAG, FUNCTION_NAME_TAG};
        return cfg;
    }()),
    toolSchemas(toolSchemas),
    streamParser(this->toolsParametersTypes) {
}

std::optional<std::string> Qwen3CoderToolParserImpl::getCurrentFunctionName() const {
    if (this->currentFunction.name.empty()) {
        return std::nullopt;
    }
    return this->currentFunction.name;
}
std::optional<rapidjson::Document> Qwen3CoderToolParser::sendFullDelta(const ToolCalls_t& toolCalls) {
    if (toolCalls.size() != 1) {
        SPDLOG_ERROR("For streaming we expected one tool call, got: {}", toolCalls.size());
        // TODO we should return status code but this require change of parsers API
        throw std::runtime_error("For streaming we expected one tool call");
    }
    auto& toolCall = toolCalls[0];
    rapidjson::Document argsDelta;
    argsDelta.Parse(toolCall.arguments.c_str());
    this->returnedCompleteDeltas.insert(this->toolCallIndex);
    rapidjson::Document argumentsWrapper;
    argumentsWrapper.SetObject();
    rapidjson::Document::AllocatorType& allocator = argumentsWrapper.GetAllocator();
    // now we need to add string toolCall.arguments to argumentsWrapper under "arguments" key
    rapidjson::Value toolCallsString(rapidjson::kStringType);
    toolCallsString.SetString(toolCall.arguments.c_str(), allocator);
    SPDLOG_TRACE("Tool call arguments string: {}", toolCall.arguments);

    argumentsWrapper.AddMember("arguments", toolCallsString, allocator);
    auto currentDelta = wrapDelta(argumentsWrapper, this->toolCallIndex);
    SPDLOG_DEBUG("First delta doc: {}", documentToString(currentDelta));
    return currentDelta;
}

std::optional<rapidjson::Document> Qwen3CoderToolParser::sendFirstDeltaIfNeeded(const std::string& toolCallName) {
    if (this->returnedFirstDeltas.size() == (this->returnedCompleteDeltas.size() + 1)) {
        SPDLOG_TRACE("Skipping first delta, already sent for current function, returnedFirstDeltas.size(): {} returnedCompleteDeltas.size(): {}", returnedFirstDeltas.size(), returnedCompleteDeltas.size());
        // we can skip sending first delta since we sent it for current function
        return std::nullopt;
    }
    int toolCallId = ++this->toolCallIndex;
    rapidjson::Document doc = wrapFirstDelta(toolCallName, toolCallId);
    this->currentJson.CopyFrom(doc, this->currentJson.GetAllocator());
    this->returnedFirstDeltas.insert(toolCallId);
    SPDLOG_DEBUG("First delta doc: {}", documentToString(doc));
    return doc;
}

std::optional<rapidjson::Document> Qwen3CoderToolParser::parseChunk(const std::string& newChunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    // streamParser will return optional toolCalls when a tool call is completed
    // if toolCalls is returned, we need to wrap it in the required JSON structure and return it
    // if toolCalls is not returned, but we are insideFunction state, we need to return the first delta with function name once
    // otherwise nullopt
    SPDLOG_DEBUG("Chunk: '{}', finishReason: {}", newChunk, static_cast<int>(finishReason));
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
Qwen3CoderToolParserImpl::Qwen3CoderToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap) :
    toolsParametersTypeMap(toolsParametersTypeMap) {}
}  // namespace ovms
