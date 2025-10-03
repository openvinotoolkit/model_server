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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

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

static void trimNewline(std::string& str) {
    if (str.empty()) {
        return;
    }
    if (str.back() == '\n') {
        str.pop_back();
    }
    if (str.empty()) {
        return;
    }
    if (str.front() == '\n') {
        str.erase(str.begin());
    }
}

std::string Functool::parametersToJson() {
    std::ostringstream oss;
    oss << "{";
    size_t i = 0;
    for (const auto& [key, value] : this->parameters) {
        oss << "\"" << key << "\": ";
        oss << value;
        if (i++ + 1 < this->parameters.size()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

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
// Exemplary schemas
// {"type":"object","properties":{"location":{"type":"string"},"provide_temperature":{"type":"boolean"}},"required":["location"]}
// {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"The location for which to get the weather, in the format of 'City, State', such as 'San Francisco, CA' if State for the city exists. 'City, Country' if State for the city doesn't exist."},"unit":{"type":"string","description":"The unit of temperature for the weather report.","enum":["celsius","fahrenheit"],"default":"fahrenheit"}}}
static const ParametersTypeMap_t parseToolSchema(const std::string& functionName, const rapidjson::Value& schema) {
    // we want to create mapping of parameter name to parameter type
    SPDLOG_TRACE("Parse tool schema for tool: {}, schema: {}", functionName, schema.GetString());
    ParametersTypeMap_t result;
    if (!schema.IsObject()) {
        SPDLOG_DEBUG("Tool schema is not a JSON object for tool: {}, schema: {}", functionName, schema.GetString());
        return result;
    }
    if (!schema.HasMember("properties") || !schema["properties"].IsObject()) {
        SPDLOG_DEBUG("Tool schema does not have properties object for tool: {}, schema: {}", functionName, schema.GetString());
        return result;
    }
    const rapidjson::Value& properties = schema["properties"];
    for (auto it = properties.MemberBegin(); it != properties.MemberEnd(); ++it) {
        if (!it->value.IsObject()) {
            SPDLOG_DEBUG("Tool schema property: {} is not an object for tool: {}, schema: {}", it->name.GetString(), functionName, schema.GetString());
            continue;
        }
        if (!it->value.HasMember("type") || !it->value["type"].IsString()) {
            SPDLOG_DEBUG("Tool schema property: {} does not have type string for tool: {}, schema: {}", it->name.GetString(), functionName, schema.GetString());
            continue;
        }
        std::string paramName = it->name.GetString();
        std::string typeStr = it->value["type"].GetString();
        ParameterType type = ParameterType::UNKNOWN;
        if (typeStr == "string") {
            type = ParameterType::STRING;
        } else if (typeStr == "number" || typeStr == "integer") {
            type = ParameterType::NUMBER;
        } else if (typeStr == "boolean") {
            type = ParameterType::BOOLEAN;
        } else if (typeStr == "array") {
            type = ParameterType::ARRAY;
        } else if (typeStr == "object") {
            type = ParameterType::OBJECT;
        } else {
            SPDLOG_DEBUG("Tool schema property: {} has unknown type: {} for tool: {}, schema: {}", paramName, typeStr, functionName, schema.GetString());
        }
        SPDLOG_TRACE("Tool:{} param:{} type:{}", functionName, paramName, typeStr);
        result.emplace(paramName, type);
    }
    return result;
}

static std::string setCorrectValueType(std::string& inputValue, const std::string& currentParameterName, const ParametersTypeMap_t& parametersType) {
    auto paramIt = parametersType.find(currentParameterName);
    if (paramIt == parametersType.end()) {
        SPDLOG_DEBUG("Parameter: {} schema not found , leaving as is", currentParameterName);
        return inputValue;
    }
    if (paramIt->second == ParameterType::STRING) {
        inputValue = "\"" + inputValue + "\"";
        return inputValue;
    }
    if (paramIt->second == ParameterType::BOOLEAN) {
        // in case of bool we need to convert to lower case
        std::transform(inputValue.begin(), inputValue.end(), inputValue.begin(), ::tolower);
        return inputValue;
    }
    return inputValue;
}

#define DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(TAG)                         \
    auto pos = this->streamContent.find(TAG, this->getLastProcessedPosition()); \
    if (pos == std::string::npos) {                                             \
        SPDLOG_TRACE("Did not find: {}", TAG);                                  \
        break;                                                                  \
    }

bool Qwen3CoderToolParserImpl::parseUntilStateChange(ToolCalls& toolCalls) {
    SPDLOG_TRACE("State: {}", this->currentState);
    auto previousState = this->currentState;
    switch (this->currentState) {
    case State::Content: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::TOOL_START_TAG);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::TOOL_START_TAG.length();
        this->currentState = State::InsideToolCall;
        this->toolCallPositions.begin.push(pos);
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
            break;
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
        std::string parameterValue(streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition));
        if (this->removeNewlineAroundParameters)
            trimNewline(parameterValue);
        // now we have parameter value in string format. We need to use toolsSchemas to determine if it is string, number, bool, array or object
        auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
        if (paramIt == this->toolsParametersTypeMap.end()) {
            SPDLOG_DEBUG("Tool schema not found for tool: {}, leaving parameter: {} as string", this->currentFunction.name, this->currentParameterName);
        } else {
            parameterValue = setCorrectValueType(parameterValue, this->currentParameterName, paramIt->second);
        }
        auto res = this->currentFunction.parameters.try_emplace(this->currentParameterName, parameterValue);
        if (!res.second)
            SPDLOG_DEBUG("Parameter: {} already exists", this->currentParameterName);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::PARAMETER_END_TAG.length();
        this->currentState = State::InsideFunction;
        break;
    }
    case State::AfterFunction: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(Qwen3CoderToolParser::TOOL_END_TAG);
        this->lastProcessedPosition = pos + Qwen3CoderToolParser::TOOL_END_TAG.length();
        this->currentState = State::Content;
        ToolCall toolCall{generateRandomId(), this->currentFunction.name, this->currentFunction.parametersToJson()};
        SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
        toolCalls.emplace_back(std::move(toolCall));
        this->currentFunction.clear();
        this->toolCallPositions.end.push(this->lastProcessedPosition);
        break;
    }
    }
    return previousState != this->currentState;
}
std::optional<ToolCalls> Qwen3CoderToolParserImpl::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        return std::nullopt;
    }
    ToolCalls toolCalls;
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

static ToolsParameterTypeMap_t createToolsParametersTypesMap(const ToolsSchemas_t& toolsSchemas) {
    SPDLOG_TRACE("Creating tools parameters types map");
    ToolsParameterTypeMap_t toolsParametersTypes;
    for (const auto& [toolName, schemaPair] : toolsSchemas) {
        SPDLOG_TRACE("Creating tools parameters types for tool: {}, schema: {}", toolName, schemaPair.second);
        toolsParametersTypes.emplace(toolName, parseToolSchema(toolName, *schemaPair.first));
    }
    return toolsParametersTypes;
}

void Qwen3CoderToolParser::lazyFillInitToolParametersTypesMap() {
    if (this->filledParametersTypesMap) {
        return;
    }
    SPDLOG_DEBUG("Filling tools parameters types map");
    this->toolsParametersTypes = createToolsParametersTypesMap(this->toolSchemas);
    this->filledParametersTypesMap = true;
}

Qwen3CoderToolParser::Qwen3CoderToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas) :
    BaseOutputParser(tokenizer),
    toolSchemas(toolSchemas),
    streamParser(this->toolsParametersTypes) {
    SPDLOG_DEBUG("Qwen3CoderToolParser created with {} tools", toolsParametersTypes.size());
}

void Qwen3CoderToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // there may be multiple parameters per function,
    // there may be multiple lines per parameter value
    // there may be no parameters for a function
    // there may be multiple tool_call sections in the content
    // there is only one function per tool call
    // <tool_call>
    // <function=FUNCTION_NAME>
    // <parameter=PARAM_NAME>
    // PARAM_VALUE
    // </parameter>
    // </function>
    // </tool_call>
    this->lazyFillInitToolParametersTypesMap();
    auto toolCallsOpt = this->streamParser.parseChunk(parsedOutput.content);
    if (toolCallsOpt.has_value()) {
        // TODO do we want to support not ending in content state?
        parsedOutput.toolCalls = std::move(toolCallsOpt.value());
        SPDLOG_DEBUG("Parsing ended successfully, removing tool calls from content");
        auto status = this->streamParser.removeToolCallsFromContentIfNeeded(parsedOutput.content);
        if (!status.ok()) {
            SPDLOG_DEBUG("Failed to remove tool calls from content: {}", status.string());
        }
        return;
    }
    SPDLOG_DEBUG("Parsing ended, no tool calls found");
    return;
}
std::optional<std::string> Qwen3CoderToolParserImpl::getCurrentFunctionName() const {
    if (this->currentFunction.name.empty()) {
        return std::nullopt;
    }
    return this->currentFunction.name;
}
std::optional<rapidjson::Document> Qwen3CoderToolParser::sendFullDelta(std::optional<ToolCalls>& toolCallsOpt) {
    auto& toolCalls = toolCallsOpt.value();
    if (toolCalls.size() != 1) {
        SPDLOG_ERROR("For streaming we expected one tool call, got: {}", toolCalls.size());
    }
    if (toolCalls.size() < 1) {
        return std::nullopt;
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

std::optional<rapidjson::Document> Qwen3CoderToolParser::parseChunk(const std::string& newChunk, ov::genai::GenerationFinishReason finishReason) {
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
        return this->sendFullDelta(toolCallsOpt);
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
