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
#include "qwen3coder_tool_parser.hpp"

namespace ovms {
const std::string Qwen3CoderToolParser::toolsStartTag = "<tool_call>";
const std::string Qwen3CoderToolParser::toolPrefixTag = "<function=";
const std::string Qwen3CoderToolParser::tagEnd = ">";
const std::string Qwen3CoderToolParser::parameterPrefixTag = "<parameter=";
const std::string Qwen3CoderToolParser::parameterEndTag = "</parameter>";
const std::string Qwen3CoderToolParser::toolEndTag = "</function>";
const std::string Qwen3CoderToolParser::toolsEndTag = "</tool_call>";

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
/*static bool isNumber(const std::string& s) {
    if (s.empty())
        return false;
    char* endptr = nullptr;
    strtod(s.c_str(), &endptr);
    return (*endptr == '\0');
}
static bool isBoolean(const std::string& s) {
    return (s == "true" ||
            s == "false" ||
            s == "True" ||
            s == "False");
}
static bool isJsonLike(const std::string& s) {
    if (s.empty())
        return false;
    return (s.front() == '{' && s.back() == '}');
}
static bool isArrayLike(const std::string& s) {
    if (s.empty())
        return false;
    return (s.front() == '[' && s.back() == ']');
}*/
/*static std::string toJsonLike(const std::string_view& s) {
    // use recursion to handle nested arrays and objects
    // first check if it is array
    // then check if it is object
    // then check if it is number or boolean
    // otherwise treat as string
    // note that array could be nested, and its elements can be strings
    // note that object could contain array
    // note that array could contain object
    // note that object could contain object
    // note that object could contain string
    // note that object could contain number
    // note that object could contain boolean
    // note that array could contain number
    // note that array could contain boolean
    // note that array could contain array
    if (s.empty()) {
        return "\"\"";  // empty string
    }
    return std::string(s);
}*/

// write toJson function
// it should know based on content of value from pair if it is string or number or bool
// if number or bool, or array or json object. Note that array could be nested, and its elements can be strings
static std::string toJson(const std::vector<std::pair<std::string, std::string>>& items) {
    std::ostringstream oss;
    oss << "{";
    /*    for (size_t i = 0; i < items.size(); ++i) {
        const auto& [key, value] = items[i];
        oss << "\"" << key << "\": ";
        oss << toJsonLike(value);
        if (i + 1 < items.size()) {
            oss << ", ";
        }
    }
}*/

    for (size_t i = 0; i < items.size(); ++i) {
        const auto& [key, value] = items[i];
        oss << "\"" << key << "\": ";

        oss << value;  // no quotes
                       /*        if (isNumber(value) ||
            isBoolean(value) ||
            isJsonLike(value) ||
            isArrayLike(value)) {
            oss << value;  // no quotes
        } else {
            oss << "\"" << value << "\"";
        }*/

        if (i + 1 < items.size()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

// write toJson function
// it should know based on content of value from pair if it is string or number or bool
/*static std::string toJson2(const std::vector<std::pair<std::string, std::string>>& parameters) {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

    for (const auto& param : parameters) {
        SPDLOG_DEBUG("Adding parameter to json: {}: {}", param.first, param.second);
        
        rapidjson::Value key(param.first.c_str(), allocator);
        rapidjson::Value value(param.second.c_str(), allocator);
        doc.AddMember(key, value, allocator);
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return buffer.GetString();
}
*/
#define CHECK_IF_FOUND(POS, TAG)                  \
    do {                                          \
        if (POS == std::string::npos) {           \
            SPDLOG_TRACE("Did not find:{}", TAG); \
            return;                               \
        }                                         \
    } while (0)
#define CHECK_IF_FOUND2(POS, TAG, STATE)                                                                                                      \
    do {                                                                                                                                      \
        if (POS == std::string::npos) {                                                                                                       \
            SPDLOG_TRACE("Did not find:{} in str:{}", TAG, content.substr(this->currentPosition, 20));                                        \
            this->currentPosition = POS;                                                                                                      \
            this->currentState = STATE;                                                                                                       \
            if (STATE == State::End) {                                                                                                        \
                this->removeToolCallsFromContent();                                                                                           \
                /*ToolCall toolCall{generateRandomId(), this->currentFunction.name, toJson(this->currentFunction.parameters)};*/              \
                /*            SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.parameters);*/ \
                /*            toolCalls.emplace_back(std::move(toolCall));*/                                                                  \
            }                                                                                                                                 \
            return false;                                                                                                                     \
        }                                                                                                                                     \
    } while (0)
void Parser::removeToolCallsFromContent() {
    if (toolsBeginStack.size() != toolsEndStack.size()) {
        SPDLOG_WARN("Mismatched tool tags, begin: {}, end: {}", toolsBeginStack.size(), toolsEndStack.size());
        throw std::runtime_error("Mismatched tool tags");  // FIXME replace with status
    }
    SPDLOG_TRACE("Removing {} tool calls from content", toolsBeginStack.size());
    while (!toolsBeginStack.empty() && !toolsEndStack.empty()) {
        auto posBegin = toolsBeginStack.top();
        auto posEnd = toolsEndStack.top();
        SPDLOG_TRACE("Removing tool call from content begin:{}, end:{}, removing:{}XYZ", posBegin, posEnd, content.substr(posBegin, posEnd - posBegin));

        content.erase(posBegin, posEnd - posBegin);
        toolsBeginStack.pop();
        toolsEndStack.pop();
    }
}

// {"type":"object","properties":{"location":{"type":"string"},"provide_temperature":{"type":"boolean"}},"required":["location"]}
// {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"The location for which to get the weather, in the format of 'City, State', such as 'San Francisco, CA' if State for the city exists. 'City, Country' if State for the city doesn't exist."},"unit":{"type":"string","description":"The unit of temperature for the weather report.","enum":["celsius","fahrenheit"],"default":"fahrenheit"}}}

static const ParametersTypeMap_t parseToolSchema(const std::string& functionName, const std::string& schema) {
    SPDLOG_ERROR("Parse tool schema for tool: {}, schema: {}", functionName, schema);
    ParametersTypeMap_t result;
    rapidjson::Document doc;
    if (doc.Parse(schema.c_str()).HasParseError()) {
        SPDLOG_WARN("Tool schema is not valid JSON for tool: {}, schema: {}", functionName, schema);
        return result;
    }
    if (!doc.IsObject()) {
        SPDLOG_WARN("Tool schema is not a JSON object for tool: {}, schema: {}", functionName, schema);
        return result;
    }
    if (!doc.HasMember("properties") || !doc["properties"].IsObject()) {
        SPDLOG_WARN("Tool schema does not have properties object for tool: {}, schema: {}", functionName, schema);
        return result;
    }
    const rapidjson::Value& properties = doc["properties"];
    for (auto it = properties.MemberBegin(); it != properties.MemberEnd(); ++it) {
        if (!it->value.IsObject()) {
            SPDLOG_WARN("Tool schema property: {} is not an object for tool: {}, schema: {}", it->name.GetString(), functionName, schema);
            continue;
        }
        if (!it->value.HasMember("type") || !it->value["type"].IsString()) {
            SPDLOG_WARN("Tool schema property: {} does not have type string for tool: {}, schema: {}", it->name.GetString(), functionName, schema);
            continue;
        }
        std::string paramName = it->name.GetString();
        std::string typeStr = it->value["type"].GetString();
        ParameterType_t type = ParameterType_t::UNKNOWN;
        if (typeStr == "string") {
            type = ParameterType_t::STRING;
        } else if (typeStr == "number" || typeStr == "integer") {
            type = ParameterType_t::NUMBER;
        } else if (typeStr == "boolean") {
            type = ParameterType_t::BOOLEAN;
        } else if (typeStr == "array") {
            type = ParameterType_t::ARRAY;
        } else if (typeStr == "object") {
            type = ParameterType_t::OBJECT;
        } else {
            SPDLOG_WARN("Tool schema property: {} has unknown type: {} for tool: {}, schema: {}", paramName, typeStr, functionName, schema);
        }
        SPDLOG_TRACE("Tool:{} param:{} type:{}", functionName, paramName, typeStr);
        result.emplace(paramName, type);
    }
    return result;
}

static std::string setCorrectValueType(std::string& inputValue, const std::string& currentParameterName, const ParametersTypeMap_t& parametersType) {
    auto paramIt = parametersType.find(currentParameterName);
    if (paramIt == parametersType.end()) {
        SPDLOG_WARN("Parameter: {} schema not found , leaving as string", currentParameterName);
        return inputValue;
    }
    if (paramIt->second == ParameterType_t::STRING) {
        inputValue = "\"" + inputValue + "\"";
        return inputValue;  // no change needed
    }
    if (paramIt->second == ParameterType_t::BOOLEAN) {
        // in case of bool we need to convert to lower case
        std::transform(inputValue.begin(), inputValue.end(), inputValue.begin(), ::tolower);
        return inputValue;
    }
    // FIXME at error handling
    return inputValue;
}

#define RETURN_NULL_IF_NOT_FOUND(POS) \
    if (POS == std::string::npos) {   \
        return std::nullopt;          \
    }

#define LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(STATE, TAG)                                                         \
    SPDLOG_TRACE("State: {}, current content:{} current pos:{}", #STATE, streamContent, this->lastStreamProcessedPosition); \
    auto pos = streamContent.find(TAG, this->lastStreamProcessedPosition);                                                  \
    RETURN_NULL_IF_NOT_FOUND(pos)

std::optional<ToolCalls> Parser::streamStep(const std::string& chunk) {
    if (chunk.empty()) {
        return std::nullopt;
    }
    this->streamContent += chunk;
    switch (this->currentState) {
    case State::Content: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(Content, Qwen3CoderToolParser::toolsStartTag);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::toolsStartTag.length();
        this->currentState = State::InsideToolCall;
        return std::nullopt;
        break;
    }
    case State::InsideToolCall: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(InsideToolCall, Qwen3CoderToolParser::toolPrefixTag);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::toolPrefixTag.length();
        this->currentState = State::InsideFunctionName;
        return std::nullopt;
        break;
    }
    case State::InsideFunctionName: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(InsideFunctionName, Qwen3CoderToolParser::tagEnd);
        this->currentFunction.name = streamContent.substr(this->lastStreamProcessedPosition, pos - this->lastStreamProcessedPosition);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        // here in theory we could wrapFirstDelta to send function name TODO
        this->currentState = State::InsideFunction;
        return std::nullopt;
        break;
    }
    case State::InsideFunction: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(InsideFunction, Qwen3CoderToolParser::parameterPrefixTag);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::parameterPrefixTag.length();
        this->currentState = State::InsideParameterName;
        return std::nullopt;
        break;
    }
    case State::InsideParameterName: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(InsideParameterName, Qwen3CoderToolParser::tagEnd);
        this->currentParameterName = streamContent.substr(this->lastStreamProcessedPosition, pos - this->lastStreamProcessedPosition);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        this->currentState = State::InsideParameter;
        return std::nullopt;
        break;
    }
    case State::InsideParameter: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(InsideParameter, Qwen3CoderToolParser::parameterEndTag);
        std::string parameterValue(streamContent.substr(this->lastStreamProcessedPosition, pos - this->lastStreamProcessedPosition));
        if (this->removeNewlineAroundParameters)
            trimNewline(parameterValue);
        // now we have parameter value in string format. We need to use toolsSchemas to determine if it is string, number, bool, array or object
        auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
        if (paramIt == this->toolsParametersTypeMap.end()) {
            SPDLOG_DEBUG("Tool schema not found for tool: {}, leaving parameter: {} as string", this->currentFunction.name, this->currentParameterName);
        } else {
            parameterValue = setCorrectValueType(parameterValue, this->currentParameterName, paramIt->second);
        }
        this->currentFunction.parameters.emplace_back(this->currentParameterName, parameterValue);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::parameterEndTag.length();
        this->currentState = State::AfterParameter;
        return std::nullopt;
        break;
    }
    case State::AfterParameter: {
        /*      
        SPDLOG_TRACE("State: AfterParameter current content:{} current pos:{}", streamContent, this->lastStreamProcessedPosition);
        auto posToolEnd = streamContent.find(Qwen3CoderToolParser::toolEndTag, this->lastStreamProcessedPosition);
        auto posNewParameter = streamContent.find(Qwen3CoderToolParser::parameterPrefixTag, this->lastStreamProcessedPosition);
        SPDLOG_ERROR("ER: rest:{} {} {} {}", streamContent.substr(lastStreamProcessedPosition), this->lastStreamProcessedPosition, posToolEnd, posNewParameter);
        if (posNewParameter < posToolEnd && posNewParameter != std::string::npos) {
            SPDLOG_ERROR("ER");
            this->lastStreamProcessedPosition = posNewParameter + Qwen3CoderToolParser::parameterPrefixTag.length();
            this->currentState = State::InsideParameterName;
            return std::nullopt;
            break;
        }
        if (posToolEnd != std::string::npos) {
            SPDLOG_ERROR("ER");
            auto pos = posToolEnd;
            this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::toolEndTag.length();
            pos = streamContent.find(Qwen3CoderToolParser::toolsEndTag, this->lastStreamProcessedPosition);
            RETURN_NULL_IF_NOT_FOUND(pos); // we do not switch state yet
            this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::toolsEndTag.length();
            this->currentState = State::Content;
            ToolCall toolCall{generateRandomId(), this->currentFunction.name, toJson(this->currentFunction.parameters)};
            SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
            toolCalls.emplace_back(std::move(toolCall));
            currentFunction.clear();
            currentState = State::Content;
            return toolCalls;
            break;
        }
        // we found neither
            SPDLOG_ERROR("ER");
        return std::nullopt; // we do not switch state yet
        break;
        */
        SPDLOG_TRACE("State: AfterParameter current content:{} current pos:{}", streamContent, this->lastStreamProcessedPosition);
        auto posToolEnd = streamContent.find(Qwen3CoderToolParser::toolEndTag, this->lastStreamProcessedPosition);
        auto posNewParameter = streamContent.find(Qwen3CoderToolParser::parameterPrefixTag, this->lastStreamProcessedPosition);
        SPDLOG_ERROR("ER: rest:{} {} {} {}", streamContent.substr(lastStreamProcessedPosition), this->lastStreamProcessedPosition, posToolEnd, posNewParameter);
        if (posNewParameter == std::string::npos && posToolEnd == std::string::npos) {
            SPDLOG_TRACE("ER: found neither no changing state");
            return std::nullopt;  // we do not switch state yet
        }
        if (posToolEnd < posNewParameter) {
            SPDLOG_TRACE("ER: found tool end");
            this->lastStreamProcessedPosition = posToolEnd + Qwen3CoderToolParser::toolEndTag.length();
            this->currentState = State::AfterFunction;
            return std::nullopt;
            break;
        }
        if (posNewParameter < posToolEnd) {
            SPDLOG_TRACE("ER: found new param");
            this->lastStreamProcessedPosition = posNewParameter + Qwen3CoderToolParser::parameterPrefixTag.length();
            this->currentState = State::InsideParameterName;
            return std::nullopt;
            break;
        }
        throw 42;
    }
    case State::AfterFunction: {
        LOG_AND_DEFINE_POS_AND_RETURN_NULL_IF_NOT_FOUND(AfterFunction, Qwen3CoderToolParser::toolsEndTag);
        this->lastStreamProcessedPosition = pos + Qwen3CoderToolParser::toolsStartTag.length();
        this->currentState = State::Content;
        ToolCall toolCall{generateRandomId(), this->currentFunction.name, toJson(this->currentFunction.parameters)};
        SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
        ToolCalls toolCalls;
        toolCalls.emplace_back(std::move(toolCall));
        this->currentFunction.clear();
        return toolCalls;
        break;
    }
    case State::ErrorEnd: {
        SPDLOG_TRACE("State: ErrorEnd");
        return std::nullopt;
    }
    case State::End: {
        SPDLOG_TRACE("State: End");
        return std::nullopt;
    }
    }
    return std::nullopt;
}

bool Parser::step(ToolCalls& toolCalls) {
    switch (currentState) {
    case State::Content: {
        SPDLOG_TRACE("State: Content");
        auto pos = content.find(Qwen3CoderToolParser::toolsStartTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::toolsStartTag, State::End);
        currentPosition = pos + Qwen3CoderToolParser::toolsStartTag.length();
        toolsBeginStack.push(pos);
        currentState = State::InsideToolCall;
        break;
    }
    case State::InsideToolCall: {
        SPDLOG_TRACE("State: InsideToolCall");
        auto pos = content.find(Qwen3CoderToolParser::toolPrefixTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::toolPrefixTag, State::ErrorEnd);
        currentPosition = pos + Qwen3CoderToolParser::toolPrefixTag.length();
        currentState = State::InsideFunctionName;
        break;
    }
    case State::InsideFunctionName: {
        SPDLOG_TRACE("State: InsideFunctionName");
        auto pos = content.find(Qwen3CoderToolParser::tagEnd, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::tagEnd, State::ErrorEnd);
        currentFunction.name = content.substr(currentPosition, pos - currentPosition);
        currentPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        currentState = State::InsideFunction;
        break;
    }
    case State::InsideFunction: {
        SPDLOG_TRACE("State: InsideFunction");
        auto pos = content.find(Qwen3CoderToolParser::parameterPrefixTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::parameterPrefixTag, State::ErrorEnd);
        currentPosition = pos + Qwen3CoderToolParser::parameterPrefixTag.length();
        currentState = State::InsideParameterName;
        break;
    }
    case State::InsideParameterName: {
        SPDLOG_TRACE("State: InsideParameterName");
        auto pos = content.find(Qwen3CoderToolParser::tagEnd, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::tagEnd, State::ErrorEnd);
        this->currentParameterName = content.substr(currentPosition, pos - currentPosition);
        currentPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        currentState = State::InsideParameter;
        break;
    }
    case State::InsideParameter: {
        SPDLOG_TRACE("State: InsideParameter");
        auto pos = content.find(Qwen3CoderToolParser::parameterEndTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::parameterEndTag, State::ErrorEnd);
        std::string parameterValue(content.substr(currentPosition, pos - currentPosition));
        if (this->removeNewlineAroundParameters)
            trimNewline(parameterValue);
        // now we have parameter value in string format. We need to use toolsSchemas to determine if it is string, number, bool, array or object
        auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
        if (paramIt == this->toolsParametersTypeMap.end()) {
            SPDLOG_DEBUG("Tool schema not found for tool: {}, leaving parameter: {} as string", this->currentFunction.name, this->currentParameterName);
        } else {
            parameterValue = setCorrectValueType(parameterValue, this->currentParameterName, paramIt->second);
        }
        currentFunction.parameters.emplace_back(this->currentParameterName, parameterValue);
        currentPosition = pos + Qwen3CoderToolParser::parameterEndTag.length();
        currentState = State::AfterParameter;
        break;
    }
    case State::AfterParameter: {
        SPDLOG_TRACE("State: AfterParameter");
        // now we have 2 options, either we have next parameter or function & tool end
        auto posToolEnd = content.find(Qwen3CoderToolParser::toolEndTag, currentPosition);
        auto posNewParameter = content.find(Qwen3CoderToolParser::parameterPrefixTag, currentPosition);
        if (posNewParameter < posToolEnd && posNewParameter != std::string::npos) {
            auto pos = posNewParameter;
            currentPosition = pos + Qwen3CoderToolParser::parameterPrefixTag.length();
            currentState = State::InsideParameterName;
            break;
        }
        if (posToolEnd < posNewParameter && posToolEnd != std::string::npos) {
            auto pos = posToolEnd;
            currentPosition = pos + Qwen3CoderToolParser::toolEndTag.length();
            pos = content.find(Qwen3CoderToolParser::toolsEndTag, currentPosition);
            CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::toolsEndTag, State::ErrorEnd);
            currentPosition = pos + Qwen3CoderToolParser::toolsEndTag.length();
            toolsEndStack.push(currentPosition);
            ToolCall toolCall{generateRandomId(), this->currentFunction.name, toJson(this->currentFunction.parameters)};
            SPDLOG_TRACE("Adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
            toolCalls.emplace_back(std::move(toolCall));
            currentFunction.clear();
            currentState = State::Content;
            break;
        }
        // we did not find either, so this is an error
        CHECK_IF_FOUND2(posNewParameter, Qwen3CoderToolParser::toolEndTag + " nor " + Qwen3CoderToolParser::parameterPrefixTag, State::ErrorEnd);
        break;
    }
    case State::AfterFunction:
    case State::ErrorEnd:
    case State::End:
        SPDLOG_TRACE("State: {}", this->currentState);
        return false;
    }
    return true;
}

static ToolsParameterTypeMap_t createToolsParametersTypesMap(const ToolsSchemas_t& toolsSchemas) {
    SPDLOG_ERROR("Creating tools parameters types map");
    ToolsParameterTypeMap_t toolsParametersTypes;  // FIXME do it once per request
    for (const auto& [toolName, schema] : toolsSchemas) {
        SPDLOG_ERROR("Creating tools parameters types for tool: {}, schema: {}", toolName, schema);
        toolsParametersTypes.emplace(toolName, parseToolSchema(toolName, schema));
    }
    return toolsParametersTypes;
}

void Qwen3CoderToolParser::lazyFillInitToolParamatersTypsMap() {
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
    streamParser(NULL_STRING_CONTENT, this->toolsParametersTypes) {
    SPDLOG_DEBUG("Qwen3CoderToolParser created with {} tools", toolsParametersTypes.size());
}

void Qwen3CoderToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // there may be multiple parameters per function, there may be multiple linses per parameter value
    // there is only one function per tool call
    // <tool_call>
    // <function=FUNCTION_NAME>
    // <parameter=PARAM_NAME>
    // PARAM_VALUE
    // </parameter>
    // </function>
    // </tool_call>a
    this->lazyFillInitToolParamatersTypsMap();
    Parser parser(parsedOutput.content, this->toolsParametersTypes);
    while (parser.step(parsedOutput.toolCalls)) {
    }
    if (parser.currentState != Parser::State::End) {
        SPDLOG_DEBUG("Parsing ended with error, leaving content as is");
        return;
    }
    return;
}
/*
{"type":"response.output_item.added","response_id":"resp_1234xyz","output_index":0,"item":{"type":"function_call","id":"fc_1234xyz","call_id":"call_1234xyz","name":"get_weather","arguments":""}}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"{\""}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"location"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"\":\""}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"Paris"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":","}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":" France"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"\"}"}
{"type":"response.function_call_arguments.done","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"arguments":"{\"location\":\"Paris, France\"}"}
{"type":"response.output_item.done","response_id":"resp_1234xyz","output_index":0,"item":{"type":"function_call","id":"fc_1234xyz","call_id":"call_1234xyz","name":"get_weather","arguments":"{\"location\":\"Paris, France\"}"}}
*/
// example1 {"location":"San Francisco"}
// example1 {"city":"San Francisco","state":"CA", "length":5, "is_day":true, "temperatures":[5,6,7], "details":{"humidity":80,"condition":"sunny"}}]}
// index is toolCallId
// // write function that will wrap toolCall arguments in the required JSON structure as above

// for Qwen3Coder we will send first response with function call name and id
// then we will send only one delta with all arguments
// we already have toJson functiont that turns all parameters into JSON string

static std::string documentToString(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

std::optional<rapidjson::Document> Qwen3CoderToolParser::parseChunk(const std::string& newChunk, ov::genai::GenerationFinishReason finishReason) {
    // parse Chunk needs to use streamParser to process the chunk
    // streamParser will return optional toolCalls when a tool call is completed
    // if toolCalls is returned, we need to wrap it in the required JSON structure and return it
    // if toolCalls is not returned, but we are insideFunction state, we need to return the first delta with function name
    // otherwise nullopt
    this->lazyFillInitToolParamatersTypsMap();
    if (newChunk.empty()) {
        return std::nullopt;
    }
    std::string chunk = newChunk;
    SPDLOG_DEBUG("Chunk: '{}', finishReason: {}", chunk, static_cast<int>(finishReason));
    auto parserState = streamParser.currentState;
    // need currentToolcalss
    auto toolCallsOpt = streamParser.streamStep(chunk);
    if (streamParser.currentState == Parser::State::InsideFunction) {
        if (returnedFirstDeltas.size() != this->returnedCompleteDeltas.size()) {
            SPDLOG_TRACE("Skipping first delta, already sent for current function, fi:{} co:{}", returnedFirstDeltas.size(), returnedCompleteDeltas.size());
            // we can skip sending first delta since we sent it for current function
            return std::nullopt;
        }
        int toolCallId = ++this->streamParser.toolCallIndex;
        const std::string& toolCallName = this->streamParser.currentFunction.name;
        SPDLOG_ERROR("Caught function name: {}, id: {}", toolCallName, toolCallId);
        rapidjson::Document doc = wrapFirstDelta(toolCallName, toolCallId);
        this->currentJson.CopyFrom(doc, this->currentJson.GetAllocator());
        returnedFirstDeltas.insert(toolCallId);
        // here log the doc
        SPDLOG_DEBUG("First delta doc: {}", documentToString(doc));
        return doc;
    }
    if (toolCallsOpt.has_value()) {
        // TODO we assume we could only get one by one
        SPDLOG_ERROR("Has value but not insideFunction");
        auto& toolCalls = toolCallsOpt.value();
        if (toolCalls.size() != 1) {
            SPDLOG_WARN("Expected one tool call, got: {}", toolCalls.size());
            throw 42;
        }
        if (toolCalls.size() < 1) {
            return std::nullopt;
        }
        auto& toolCall = toolCalls[0];
        rapidjson::Document argsDelta;
        argsDelta.Parse(toolCall.arguments.c_str());
        this->returnedCompleteDeltas.insert(this->streamParser.toolCallIndex);
        rapidjson::Document argumentsWrapper;
        argumentsWrapper.SetObject();
        rapidjson::Document::AllocatorType& allocator = argumentsWrapper.GetAllocator();
        // now we need to add string toolCall.arguments to argumentsWrapper under "arguments" key
        rapidjson::Value toolCallsString(rapidjson::kStringType);
        toolCallsString.SetString(toolCall.arguments.c_str(), allocator);
        argumentsWrapper.AddMember("arguments", toolCallsString, allocator);

        auto currentDelta = wrapDelta(argumentsWrapper, this->streamParser.toolCallIndex);
        SPDLOG_DEBUG("First delta doc: {}", documentToString(currentDelta));
        return currentDelta;
    }
    if (parserState != streamParser.currentState) {
        SPDLOG_DEBUG("Parser state changed from {} to {}", parserState, this->streamParser.currentState);
    }
    parserState = streamParser.currentState;
    return std::nullopt;  // FIXME
}
Parser::Parser(std::string& content, const ToolsParameterTypeMap_t& toolsParametersTypeMap) :
    content(content),
    toolsParametersTypeMap(toolsParametersTypeMap) {}
}  // namespace ovms
