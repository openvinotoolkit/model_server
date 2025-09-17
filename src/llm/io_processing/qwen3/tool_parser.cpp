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
#include <string>
#include <stack>
#include <vector>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "../utils.hpp"

namespace ovms {
const std::string Qwen3CoderToolParser::toolsStartTag = "<tool_call>";
const std::string Qwen3CoderToolParser::toolsEndTag = "</tool_call>";
const std::string Qwen3CoderToolParser::toolPrefixTag = "<function=";
const std::string Qwen3CoderToolParser::toolEndTag = "</function>";
const std::string Qwen3CoderToolParser::parameterPrefixTag = "<parameter=";
const std::string Qwen3CoderToolParser::parameterEndTag = "</parameter>";
const std::string Qwen3CoderToolParser::tagEnd = ">";

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
static bool isNumber(const std::string& s) {
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
}
/*static std::string toJsonLike(const std::string_view& s) {
    // use recursion to handle nested arrays and objects
    // first check if it is array
    // then check if it is object
    // then check if it is number or boolean
    // otherwise treat as string
    // note that array could be nested, and its elemetns can be strings
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
// it shoudl know based on content of value from pair if it is string or number or bool
// if number or bool, or array or json object. Note that array could be nested, and its elemetns can be strings
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

        if (isNumber(value) ||
            isBoolean(value) ||
            isJsonLike(value) ||
            isArrayLike(value)) {
            oss << value;  // no quotes
        } else {
            oss << "\"" << value << "\"";
        }

        if (i + 1 < items.size()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

// write toJson function
// it shoudl know based on content of value from pair if it is string or number or bool
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
bool Parser::step(ToolCalls& toolCalls) {
    switch (currentState) {
    case Content: {
        SPDLOG_TRACE("State: Content");
        auto pos = content.find(Qwen3CoderToolParser::toolsStartTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::toolsStartTag, State::End);
        currentPosition = pos + Qwen3CoderToolParser::toolsStartTag.length();
        toolsBeginStack.push(pos);
        currentState = State::InsideToolCall;
        break;
    }
    case InsideToolCall: {
        SPDLOG_TRACE("State: InsideToolCall");
        auto pos = content.find(Qwen3CoderToolParser::toolPrefixTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::toolPrefixTag, State::ErrorEnd);
        currentPosition = pos + Qwen3CoderToolParser::toolPrefixTag.length();
        currentState = State::InsideFunctionName;
        break;
    }
    case InsideFunctionName: {
        SPDLOG_TRACE("State: InsideFunctionName");
        auto pos = content.find(Qwen3CoderToolParser::tagEnd, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::tagEnd, State::ErrorEnd);
        currentFunction.name = content.substr(currentPosition, pos - currentPosition);
        currentPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        currentState = State::InsideFunction;
        break;
    }
    case InsideFunction: {
        SPDLOG_TRACE("State: InsideFunction");
        auto pos = content.find(Qwen3CoderToolParser::parameterPrefixTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::parameterPrefixTag, State::ErrorEnd);
        currentPosition = pos + Qwen3CoderToolParser::parameterPrefixTag.length();
        currentState = State::InsideParameterName;
        break;
    }
    case InsideParameterName: {
        SPDLOG_TRACE("State: InsideParameterName");
        auto pos = content.find(Qwen3CoderToolParser::tagEnd, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::tagEnd, State::ErrorEnd);
        this->currentParameterName = content.substr(currentPosition, pos - currentPosition);
        ;
        currentPosition = pos + Qwen3CoderToolParser::tagEnd.length();
        currentState = State::InsideParameter;
        break;
    }
    case InsideParameter: {
        SPDLOG_TRACE("State: InsideParameter");
        auto pos = content.find(Qwen3CoderToolParser::parameterEndTag, currentPosition);
        CHECK_IF_FOUND2(pos, Qwen3CoderToolParser::parameterEndTag, State::ErrorEnd);
        std::string parameterValue(content.substr(currentPosition, pos - currentPosition));
        if (this->removeNewlineAroundParameters)
            trimNewline(parameterValue);
        currentFunction.parameters.emplace_back(this->currentParameterName, parameterValue);
        currentPosition = pos + Qwen3CoderToolParser::parameterEndTag.length();
        currentState = State::AfterParameter;
        break;
    }
    case AfterParameter: {
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
    case ErrorEnd:
        SPDLOG_TRACE("State: ErrorEnd");
        return false;
    case End:
        SPDLOG_TRACE("State: End");
        return false;
    }
    return true;
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
    //
    // FIXME check for npos at each step
    // For each if (itFunctionNameEnd == std::string::npos) {SPDLOG_ERROR("No tag end found"); return;}
    // we need to replace it with macro
    Parser parser(parsedOutput.content);
    while (parser.step(parsedOutput.toolCalls))
        ;
    if (parser.currentState != Parser::State::End) {
        SPDLOG_DEBUG("Parsing ended with error, leaving content as is");
        return;
    }
    return;
}

std::optional<rapidjson::Document> Qwen3CoderToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    return std::nullopt;  // FIXME
}

}  // namespace ovms
