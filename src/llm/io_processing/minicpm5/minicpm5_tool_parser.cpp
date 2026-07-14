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
#include <string>
#include <stack>
#include <vector>

#include "rapidjson/error/en.h"

#include "src/llm/io_processing/utils.hpp"
#include "src/logging.hpp"
#include "src/utils/rapidjson_utils.hpp"
#include "minicpm5_tool_parser.hpp"

namespace ovms {

// ---- Tag string constants ----
const std::string Minicpm5ToolParser::FUNCTION_START_TAG = "<function";
const std::string Minicpm5ToolParser::NAME_ATTR_PREFIX = "name=";  // will handle both " and '
const std::string Minicpm5ToolParser::XML_TAG_END = ">";
const std::string Minicpm5ToolParser::PARAM_START_TAG = "<param";
const std::string Minicpm5ToolParser::PARAM_END_TAG = "</param>";
const std::string Minicpm5ToolParser::FUNCTION_END_TAG = "</function>";
const std::string Minicpm5ToolParser::EOS_TOKEN_STR = "<|im_end|>";
const std::string Minicpm5ToolParser::SOS_TOKEN_STR = "<s>";

// Schema helpers, JSON helpers and string helpers are shared with qwen3coder; see utils.{hpp,cpp}.

// ---- Minicpm5ToolParserImpl ----

Minicpm5ToolParserImpl::Minicpm5ToolParserImpl(const ToolsParameterTypeMap_t& toolsParametersTypeMap) :
    toolsParametersTypeMap(toolsParametersTypeMap) {}

/*
 * Given the portion of streamContent that starts immediately after "name=" (i.e. at the
 * opening quote/apostrophe), extract the attribute value and return it.
 * tagEnd is the position of the '>' that closes the enclosing tag.
 * Returns the extracted value, or empty string on failure.
 */
std::string Minicpm5ToolParserImpl::extractNameAttribute(
    const std::string& content, size_t nameAttrValueStart, size_t tagEnd) {
    if (nameAttrValueStart >= tagEnd || nameAttrValueStart >= content.size()) {
        return {};
    }
    char quote = content[nameAttrValueStart];
    if (quote != '"' && quote != '\'') {
        // No quote: read until next whitespace or '>'
        size_t end = content.find_first_of(" \t\n\r>/", nameAttrValueStart);
        if (end == std::string::npos || end > tagEnd)
            end = tagEnd;
        return content.substr(nameAttrValueStart, end - nameAttrValueStart);
    }
    size_t closeQuote = content.find(quote, nameAttrValueStart + 1);
    if (closeQuote == std::string::npos || closeQuote > tagEnd) {
        return {};
    }
    return content.substr(nameAttrValueStart + 1, closeQuote - nameAttrValueStart - 1);
}

void Minicpm5ToolParserImpl::addParameterToCurrentFunctionDoc(std::string& parameterValueAsString) {
    if (this->removeNewlineAroundParameters)
        trimNewline(parameterValueAsString);

    auto paramIt = this->toolsParametersTypeMap.find(this->currentFunction.name);
    auto& currentFunctionArgsDoc = this->currentFunction.argumentsAsDocument;
    auto& allocator = currentFunctionArgsDoc.GetAllocator();
    auto& key = this->currentParameterName;
    rapidjson::Value keyVal(key.c_str(), allocator);
    rapidjson::Value valueCopy;

    rapidjson::Document temp;
    // Boolean normalisation (shared helper, same as qwen3coder)
    if (paramIt != this->toolsParametersTypeMap.end()) {
        auto paramJt = paramIt->second.find(currentParameterName);
        if (paramJt != paramIt->second.end() && paramJt->second == ParameterType::BOOLEAN) {
            normalizeBooleanString(parameterValueAsString);
        }
    }

    temp.Parse(parameterValueAsString.c_str());

    if (temp.HasParseError() && !parameterValueAsString.empty() &&
        (parameterValueAsString.front() == '{' || parameterValueAsString.front() == '[')) {
        std::string converted = replaceSingleWithDoubleQuotes(parameterValueAsString);
        rapidjson::Document retryDoc;
        retryDoc.Parse(converted.c_str());
        if (!retryDoc.HasParseError()) {
            SPDLOG_TRACE("Minicpm5: successfully parsed after single-to-double quote conversion: {}", converted);
            parameterValueAsString = std::move(converted);
            valueCopy.CopyFrom(retryDoc, allocator);
        }
    } else if (temp.HasParseError()) {
        rapidjson::ParseErrorCode errorCode = temp.GetParseError();
        size_t errorOffset = temp.GetErrorOffset();
        SPDLOG_TRACE("Minicpm5: RapidJSON cannot parse param: {} value: {}; error offset: {}; code: {}; falling back to string",
            this->currentParameterName, parameterValueAsString, errorOffset, rapidjson::GetParseError_En(errorCode));
        valueCopy.SetString(parameterValueAsString.c_str(), static_cast<rapidjson::SizeType>(parameterValueAsString.size()), allocator);
    } else {
        valueCopy.CopyFrom(temp, allocator);
        if (paramIt != this->toolsParametersTypeMap.end()) {
            auto paramJt = paramIt->second.find(currentParameterName);
            if (paramJt != paramIt->second.end() && paramJt->second == ParameterType::STRING) {
                enforceStringValue(valueCopy, allocator);
            }
        }
    }
    if (!currentFunctionArgsDoc.HasMember(keyVal)) {
        currentFunctionArgsDoc.AddMember(keyVal, valueCopy, allocator);
    } else {
        SPDLOG_DEBUG("Minicpm5: parameter {} already exists in document", key);
    }
}

Status Minicpm5ToolParserImpl::removeToolCallsFromContentIfNeeded(std::string& outContent) {
    if (toolCallPositions.begin.size() != toolCallPositions.end.size()) {
        SPDLOG_DEBUG("Minicpm5: mismatched tool tags, begin: {}, end: {}",
            toolCallPositions.begin.size(), toolCallPositions.end.size());
        return Status(StatusCode::INTERNAL_ERROR, "Mismatched tool tags");
    }
    while (!toolCallPositions.begin.empty() && !toolCallPositions.end.empty()) {
        auto posBegin = toolCallPositions.begin.top();
        auto posEnd = toolCallPositions.end.top();
        SPDLOG_TRACE("Minicpm5: removing tool call from outContent begin:{}, end:{}", posBegin, posEnd);
        outContent.erase(posBegin, posEnd - posBegin);
        toolCallPositions.begin.pop();
        toolCallPositions.end.pop();
    }

    const std::vector<std::string> tokensToErase = {
        Minicpm5ToolParser::SOS_TOKEN_STR,
        Minicpm5ToolParser::EOS_TOKEN_STR};

    for (const auto& token : tokensToErase) {
        size_t pos = 0;
        while ((pos = outContent.find(token, pos)) != std::string::npos) {
            outContent.erase(pos, token.length());
        }
    }

    return StatusCode::OK;
}

void Minicpm5ToolParserImpl::handleInsideContentState() {
    // Look for the next <function tag; everything else is plain content.
    auto posFunc = this->streamContent.find(Minicpm5ToolParser::FUNCTION_START_TAG, this->lastProcessedPosition);
    if (posFunc == std::string::npos) {
        SPDLOG_TRACE("Minicpm5: no <function> found");
        return;
    }
    this->toolCallPositions.begin.push(posFunc);
    // Skip past "<function" — we now need to read the name="..." attribute
    this->lastProcessedPosition = posFunc + Minicpm5ToolParser::FUNCTION_START_TAG.size();
    this->currentState = State::InsideFunctionName;
}

void Minicpm5ToolParserImpl::handleInsideFunctionNameState() {
    // We are positioned somewhere inside the <function ... > opening tag.
    // First find the closing '>' of this tag.
    auto tagEnd = this->streamContent.find('>', this->lastProcessedPosition);
    if (tagEnd == std::string::npos) {
        SPDLOG_TRACE("Minicpm5: waiting for '>' of <function> tag");
        return;
    }
    // Find name=" (or name=') inside the tag body
    auto nameAttr = this->streamContent.find(Minicpm5ToolParser::NAME_ATTR_PREFIX, this->lastProcessedPosition);
    if (nameAttr == std::string::npos || nameAttr >= tagEnd) {
        SPDLOG_DEBUG("Minicpm5: <function> tag has no name= attribute — skipping");
        this->lastProcessedPosition = tagEnd + 1;
        this->currentState = State::Content;
        this->toolCallPositions.begin.pop();  // undo the push
        return;
    }
    size_t valueStart = nameAttr + Minicpm5ToolParser::NAME_ATTR_PREFIX.size();
    this->currentFunction.name = extractNameAttribute(this->streamContent, valueStart, tagEnd);
    if (this->currentFunction.name.empty()) {
        SPDLOG_DEBUG("Minicpm5: could not extract function name — skipping tag");
        this->lastProcessedPosition = tagEnd + 1;
        this->currentState = State::Content;
        this->toolCallPositions.begin.pop();
        return;
    }
    SPDLOG_TRACE("Minicpm5: function name: {}", this->currentFunction.name);
    this->lastProcessedPosition = tagEnd + 1;  // past '>'
    this->currentState = State::InsideFunction;
}

void Minicpm5ToolParserImpl::handleInsideFunctionState(ToolCalls_t& toolCalls) {
    // Expect either <param or </function>
    auto funcEnd = this->streamContent.find(Minicpm5ToolParser::FUNCTION_END_TAG, this->lastProcessedPosition);
    auto paramStart = this->streamContent.find(Minicpm5ToolParser::PARAM_START_TAG, this->lastProcessedPosition);
    if (funcEnd == std::string::npos && paramStart == std::string::npos) {
        // Waiting for more data
    } else if (paramStart != std::string::npos && (funcEnd == std::string::npos || paramStart < funcEnd)) {
        // Next <param
        this->lastProcessedPosition = paramStart + Minicpm5ToolParser::PARAM_START_TAG.size();
        this->currentState = State::InsideParamName;
    } else {
        // </function>
        this->lastProcessedPosition = funcEnd + Minicpm5ToolParser::FUNCTION_END_TAG.size();
        this->currentState = State::Content;
        // Finalise tool call
        std::string argumentsAsString;
        {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            this->currentFunction.argumentsAsDocument.Accept(writer);
            argumentsAsString = buffer.GetString();
        }
        ToolCall toolCall{generateRandomId(), this->currentFunction.name, argumentsAsString};
        SPDLOG_TRACE("Minicpm5: adding tool call: id={}, name={}, params={}", toolCall.id, toolCall.name, toolCall.arguments);
        toolCalls.emplace_back(std::move(toolCall));
        this->currentFunction.clear();
        this->toolCallPositions.end.push(this->lastProcessedPosition);
    }
}

void Minicpm5ToolParserImpl::handleInsideParamNameState() {
    // We are positioned inside <param (after "<param").
    // Find the '>' closing the <param ...> tag.
    auto tagEnd = this->streamContent.find('>', this->lastProcessedPosition);
    if (tagEnd == std::string::npos) {
        SPDLOG_TRACE("Minicpm5: waiting for '>' of <param> tag");
        return;
    }
    auto nameAttr = this->streamContent.find(Minicpm5ToolParser::NAME_ATTR_PREFIX, this->lastProcessedPosition);
    if (nameAttr == std::string::npos || nameAttr >= tagEnd) {
        SPDLOG_DEBUG("Minicpm5: <param> tag has no name= attribute — skipping");
        this->lastProcessedPosition = tagEnd + 1;
        this->currentState = State::InsideFunction;
        return;
    }
    size_t valueStart = nameAttr + Minicpm5ToolParser::NAME_ATTR_PREFIX.size();
    this->currentParameterName = extractNameAttribute(this->streamContent, valueStart, tagEnd);
    if (this->currentParameterName.empty()) {
        SPDLOG_DEBUG("Minicpm5: could not extract param name — skipping");
        this->lastProcessedPosition = tagEnd + 1;
        this->currentState = State::InsideFunction;
        return;
    }
    SPDLOG_TRACE("Minicpm5: param name: {}", this->currentParameterName);
    this->lastProcessedPosition = tagEnd + 1;  // past '>'
    this->currentState = State::InsideParam;
}

void Minicpm5ToolParserImpl::handleInsideParamState() {
    // Read until </param>
    auto endPos = this->streamContent.find(Minicpm5ToolParser::PARAM_END_TAG, this->lastProcessedPosition);
    if (endPos == std::string::npos) {
        SPDLOG_TRACE("Minicpm5: waiting for </param>");
        return;
    }
    std::string paramValue = this->streamContent.substr(this->lastProcessedPosition, endPos - this->lastProcessedPosition);
    addParameterToCurrentFunctionDoc(paramValue);
    this->lastProcessedPosition = endPos + Minicpm5ToolParser::PARAM_END_TAG.size();
    this->currentState = State::InsideFunction;
}

bool Minicpm5ToolParserImpl::parseUntilStateChange(ToolCalls_t& toolCalls) {
    SPDLOG_TRACE("Minicpm5: state: {}", this->currentState);
    auto previousState = this->currentState;

    switch (this->currentState) {
    case State::Content:
        handleInsideContentState();
        break;
    case State::InsideFunctionName:
        handleInsideFunctionNameState();
        break;
    case State::InsideFunction:
        handleInsideFunctionState(toolCalls);
        break;
    case State::InsideParamName:
        handleInsideParamNameState();
        break;
    case State::InsideParam:
        handleInsideParamState();
        break;
    }

    return previousState != this->currentState;
}

std::optional<ToolCalls_t> Minicpm5ToolParserImpl::parseChunk(const std::string& chunk) {
    if (chunk.empty())
        return std::nullopt;
    ToolCalls_t toolCalls;
    this->streamContent += chunk;
    while (parseUntilStateChange(toolCalls)) {
    }
    if (!toolCalls.empty()) {
        return std::move(toolCalls);
    }
    return std::nullopt;
}

std::optional<std::string> Minicpm5ToolParserImpl::getCurrentFunctionName() const {
    if (this->currentFunction.name.empty())
        return std::nullopt;
    return this->currentFunction.name;
}

// ---- Minicpm5ToolParser ----

void Minicpm5ToolParser::lazyFillInitToolParametersTypesMap() {
    if (this->filledParametersTypesMap)
        return;
    SPDLOG_DEBUG("Minicpm5ToolParser: filling tools parameters types map");
    this->toolsParametersTypes = createToolsParametersTypesMap(this->toolSchemas);
    this->filledParametersTypesMap = true;
    SPDLOG_DEBUG("Minicpm5ToolParser: created with {} tools", this->toolsParametersTypes.size());
}

Minicpm5ToolParser::Minicpm5ToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas) :
    BaseOutputParser(tokenizer),
    toolSchemas(toolSchemas),
    streamParser(this->toolsParametersTypes) {}

const std::vector<int64_t> Minicpm5ToolParser::removeReasoningTokens(const std::vector<int64_t>& generatedTokens) {
    std::vector<int64_t> tokensWithoutReasoning;
    tokensWithoutReasoning.reserve(generatedTokens.size());
    auto reasoningStartIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningStartTokenId);
    auto reasoningEndIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningEndTokenId);
    if (reasoningStartIt == generatedTokens.end() && reasoningEndIt == generatedTokens.end()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Minicpm5ToolParser: Reasoning start or end token not found in the generated tokens. Start token found: {}, End token found: {}, Start position: {}, End position: {}",
            reasoningStartIt != generatedTokens.end(), reasoningEndIt != generatedTokens.end(), std::distance(generatedTokens.begin(), reasoningStartIt), std::distance(generatedTokens.begin(), reasoningEndIt));
        tokensWithoutReasoning.insert(tokensWithoutReasoning.end(), generatedTokens.begin(), generatedTokens.end());
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Minicpm5ToolParser: Reasoning tokens found. Start position: {}, End position: {}",
            std::distance(generatedTokens.begin(), reasoningStartIt), std::distance(generatedTokens.begin(), reasoningEndIt));
        if (reasoningStartIt == generatedTokens.end()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Minicpm5ToolParser: Reasoning start wasn't found, but reasoning end was found. Start position: {}, End position: {}",
                std::distance(generatedTokens.begin(), reasoningStartIt), std::distance(generatedTokens.begin(), reasoningEndIt));
            reasoningStartIt = generatedTokens.begin();
        }
        tokensWithoutReasoning.insert(tokensWithoutReasoning.end(), generatedTokens.begin(), reasoningStartIt);
        tokensWithoutReasoning.insert(tokensWithoutReasoning.end(), reasoningEndIt + 1, generatedTokens.end());
    }
    return tokensWithoutReasoning;
}

void Minicpm5ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    auto tokensWithoutReasoning = this->removeReasoningTokens(generatedTokens);
    std::string contentWithSpecialTokens = this->tokenizer.decode(tokensWithoutReasoning, ov::genai::skip_special_tokens(false));
    this->lazyFillInitToolParametersTypesMap();
    auto toolCallsOpt = this->streamParser.parseChunk(contentWithSpecialTokens);
    if (toolCallsOpt.has_value()) {
        parsedOutput.toolCalls = std::move(toolCallsOpt.value());
        SPDLOG_DEBUG("Minicpm5ToolParser: parse done, removing tool calls from content");
        auto status = this->streamParser.removeToolCallsFromContentIfNeeded(contentWithSpecialTokens);
        if (!status.ok()) {
            SPDLOG_DEBUG("Minicpm5ToolParser: failed to remove tool calls from content: {}", status.string());
        }
        parsedOutput.content = std::move(contentWithSpecialTokens);
        return;
    }
    SPDLOG_DEBUG("Minicpm5ToolParser: parse done, no tool calls found");
}

std::optional<rapidjson::Document> Minicpm5ToolParser::sendFullDelta(const ToolCalls_t& toolCalls) {
    if (toolCalls.size() != 1) {
        SPDLOG_ERROR("Minicpm5ToolParser: for streaming expected one tool call, got: {}", toolCalls.size());
        throw std::runtime_error("Minicpm5ToolParser: for streaming expected one tool call");
    }
    auto& toolCall = toolCalls[0];
    // If the first delta was not sent yet (complete tool call in a single chunk),
    // return a combined delta with id, type, name AND arguments.
    if (this->returnedFirstDeltas.find(this->toolCallIndex) == this->returnedFirstDeltas.end() ||
        this->toolCallIndex == -1) {
        int toolCallId = ++this->toolCallIndex;
        this->returnedFirstDeltas.insert(toolCallId);
        this->returnedCompleteDeltas.insert(toolCallId);
        return wrapCombinedDelta(toolCall);
    }
    this->returnedCompleteDeltas.insert(this->toolCallIndex);
    rapidjson::Document argumentsWrapper;
    argumentsWrapper.SetObject();
    rapidjson::Document::AllocatorType& allocator = argumentsWrapper.GetAllocator();
    rapidjson::Value toolCallsString(rapidjson::kStringType);
    toolCallsString.SetString(toolCall.arguments.c_str(), allocator);
    SPDLOG_TRACE("Minicpm5ToolParser: tool call arguments string: {}", toolCall.arguments);
    argumentsWrapper.AddMember("arguments", toolCallsString, allocator);
    auto currentDelta = wrapDelta(argumentsWrapper, this->toolCallIndex);
    SPDLOG_DEBUG("Minicpm5ToolParser: full delta: {}", documentToString(currentDelta));
    return currentDelta;
}

rapidjson::Document Minicpm5ToolParser::wrapCombinedDelta(const ToolCall& toolCall) {
    rapidjson::Document wrappedDelta;
    wrappedDelta.SetObject();
    rapidjson::Document::AllocatorType& allocator = wrappedDelta.GetAllocator();

    rapidjson::Value toolCalls(rapidjson::kArrayType);
    rapidjson::Value toolCallObj(rapidjson::kObjectType);
    rapidjson::Value idValue(generateRandomId().c_str(), allocator);
    toolCallObj.AddMember("id", idValue, allocator);
    toolCallObj.AddMember("type", "function", allocator);
    toolCallObj.AddMember("index", this->toolCallIndex, allocator);

    rapidjson::Value functionObj(rapidjson::kObjectType);
    rapidjson::Value nameValue(toolCall.name.c_str(), allocator);
    functionObj.AddMember("name", nameValue, allocator);
    toolCallObj.AddMember("function", functionObj, allocator);

    rapidjson::Value argumentsValue(rapidjson::kStringType);
    argumentsValue.SetString(toolCall.arguments.c_str(), allocator);
    toolCallObj.AddMember("arguments", argumentsValue, allocator);

    toolCalls.PushBack(toolCallObj, allocator);
    rapidjson::Value deltaWrapper(rapidjson::kObjectType);
    deltaWrapper.AddMember("tool_calls", toolCalls, allocator);
    wrappedDelta.AddMember("delta", deltaWrapper, allocator);
    SPDLOG_DEBUG("Minicpm5ToolParser: combined delta: {}", documentToString(wrappedDelta));
    return wrappedDelta;
}

std::optional<rapidjson::Document> Minicpm5ToolParser::sendFirstDeltaIfNeeded(const std::string& toolCallName) {
    if (this->returnedFirstDeltas.size() == (this->returnedCompleteDeltas.size() + 1)) {
        SPDLOG_TRACE("Minicpm5ToolParser: skipping first delta, already sent for current function");
        return std::nullopt;
    }
    int toolCallId = ++this->toolCallIndex;
    rapidjson::Document doc = wrapFirstDelta(toolCallName, toolCallId);
    this->currentJson.CopyFrom(doc, this->currentJson.GetAllocator());
    this->returnedFirstDeltas.insert(toolCallId);
    SPDLOG_DEBUG("Minicpm5ToolParser: first delta: {}", documentToString(doc));
    return doc;
}

std::optional<rapidjson::Document> Minicpm5ToolParser::parseChunk(
    const std::string& newChunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {
    SPDLOG_DEBUG("Minicpm5ToolParser: chunk: '{}'", newChunk);
    this->lazyFillInitToolParametersTypesMap();
    if (newChunk.empty())
        return std::nullopt;
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
