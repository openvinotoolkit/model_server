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
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "src/logging.hpp"
#include "src/llm/io_processing/utils.hpp"
#include "src/llm/io_processing/onyx/onyx_tool_parser.hpp"

namespace ovms {

const std::string OnyxToolParserImpl::FUNCTIONS_RECIPIENT_TAG = "to=functions.";
const std::string OnyxToolParserImpl::MESSAGE_TAG = "<|message|>";
const std::string OnyxToolParserImpl::END_TAG = "<|eom|>";

#define DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(TAG)                    \
    auto pos = this->streamContent.find(TAG, this->lastProcessedPosition); \
    if (pos == std::string::npos) {                                       \
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Did not find: {}", TAG); \
        break;                                                             \
    }

bool OnyxToolParserImpl::parseUntilStateChange(ToolCalls_t& toolCalls) {
    auto previousState = this->currentState;
    switch (this->currentState) {
    case State::Content: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(FUNCTIONS_RECIPIENT_TAG);
        this->toolCallPositions.begin.push(pos);
        this->lastProcessedPosition = pos + FUNCTIONS_RECIPIENT_TAG.length();
        this->currentState = State::InsideName;
        break;
    }
    case State::InsideName: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(MESSAGE_TAG);
        this->currentFunctionName = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        this->lastProcessedPosition = pos + MESSAGE_TAG.length();
        this->currentState = State::InsideArguments;
        break;
    }
    case State::InsideArguments: {
        DEFINE_TAG_POSITION_AND_BREAK_IF_NOT_FOUND(END_TAG);
        std::string argumentsPart = streamContent.substr(this->lastProcessedPosition, pos - this->lastProcessedPosition);
        this->lastProcessedPosition = pos + END_TAG.length();
        this->currentState = State::Content;
        this->toolCallPositions.end.push(this->lastProcessedPosition);
        ToolCall toolCall{generateRandomId(), this->currentFunctionName, argumentsPart};
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Adding tool call: id={}, name={}, arguments={}", toolCall.id, toolCall.name, toolCall.arguments);
        toolCalls.emplace_back(std::move(toolCall));
        this->currentFunctionName.clear();
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
    if (this->currentFunctionName.empty()) {
        return std::nullopt;
    }
    return this->currentFunctionName;
}

Status OnyxToolParserImpl::removeToolCallsFromContentIfNeeded(std::string& outContent) {
    if (toolCallPositions.begin.size() != toolCallPositions.end.size()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Mismatched tool tags, begin: {}, end: {}", toolCallPositions.begin.size(), toolCallPositions.end.size());
        return Status(StatusCode::INTERNAL_ERROR, "Mismatched tool tags");
    }
    while (!toolCallPositions.begin.empty() && !toolCallPositions.end.empty()) {
        auto posBegin = toolCallPositions.begin.top();
        auto posEnd = toolCallPositions.end.top();
        // Also consume the leading " " the chat template renders before "to=" (a single
        // generate() call only ever produces one such segment, see OnyxReasoningParser's
        // class comment for why this can't collide with anything preceding it).
        if (posBegin > 0 && outContent[posBegin - 1] == ' ') {
            posBegin -= 1;
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Removing tool call from outContent begin:{}, end:{}, removing:{}", posBegin, posEnd, outContent.substr(posBegin, posEnd - posBegin));
        outContent.erase(posBegin, posEnd - posBegin);
        toolCallPositions.begin.pop();
        toolCallPositions.end.pop();
    }
    return StatusCode::OK;
}

void OnyxToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // <|start|>assistant to=functions.<name><|message|>{raw JSON args}<|eom|>
    //
    // Mirrors Qwen3CoderToolParser::parse(): drive the same streamParser used for
    // streaming with the whole content as a single chunk, and reuse whatever it
    // assembled -- unary is a single-shot edge case of streaming, not a parallel
    // reimplementation of the tag walk.
    auto toolCallsOpt = this->streamParser.parseChunk(parsedOutput.content);
    if (!toolCallsOpt.has_value()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing ended, no tool calls found");
        return;
    }
    parsedOutput.toolCalls = std::move(toolCallsOpt.value());
    for (const auto& toolCall : parsedOutput.toolCalls) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unary | Onyx Tool | id: [{}], name: [{}], arguments: [{}]", toolCall.id, toolCall.name, toolCall.arguments);
    }
    auto status = this->streamParser.removeToolCallsFromContentIfNeeded(parsedOutput.content);
    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to remove tool calls from content: {}", status.string());
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
    // ASSUMPTION: mirroring Qwen3CoderToolParser, in streaming we only ever complete one
    // tool call per parseChunk() call -- there is no way to send multiple tool calls to
    // the client in a single streaming delta.
    if (toolCalls.size() != 1) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "For streaming we expected one tool call, got: {}", toolCalls.size());
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
    // streamParser returns assembled toolCalls once a call closes ("<|eom|>" seen); until
    // then, if the function name is already known, send the first delta for it once.
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

