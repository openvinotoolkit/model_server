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
#include "lfm2_tool_parser.hpp"

namespace ovms {

const std::string Lfm2ToolParser::TOOL_CALL_START_TAG = "<|tool_call_start|>";
const std::string Lfm2ToolParser::TOOL_CALL_END_TAG = "<|tool_call_end|>";

const int64_t Lfm2ToolParser::toolCallStartTokenId = 10;  // <|tool_call_start|>
const int64_t Lfm2ToolParser::toolCallEndTokenId = 11;    // <|tool_call_end|>

bool Lfm2ToolParser::parseNewContent() {
    switch (this->currentState) {
    case State::Content: {
        return parseInContentState(this->streamingContent, this->streamingPosition, this->currentState, this->tagIds);
    }
    case State::ToolCallStarted: {
        auto wasParsedCorrectly = parseInToolCallState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState);
        if (wasParsedCorrectly) {
            this->toolCallIndex++;
        }
        return wasParsedCorrectly;
    }
    case State::ToolCallParameters: {
        return parseInToolCallParametersState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState);
    }
    case State::ToolCallEnded: {
        return parseInToolCallEndedState(this->streamingContent, this->streamingPosition, this->currentState, TOOL_CALL_END_TAG);
    }
    case State::AfterToolCall:
        break;
    }
    return false;
}

std::optional<rapidjson::Document> Lfm2ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        return std::nullopt;
    }

    this->streamingContent += chunk;

    if (parseNewContent()) {
        if (this->currentState == State::ToolCallParameters) {
            return BaseOutputParser::wrapFirstDelta(this->toolCall.name, this->toolCallIndex);
        }
        if (this->currentState == State::ToolCallEnded) {
            return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
        }
        if (this->currentState == State::Content) {
            size_t contentEnd = this->streamingContent.find(TOOL_CALL_START_TAG, this->streamingPosition);
            std::string content;
            if (contentEnd != std::string::npos) {
                content = this->streamingContent.substr(this->streamingPosition, contentEnd - this->streamingPosition);
            } else {
                content = this->streamingContent.substr(this->streamingPosition);
            }
            this->streamingPosition += content.size();
            cutEOSFromContent(content);

            if (!content.empty()) {
                return wrapDeltaContent(content);
            }
        }
        if (this->currentState == State::AfterToolCall) {
            this->currentState = State::Content;
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        if ((this->currentState == State::ToolCallParameters || this->currentState == State::ToolCallEnded) && !this->toolCall.arguments.empty()) {
            return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
        }

        if (this->currentState == State::Content && this->streamingPosition < this->streamingContent.size()) {
            auto content = this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();
            cutEOSFromContent(content);

            if (!content.empty()) {
                return wrapDeltaContent(content);
            }
        }
    }

    return std::nullopt;
}

bool Lfm2ToolParser::parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall) {
    size_t argsPos = toolStr.find(TOOL_ARGS_START_INDICATOR);
    if (argsPos != std::string::npos) {
        std::string toolName = toolStr.substr(0, argsPos);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);

        int argsStrLen = toolStr.length() - argsPos - TOOL_ARGS_START_INDICATOR.length() - TOOL_ARGS_END_INDICATOR.length();
        std::string argsStr = toolStr.substr(argsPos + TOOL_ARGS_START_INDICATOR.length(), argsStrLen);
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed args string: {}", argsStr);
        std::vector<Lfm2ToolParser::Argument> arguments = parseArguments(argsStr);

        toolCall.name = toolName;
        rapidjson::Document argsDoc(rapidjson::kObjectType);
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
        argsWriter.StartObject();
        for (const Lfm2ToolParser::Argument& argument : arguments) {
            argsWriter.Key(argument.name.c_str());
            writeArgumentToWriter(argument.value, argsWriter);
        }
        argsWriter.EndObject();
        toolCall.arguments = sb.GetString();
        toolCall.id = generateRandomId();
        return true;
    }
    return false;
}
}  // namespace ovms
