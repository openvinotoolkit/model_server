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
#include "./lfm25_tool_parser.hpp"

namespace ovms {

const std::string Lfm25ToolParser::TOOL_CALL_START_TAG = "<|tool_call_start|>";
const std::string Lfm25ToolParser::TOOL_CALL_END_TAG = "<|tool_call_end|>";

const int64_t Lfm25ToolParser::toolCallStartTokenId = 124905; // <|tool_call_start|>
const int64_t Lfm25ToolParser::toolCallEndTokenId = 124906;   // <|tool_call_end|>
const int64_t Lfm25ToolParser::reasoningEndTokenId = 124902;  // </think>

bool Lfm25ToolParser::parseNewContent() {
    switch (currentState) {
    case State::Content: {
        return parseInContentState(this->streamingContent, this->streamingPosition, TOOL_CALL_START_TAG, TOOL_CALL_END_TAG);
    }
    case State::ToolCallStarted: {
        auto wasParsedCorrectly = parseInToolCallState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState);
        if (wasParsedCorrectly) {
            this->toolCallIndex++;
        }
        return wasParsedCorrectly;
    }
    case State::ToolCallParameters: {
        return parseToolCallParametersState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState);
    }
    case State::ToolCallEnded: {
        return parseInToolCallEndedState(this->streamingContent, this->streamingPosition, this->currentState, TOOL_CALL_END_TAG);
    }
    case State::AfterToolCall:
        break;
    }
    return false;
}

std::optional<rapidjson::Document> Lfm25ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
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

void Lfm25ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    parseUnaryResponse(parsedOutput, generatedTokens, tokenizer, toolCallStartTokenId, toolCallEndTokenId, reasoningEndTokenId);
}
}  // namespace ovms
