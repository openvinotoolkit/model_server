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
#include <vector>
#include <regex>

#include "src/logging.hpp"
#include "src/llm/io_processing/utils.hpp"
#include "src/stringutils.hpp"
#include "tool_parser.hpp"

namespace ovms {

std::optional<Delta> DevstralToolParser::sendFullDelta(ToolCall& toolCall) {
    return ToolCallDelta{this->toolCallIndex, std::nullopt, std::nullopt, toolCall.arguments};
}

ToolCallDelta DevstralToolParser::wrapCombinedDelta(ToolCall& toolCall) {
    return ToolCallDelta{this->toolCallIndex, generateRandomId(), toolCall.name, toolCall.arguments};
}

ContentDelta DevstralToolParser::parseContentChunk() {
    ContentDelta d{std::move(streamContent)};
    streamContent.clear();
    return d;
}

std::optional<Delta> DevstralToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    /* 
    Devstral [TOOL_CALL]tool_name[ARGS]arguments[</s>]
    It does not support parallel tool calls, so tool calls are always in sequence.

    We have three processing states:
        AWAITING_START_TAG,
        AWAITING_ARGS_TAG,
        PROCESSING_ARGS

    We store the history of chunks in streamContent string. After state changes are detected, we clear the streamContent to only keep unprocessed part.
    */

    // Ignore no-op empty chunks when there is nothing buffered to flush.
    // Keep processing empty STOP chunks only when streamContent already holds
    // pending argument text (missing end-tag finalization path).
    if (chunk.empty() && this->streamContent.empty()) {
        return std::nullopt;
    }

    this->streamContent += chunk;
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Chunk content: '{}', StreamContent: '{}', State: {}", chunk, this->streamContent, std::to_string(this->internalState));
    if (this->internalState == AWAITING_START_TAG) {
        // if chunk ends with </s> we need to remove it and return parsed content immediately
        if (chunk.size() >= this->parsingEndTag.size() &&
            chunk.substr(chunk.size() - this->parsingEndTag.size()) == this->parsingEndTag) {
            // remove </s> from streamContent
            this->streamContent = this->streamContent.substr(0, this->streamContent.size() - this->parsingEndTag.size());
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Found end tag in chunk while awaiting start tag. Returning content chunk.");
            return parseContentChunk();
        }
        size_t pos = chunk.find(this->parsingToolCallsStartTag);
        if (pos != std::string::npos) {
            this->internalState = AWAITING_ARGS_TAG;
            this->toolCallIndex++;
            if (pos == 0) {
                this->streamContent.clear();
                return std::nullopt;
            } else {
                this->streamContent = this->streamContent.substr(pos + this->parsingToolCallsStartTag.length());  // "[TOOLS_CALLS]" length is 13
                return parseContentChunk();
            }
        } else {
            return parseContentChunk();
        }
    }
    if (this->internalState == AWAITING_ARGS_TAG) {
        size_t pos = this->streamContent.find(this->parsingArgsStartTag);
        if (pos == std::string::npos) {
            // [ARGS] not found — check if generation has ended (end tag or finish reason).
            // Flush whatever accumulated as plain content so it is not silently dropped.
            size_t endPos = this->streamContent.find(this->parsingEndTag);
            if (endPos != std::string::npos || finishReason != ov::genai::GenerationFinishReason::NONE) {
                if (endPos != std::string::npos) {
                    this->streamContent = this->streamContent.substr(0, endPos);
                }
                if (!this->streamContent.empty()) {
                    return parseContentChunk();
                }
            }
            return std::nullopt;
        }
        if (pos != std::string::npos) {
            this->internalState = PROCESSING_ARGS;
            this->toolName = this->streamContent.substr(0, pos);
            ovms::trim(this->toolName);  // trim in case of extra spaces/newlines
            this->streamContent = this->streamContent.substr(pos + this->parsingArgsStartTag.length());
            // check if chunk ends with </s>, if yes, we need return full tool call delta
            if (this->streamContent.size() >= this->parsingEndTag.size() &&
                this->streamContent.substr(this->streamContent.size() - this->parsingEndTag.size()) == this->parsingEndTag) {
                // remove </s> from streamContent
                ToolCall toolCall;
                toolCall.name = this->toolName;
                this->streamContent = this->streamContent.substr(0, this->streamContent.size() - this->parsingEndTag.size());
                if (!this->streamContent.empty()) {
                    toolCall.arguments = this->streamContent;
                } else {
                    toolCall.arguments = "{}";
                }
                this->streamContent = "";
                return wrapCombinedDelta(toolCall);
            } else {
                return ToolCallDelta{this->toolCallIndex, generateRandomId(), this->toolName, ""};
            }
        } else {
            return std::nullopt;
        }
    }
    if (this->internalState == PROCESSING_ARGS) {
        size_t endPos = this->streamContent.find(this->parsingEndTag);
        std::string arguments;
        if (endPos != std::string::npos) {
            arguments = this->streamContent.substr(0, endPos);
        } else {
            arguments = this->streamContent;
        }

        // When the end tag arrives with no preceding argument content and we have already emitted
        // argument content in prior calls (e.g. char-by-char feeding via parse()), suppress the
        // spurious "{}" delta that would otherwise be appended to the accumulated arguments.
        if (arguments.empty() && argumentsEmitted) {
            this->streamContent = "";
            return std::nullopt;
        }

        ToolCall toolCall;
        if (!arguments.empty())
            toolCall.arguments = arguments;
        else
            toolCall.arguments = "{}";
        toolCall.name = this->toolName;
        this->streamContent = "";
        argumentsEmitted = !arguments.empty();
        return sendFullDelta(toolCall);
    }
    return std::nullopt;
}
// Static member definitions
const std::string DevstralToolParser::parsingArgsStartTag = "[ARGS]";
const std::string DevstralToolParser::parsingToolCallsStartTag = "[TOOL_CALLS]";
const std::string DevstralToolParser::parsingEndTag = "</s>";
const int64_t DevstralToolParser::argsTokenId = 32;  // [ARGS]
const int64_t DevstralToolParser::botTokenId = 9;    // [TOOL_CALLS]
}  // namespace ovms
