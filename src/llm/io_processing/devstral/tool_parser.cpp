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
    Devstral [TOOL_CALLS]tool_name[ARGS]arguments[TOOL_CALLS]tool_name[ARGS]arguments[</s>]

    We have three processing states:
        AWAITING_START_TAG,
        AWAITING_ARGS_TAG,
        PROCESSING_ARGS

    We store the history of chunks in streamContent string. After state changes are detected, we clear the streamContent to only keep unprocessed part.
    */

    // Ignore no-op empty chunks when there is nothing buffered to flush.
    // Keep processing empty STOP chunks only when streamContent already holds
    // pending argument text (missing end-tag finalization path).
    if (chunk.empty() && this->streamContent.empty() &&
        (finishReason == ov::genai::GenerationFinishReason::NONE ||
            this->internalState != PROCESSING_ARGS || argumentsEmitted)) {
        return std::nullopt;
    }

    this->streamContent += chunk;
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Chunk content: '{}', StreamContent: '{}', State: {}", chunk, this->streamContent, std::to_string(this->internalState));
    if (this->internalState == PROCESSING_ARGS) {
        size_t pos = this->streamContent.find(this->parsingToolCallsStartTag);
        if (pos != std::string::npos) {
            std::optional<Delta> delta;
            if (pos != 0 || !argumentsEmitted) {
                delta = ToolCallDelta{this->toolCallIndex, std::nullopt, std::nullopt,
                    pos != 0 ? this->streamContent.substr(0, pos) : "{}"};
            }
            this->streamContent = this->streamContent.substr(pos + this->parsingToolCallsStartTag.size());
            this->internalState = AWAITING_ARGS_TAG;
            this->toolCallIndex++;
            this->toolName.clear();
            argumentsEmitted = false;
            if (delta.has_value()) {
                return delta;
            }
        }
    }
    if (this->internalState == AWAITING_START_TAG) {
        // if chunk ends with </s> we need to remove it and return parsed content immediately
        if (chunk.size() >= this->parsingEndTag.size() &&
            chunk.substr(chunk.size() - this->parsingEndTag.size()) == this->parsingEndTag) {
            // remove </s> from streamContent
            this->streamContent = this->streamContent.substr(0, this->streamContent.size() - this->parsingEndTag.size());
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Found end tag in chunk while awaiting start tag. Returning content chunk.");
            return parseContentChunk();
        }
        size_t pos = this->streamContent.find(this->parsingToolCallsStartTag);
        if (pos != std::string::npos) {
            this->internalState = AWAITING_ARGS_TAG;
            this->toolCallIndex++;
            argumentsEmitted = false;
            std::string content = this->streamContent.substr(0, pos);
            this->streamContent = this->streamContent.substr(pos + this->parsingToolCallsStartTag.length());
            if (!content.empty()) {
                return ContentDelta{std::move(content)};
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
        } else {
            this->internalState = PROCESSING_ARGS;
            this->toolName = this->streamContent.substr(0, pos);
            ovms::trim(this->toolName);  // trim in case of extra spaces/newlines
            this->streamContent = this->streamContent.substr(pos + this->parsingArgsStartTag.length());
            size_t nextCallPos = this->streamContent.find(this->parsingToolCallsStartTag);
            if (nextCallPos != std::string::npos) {
                ToolCall toolCall;
                toolCall.name = this->toolName;
                toolCall.arguments = nextCallPos != 0 ? this->streamContent.substr(0, nextCallPos) : "{}";
                auto delta = wrapCombinedDelta(toolCall);
                this->streamContent.erase(0, nextCallPos + this->parsingToolCallsStartTag.size());
                this->internalState = AWAITING_ARGS_TAG;
                this->toolCallIndex++;
                this->toolName.clear();
                argumentsEmitted = false;
                return delta;
            }
            const bool endsWithEndTag = this->streamContent.size() >= this->parsingEndTag.size() &&
                                        this->streamContent.substr(this->streamContent.size() - this->parsingEndTag.size()) == this->parsingEndTag;
            if (endsWithEndTag || finishReason != ov::genai::GenerationFinishReason::NONE) {
                ToolCall toolCall;
                toolCall.name = this->toolName;
                if (endsWithEndTag) {
                    this->streamContent.resize(this->streamContent.size() - this->parsingEndTag.size());
                }
                if (!this->streamContent.empty()) {
                    toolCall.arguments = this->streamContent;
                } else {
                    toolCall.arguments = "{}";
                }
                this->streamContent = "";
                argumentsEmitted = true;
                return wrapCombinedDelta(toolCall);
            } else {
                return ToolCallDelta{this->toolCallIndex, generateRandomId(), this->toolName, ""};
            }
        }
        return std::nullopt;
    }
    if (this->internalState == PROCESSING_ARGS) {
        size_t endPos = this->streamContent.find(this->parsingEndTag);
        std::string arguments;
        if (endPos != std::string::npos) {
            arguments = this->streamContent.substr(0, endPos);
            this->streamContent.clear();
        } else {
            size_t argumentSize = this->streamContent.size();
            if (finishReason == ov::genai::GenerationFinishReason::NONE) {
                for (size_t prefixSize = 1; prefixSize < this->parsingToolCallsStartTag.size() && prefixSize <= this->streamContent.size(); ++prefixSize) {
                    if (this->streamContent.compare(this->streamContent.size() - prefixSize, prefixSize,
                            this->parsingToolCallsStartTag, 0, prefixSize) == 0) {
                        argumentSize = this->streamContent.size() - prefixSize;
                    }
                }
            }
            arguments = this->streamContent.substr(0, argumentSize);
            this->streamContent.erase(0, argumentSize);
            if (arguments.empty() && finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;
            }
        }

        // Suppress the spurious "{}" delta that would otherwise be appended to the accumulated arguments.
        if (arguments.empty() && argumentsEmitted) {
            return std::nullopt;
        }

        ToolCall toolCall;
        if (!arguments.empty())
            toolCall.arguments = arguments;
        else
            toolCall.arguments = "{}";
        toolCall.name = this->toolName;
        argumentsEmitted = true;
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
