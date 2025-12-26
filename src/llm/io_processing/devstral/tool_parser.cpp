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

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "../utils.hpp"
#include "src/stringutils.hpp"

namespace ovms {

void DevstralToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;
    // expected format: [TOOL_CALLS]tool_name[ARGS]{"arg1": "value1", ...}
    if (parsedOutput.content.empty() || generatedTokens.size() <= 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No content to parse for tool calls");
        return;
    }
    size_t firstToolTokenIndex;
    auto it = std::find(generatedTokens.begin(), generatedTokens.end(), this->botTokenId);
    if (it != generatedTokens.end()) {
        firstToolTokenIndex = std::distance(generatedTokens.begin(), it);
    } else {
        return;
    }

    size_t firstArgsTokenIndex;
    auto itArgs = std::find(generatedTokens.begin() + firstToolTokenIndex, generatedTokens.end(), this->argsTokenId);
    if (itArgs != generatedTokens.end()) {
        firstArgsTokenIndex = std::distance(generatedTokens.begin(), itArgs);
    } else {
        return;
    }
    if (firstToolTokenIndex > firstArgsTokenIndex) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "First tool token index is greater than first args token index.");
        return;
    }
    std::vector<int64_t> toolNameTokens(generatedTokens.begin() + (firstToolTokenIndex + 1), generatedTokens.begin() + (firstArgsTokenIndex));
    std::vector<int64_t> argumentsTokens(generatedTokens.begin() + (firstArgsTokenIndex + 1), generatedTokens.end());

    ToolCall toolCall;
    std::string toolName = tokenizer.decode(toolNameTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});
    std::string arguments = tokenizer.decode(argumentsTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});
    toolCall.name = toolName;
    toolCall.arguments = arguments;
    toolCall.id = generateRandomId();  // Generate a random ID for the tool call
    parsedOutput.toolCalls.push_back(toolCall);

    // get subset of generatedTokens starting from begin() to firstArgsTokenIndex
    std::vector<int64_t> contentTokens;
    if (firstToolTokenIndex > 0) {
        contentTokens = std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + firstToolTokenIndex);
        parsedOutput.content = tokenizer.decode(contentTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});  // Return only the content till tool call
    } else {
        parsedOutput.content = "";
    }
    return;
}

std::optional<rapidjson::Document> DevstralToolParser::sendFullDelta(ToolCall& toolCall) {
    rapidjson::Document argsDelta;
    argsDelta.Parse(toolCall.arguments.c_str());
    rapidjson::Document argumentsWrapper;
    argumentsWrapper.SetObject();
    rapidjson::Document::AllocatorType& allocator = argumentsWrapper.GetAllocator();
    // now we need to add string toolCall.arguments to argumentsWrapper under "arguments" key
    rapidjson::Value toolCallsString(rapidjson::kStringType);
    toolCallsString.SetString(toolCall.arguments.c_str(), allocator);
    argumentsWrapper.AddMember("arguments", toolCallsString, allocator);
    auto currentDelta = wrapDelta(argumentsWrapper, this->toolCallIndex);
    return currentDelta;
}

std::optional<rapidjson::Document> DevstralToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    /* 
    Devstral [TOOL_CALL]tool_name[ARGS]arguments[</s>]
    It does not support parallel tool calls, so tool calls are always in sequence.

    We have three processing states:
        AWAITING_START_TAG,
        AWAITING_ARGS_TAG,
        PROCESSING_ARGS

    We store the history of chunks in streamContent string. After state changes are detected, we clear the streamContent to only keep unprocessed part.
    */

    this->streamContent += chunk;
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Chunk content: '{}'", chunk);
    if (this->internalState == AWAITING_START_TAG) {
        size_t pos = chunk.find(this->streamingParsingToolCallsStartTag);
        if (pos != std::string::npos) {
            this->internalState = AWAITING_ARGS_TAG;
            this->toolCallIndex++;
            if (pos == 0) {
                this->streamContent.clear();
            } else {
                this->streamContent = this->streamContent.substr(pos + this->streamingParsingToolCallsStartTag.length());  // "[TOOLS_CALLS]" length is 13
            }
        } else {
            return std::nullopt;
        }
    }
    if (this->internalState == AWAITING_ARGS_TAG) {
        // check if [ARGS] tag is present in the chunk and update state accordingly
        size_t pos = this->streamContent.find(this->streamingParsingArgsStartTag);
        if (pos != std::string::npos) {
            this->internalState = PROCESSING_ARGS;
            this->toolName = this->streamContent.substr(0, pos);
            this->streamContent = this->streamContent.substr(pos + this->streamingParsingArgsStartTag.length());  // "[ARGS]" length is 6
            return wrapFirstDelta(this->toolName, this->toolCallIndex);
        } else {
            return std::nullopt;
        }
    }
    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        size_t endPos = this->streamContent.find(this->streamingEndTag);
        std::string arguments;
        if (endPos != std::string::npos) {
            arguments = this->streamContent.substr(0, endPos);
        } else {
            arguments = this->streamContent;
        }
        if (!arguments.empty()) {
            ToolCall toolCall;
            toolCall.arguments = arguments;
            toolCall.name = this->toolName;
            return sendFullDelta(toolCall);
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No valid arguments found in streamContent.");
            return std::nullopt;
        }
    }
    return std::nullopt;
}
}  // namespace ovms
