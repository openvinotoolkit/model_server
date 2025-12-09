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

    if (parsedOutput.content.empty() || generatedTokens.size() <= 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No content to parse for tool calls");
        return;
    }

    // Parser will consume entire model output only if the first generated token is the beginning of tools token.
    // expected format: [TOOL_CALLS]tool_name[ARGS]{"arg1": "value1", ...}

    //size_t pos = 0;
    //size_t firstToolCallPos;

    // Save position of the first tool call start tag to properly clear content after parsing.
    //firstToolCallPos = parsedOutput.content.find("[TOOL_CALLS]", pos);
    //find position in vector generatedTokens with value 9
    size_t firstToolTokenIndex;
    auto it = std::find(generatedTokens.begin(), generatedTokens.end(), this->botTokenId);
    if (it != generatedTokens.end()) {
        firstToolTokenIndex = std::distance(generatedTokens.begin(), it);
    } else {
        return;
    }
  
    size_t firstArgsTokenIndex;
    auto it_args = std::find(generatedTokens.begin() + firstToolTokenIndex, generatedTokens.end(), this->argsTokenId);
    if (it_args != generatedTokens.end()) {
        firstArgsTokenIndex = std::distance(generatedTokens.begin(), it_args);
    } else {
        return;
    }
    if (firstToolTokenIndex > firstArgsTokenIndex) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "First tool token index is greater than first args token index.");
        return;
    }
    std::vector<int64_t> tool_name_tokens(generatedTokens.begin() + (firstToolTokenIndex + 1), generatedTokens.begin() + (firstArgsTokenIndex));
    std::vector<int64_t> arguments_tokens(generatedTokens.begin() + (firstArgsTokenIndex + 1), generatedTokens.end());
    
    ToolCall toolCall;
    std::string tool_name = tokenizer.decode(tool_name_tokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});
    if (this->toolSchemas.find(tool_name) == this->toolSchemas.end()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool name '{}' not valid.", tool_name);
        return;
    }
    std::string arguments = tokenizer.decode(arguments_tokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});

    toolCall.name = tool_name;
    toolCall.arguments = arguments;
    toolCall.id = generateRandomId();  // Generate a random ID for the tool call
    parsedOutput.toolCalls.push_back(toolCall);

    // get subset of generatedTokens starting from begin() to firstArgsTokenIndex
    std::vector<int64_t> content_tokens;
    if (firstToolTokenIndex > 0) {
        content_tokens = std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + firstToolTokenIndex);
        parsedOutput.content = tokenizer.decode(content_tokens, ov::AnyMap{ov::genai::skip_special_tokens(true)}); // Return only the contnet till tool call in content
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
    if (this->internalState == AWAITING_START_TAG) {
        size_t pos = chunk.find("[TOOL_CALLS]");
        if (pos != std::string::npos) {
            this->internalState = AWAITING_ARGS_TAG;
            this->toolCallIndex++;
            if (pos == 0) {
                this->streamContent.clear();
            } else {
                this->streamContent = this->streamContent.substr(pos + 13); // "[TOOLS_CALLS]" length is 13
            }
        } else {
            return std::nullopt;
        }
    }
    if (this->internalState == AWAITING_ARGS_TAG) {
        //check if [ARGS] tag is present in the chunk and update state accordingly
        size_t pos = this->streamContent.find("[ARGS]");
        if (pos != std::string::npos) {
            this->internalState = PROCESSING_ARGS;
            this->toolName = this->streamContent.substr(0, pos);
            if (this->toolSchemas.find(this->toolName) == this->toolSchemas.end()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool name '{}' not valid.", this->toolName);
                return std::nullopt;
            }
            this->streamContent = this->streamContent.substr(pos + 6); // "[ARGS]" length is 6
            return wrapFirstDelta(this->toolName, this->toolCallIndex);
        } else {
            return std::nullopt;
        }
    }
    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        size_t end_pos = this->streamContent.find("</s>");
        std::string arguments;
        if (end_pos != std::string::npos) {
            arguments = this->streamContent.substr(0, end_pos);
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
