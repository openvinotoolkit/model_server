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
#include "src/logging.hpp"
#include "src/llm/io_processing/utils.hpp"
#include "src/stringutils.hpp"
#include "tool_parser.hpp"

namespace ovms {

void DevstralToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
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
    ovms::trim(toolName);  // trim in case of extra spaces/newlines
    toolCall.name = toolName;
    if (arguments.empty()) {
        arguments = "{}";  // set empty arguments to {}
    }
    toolCall.arguments = arguments;
    toolCall.id = generateRandomId();  // Generate a random ID for the tool call
    parsedOutput.toolCalls.push_back(toolCall);

    // get subset of generatedTokens starting from begin() to firstArgsTokenIndex
    std::vector<int64_t> contentTokens;
    if (firstToolTokenIndex > 0) {
        contentTokens = std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + firstToolTokenIndex);
        parsedOutput.content = tokenizer.decode(contentTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});  // Return only the content till tool call
    } else {
        parsedOutput.content = tokenizer.decode(contentTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});
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

rapidjson::Document DevstralToolParser::wrapCombinedDelta(ToolCall& toolCall) {
    rapidjson::Document wrappedDelta;
    wrappedDelta.SetObject();
    rapidjson::Value toolCalls(rapidjson::kArrayType);
    rapidjson::Value toolCallObj(rapidjson::kObjectType);
    rapidjson::Value idValue(generateRandomId().c_str(), wrappedDelta.GetAllocator());
    rapidjson::Value toolCallsString(rapidjson::kStringType);

    toolCallObj.AddMember("id", idValue, wrappedDelta.GetAllocator());
    toolCallObj.AddMember("type", "function", wrappedDelta.GetAllocator());
    toolCallObj.AddMember("index", toolCallIndex, wrappedDelta.GetAllocator());
    rapidjson::Value functionObj(rapidjson::kObjectType);
    rapidjson::Value nameValue(toolCall.name.c_str(), wrappedDelta.GetAllocator());
    functionObj.AddMember("name", nameValue, wrappedDelta.GetAllocator());
    // now we need to add string toolCall.arguments to argumentsWrapper under "arguments" key

    toolCallsString.SetString(toolCall.arguments.c_str(), wrappedDelta.GetAllocator());
    functionObj.AddMember("arguments", toolCallsString, wrappedDelta.GetAllocator());
    toolCallObj.AddMember("function", functionObj, wrappedDelta.GetAllocator());
    toolCalls.PushBack(toolCallObj, wrappedDelta.GetAllocator());
    rapidjson::Value deltaWrapper(rapidjson::kObjectType);
    deltaWrapper.AddMember("tool_calls", toolCalls, wrappedDelta.GetAllocator());
    wrappedDelta.AddMember("delta", deltaWrapper, wrappedDelta.GetAllocator());
    return wrappedDelta;
}

rapidjson::Document DevstralToolParser::parseContentChunk() {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("delta");
    writer.StartObject();
    writer.String("content");
    writer.String(streamContent.c_str());
    writer.EndObject();
    writer.EndObject();
    rapidjson::Document doc;
    doc.Parse(buffer.GetString());
    streamContent.clear();
    return doc;
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
                return wrapFirstDelta(this->toolName, this->toolCallIndex);
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

        ToolCall toolCall;
        if (!arguments.empty()) 
            toolCall.arguments = arguments;
        else
            toolCall.arguments = "{}";
        toolCall.name = this->toolName;
        this->streamContent = "";
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
