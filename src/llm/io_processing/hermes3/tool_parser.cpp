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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "../utils.hpp"
#include "../../../stringutils.hpp"

namespace ovms {

void Hermes3ToolParser::movePostColonContentToUnprocessedBuffer(std::string& chunk) {
    size_t colonPos = chunk.find(':');
    if (colonPos != std::string::npos) {
        // Store everything after the colon in unprocessedBuffer to process in the next call
        unprocessedBuffer = chunk.substr(colonPos + 1) + unprocessedBuffer;
        // Keep everything up to and including the colon
        chunk = chunk.substr(0, colonPos + 1);
    }
}

void Hermes3ToolParser::movePostEndTagContentToUnprocessedBuffer(std::string& chunk) {
    size_t endTagPos = chunk.find(toolCallEndTag);
    if (endTagPos != std::string::npos) {
        // Store everything after the end tag in unprocessedBuffer to process in the next call
        unprocessedBuffer = chunk.substr(endTagPos + toolCallEndTag.length()) + unprocessedBuffer;
        // Keep everything up to and including the end tag
        chunk = chunk.substr(0, endTagPos + toolCallEndTag.length());
    }
}

void Hermes3ToolParser::moveStartTagWithContentToUnprocessedBuffer(std::string& chunk) {
    size_t startTagPos = chunk.find(toolCallStartTag);
    if (startTagPos != std::string::npos && startTagPos != 0) {
        // Store everything from the start tag in unprocessedBuffer to process in the next call
        unprocessedBuffer = chunk.substr(startTagPos) + unprocessedBuffer;
        // Keep everything before the start tag
        chunk = chunk.substr(0, startTagPos);
    }
    // If startTag is at position 0, do nothing since there is no content before start tag
}

bool Hermes3ToolParser::findAndHandleStartTag(std::string& chunk) {
    size_t startTagPos = chunk.find(toolCallStartTag);
    if (startTagPos != std::string::npos) {
        clearState();
        toolCallIndex++;

        size_t afterTagPos = startTagPos + toolCallStartTag.length();
        if (afterTagPos < chunk.size()) {
            // Remove everything before and including the start tag
            chunk = chunk.substr(afterTagPos);
        } else {
            // We only have the start tag which is special value, so we do not return delta.
            chunk.clear();
        }
        return true;
    }
    return false;
}

bool Hermes3ToolParser::findAndHandleEndTag(std::string& chunk) {
    size_t endTagPos = chunk.find(toolCallEndTag);
    if (endTagPos != std::string::npos) {
        toolCallCompleted = true;
        // Remove special tag from the chunk
        chunk = chunk.substr(0, endTagPos);
        return true;
    }
    return false;
}

void Hermes3ToolParser::openArgumentsString(std::string& chunk) {
    size_t firstNonWhitespaceCharacter = chunk.find_first_not_of(" \t\n\r\f\v");
    if (firstNonWhitespaceCharacter != std::string::npos) {
        chunk.insert(firstNonWhitespaceCharacter, "\"");
    } else {
        // If the chunk is all whitespace, just insert at the end
        chunk.append("\"");
    }
}

void Hermes3ToolParser::closeArgumentsString(std::string& chunk) {
    size_t lastClosingBrace = chunk.find_last_of('}');
    if (lastClosingBrace != std::string::npos) {
        chunk.insert(lastClosingBrace, "\"");
    } else {
        // If there is no closing brace, we just add closing quote at the end of the chunk as arguments must still be a string.
        chunk.append("\"");
    }
}

void Hermes3ToolParser::clearState() {
    unprocessedBuffer.clear();
    jsonBuilder.clear();
    lastJson.SetObject();
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
}

void Hermes3ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    const std::string startTag = "<tool_call>";
    const std::string endTag = "</tool_call>";
    std::vector<std::string> tools;
    size_t pos = 0;
    size_t firstToolCallPos;

    // If immediate parsing is enabled, we assume tool calls start from the beginning of the content.
    // Otherwise, we search for the first occurrence of the tool call start tag.
    if (!immediateParsingEnabled) {
        firstToolCallPos = parsedOutput.content.find(startTag, pos);
    } else {
        // Read first tool call without opening tag
        firstToolCallPos = 0;
        size_t end = parsedOutput.content.find(endTag, firstToolCallPos);
        std::string tool;
        if (end != std::string::npos) {
            tool = parsedOutput.content.substr(0, end);
            pos = end + endTag.length();
        } else {
            tool = parsedOutput.content;
            pos = parsedOutput.content.length();
        }
        if (!tool.empty()) {
            tools.push_back(tool);
        }
    }

    while (true) {
        size_t start = parsedOutput.content.find(startTag, pos);
        if (start == std::string::npos) {
            break;
        }
        start += startTag.length();
        size_t end = parsedOutput.content.find(endTag, start);
        std::string tool;
        if (end != std::string::npos) {
            tool = parsedOutput.content.substr(start, end - start);
            pos = end + endTag.length();
        } else {
            tool = parsedOutput.content.substr(start);
            pos = parsedOutput.content.length();
        }
        if (!tool.empty()) {
            tools.push_back(tool);
        }
    }

    for (const std::string& tool : tools) {
        ToolCall toolCall;
        rapidjson::Document toolDoc;
        toolDoc.Parse(tool.c_str());
        if (toolDoc.HasParseError()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse tool call as JSON");
            continue;
        }
        if (toolDoc.HasMember("name") && toolDoc["name"].IsString()) {
            toolCall.name = toolDoc["name"].GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid name field");
            continue;
        }

        if (toolDoc.HasMember("arguments") && toolDoc["arguments"].IsObject()) {
            rapidjson::StringBuffer sb;
            rapidjson::Writer<rapidjson::StringBuffer> toolWriter(sb);
            toolDoc["arguments"].Accept(toolWriter);
            toolCall.arguments = sb.GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid parameters object");
            continue;
        }
        toolCall.id = generateRandomId();  // Generate a random ID for the tool call
        parsedOutput.toolCalls.push_back(toolCall);
    }
    // Remove tool calls from the content
    if (firstToolCallPos != std::string::npos) {
        parsedOutput.content.erase(firstToolCallPos);
    }
}

std::optional<rapidjson::Document> Hermes3ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    /* 
    We first collect data until we have full function name - that's when we return the first delta.
    Every next delta contains next parts of the arguments. Hermes3 generates arguments as JSON, but OpenAI API expects them in a string format.
    That's why once we reach 'arguments' key, we add double quote to force string type and escape all double quotes that come in next parts.
    To know when we reach the end of the arguments string, we return delta with a one-chunk delay. This way, when we reach end of tool call, we modify previous chunk to close
    arguments string properly and return such modified chunk.
    */

    /*
    PHASE 0: Prepare data and state for processing
    - If previous call finished tool call (received </tool_call> tag), we clear state.
    - If current call finishes tool call (finishReason != NONE), we set flag to clear state in the next call.
    - If chunk is empty, we return std::nullopt.
    - We prepend unprocessedBuffer to the chunk and clear unprocessedBuffer.
    */

    // Check if previous call finished tool call (received </tool_call> tag)
    if (toolCallCompleted) {
        clearState();
        toolCallCompleted = false;
    }

    toolCallCompleted = (finishReason != ov::genai::GenerationFinishReason::NONE);

    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Hermes3ToolParser");
        return std::nullopt;
    }

    std::string modifiedChunk = unprocessedBuffer + chunk;
    unprocessedBuffer.clear();

    /*
    PHASE 1: Split chunk if needed for more convenient processing
    We want to make sure that:
    1) We do not have both key and value in the chunk if we do not have 'arguments' key in the JSON yet.
       This is because we need special handling of the value for arguments.
    2) We do not have end tag in the middle of the chunk. If we do, we split the chunk and process everything before and including the end tag in this call while
       everything after the end tag is processed in the next call.
    3) We do not have start tag in the middle of the chunk. If we do, we split the chunk and process everything before the start tag in this call while everything after
       and including the start tag is processed in the next call.
    */

    // If effective, after this operation modifiedChunk contains content up to and including the first colon while content after it is in unprocessedBuffer.
    if (argumentsDelayWindow[0].empty()) {
        movePostColonContentToUnprocessedBuffer(modifiedChunk);
    }

    // If effective, after this operation modifiedChunk contains content up to and including the end tag while content after it is in unprocessedBuffer.
    movePostEndTagContentToUnprocessedBuffer(modifiedChunk);

    // If effective, after this operation modifiedChunk contains content before the start tag while start tag with content after it is in unprocessedBuffer.
    moveStartTagWithContentToUnprocessedBuffer(modifiedChunk);

    // At this point modifiedChunk have either:
    // 1) (text)+
    // 2) (text)*</tool_call>
    // 3) <tool_call>(text)*

    // We assume single chunk will not contain whole tool call i.e. <tool_call>(text)*</tool_call>
    // In case of (text)*</tool_call><tool_call>(text)*, the chunk is split.

    /* PHASE 2: Handle start and end tags
    In this phase we get rid of special tags from the chunk and set flags to handle state and close arguments properly.
    If we have only special tag in the chunk, we return std::nullopt as there is nothing to add to the JSON builder.
    The exception is when we have end tag and 'arguments' in JSON as we need to close the arguments in delayed chunk properly.
    */

    // Handle chunk with start and end tags
    // Remove end tag from the chunk and set flag to clear state in the next call
    auto endTagFound = findAndHandleEndTag(modifiedChunk);
    if (endTagFound && modifiedChunk.empty() && !lastJson.HasMember("arguments")) {
        return std::nullopt;  // We only had end tag which is special value, so we do not return delta.
    }

    // Remove start tag from the chunk, clear state and increase tool call index
    auto startTagFound = findAndHandleStartTag(modifiedChunk);
    if (startTagFound && modifiedChunk.empty()) {
        return std::nullopt;  // We only had start tag which is special value, so we do not return delta.
    }

    // At this point modifiedChunk contains only content without any special tags.

    /* PHASE 3: If we have 'arguments' in JSON, we need to modify the chunk to keep JSON valid
    - If we are starting to collect arguments, we add opening quote before the first non-whitespace character.
    - We escape all double quotes in the chunk.
    - If we are finishing tool call, we add closing quote before the last closing brace.
    - We use argumentsDelayWindow to store last two chunks of arguments to return delta with one-chunk delay for proper string closing mechanism.
    */

    if (lastJson.HasMember("arguments")) {
        // Since inside a string, we need to escape characters like quotes, new lines, tabs, etc.
        escapeSpecialCharacters(modifiedChunk);

        bool processingFirstArgumentsChunk = argumentsDelayWindow[0].empty();
        // Handle the case when we are starting to collect arguments.
        // Force arguments string type and fill first element of the delay array.
        if (processingFirstArgumentsChunk) {
            // If we are starting to collect arguments, we add opening quote before the first non-whitespace character
            openArgumentsString(modifiedChunk);
            argumentsDelayWindow[0] = modifiedChunk;
        } else if (!argumentsDelayWindow[1].empty()) {
            // We already have two chunks, so we can move delay window forward
            argumentsDelayWindow[0] = argumentsDelayWindow[1];
            if (toolCallCompleted) {
                // There will be no next call for that tool call, so we merge current chunk to the last one
                // to make sure we don't miss returning the last part of model output.
                argumentsDelayWindow[0] += modifiedChunk;
            }
        }

        // We received end tag in the current chunk or finish reason is not NONE, so we need to close arguments string properly.
        if (toolCallCompleted) {
            closeArgumentsString(argumentsDelayWindow[0]);
        }

        if (processingFirstArgumentsChunk && !toolCallCompleted) {
            return std::nullopt;  // We just started collecting arguments and tool call is not finished yet, so we don't have anything to return
        }

        // At this point argumentsDelayWindow[0] contains the chunk to be pushed to the JSON builder in this call
        // while modifiedChunk contains the chunk to be pushed in the next call (if any).
        argumentsDelayWindow[1] = modifiedChunk;
    }

    /* PHASE 4: Add chunk to the JSON builder and compute delta if possible
    We have three cases:
    1) 'arguments' has just appeared in the current chunk. If so, we return first delta.
    2) 'arguments' already exists in the last JSON, we compute delta and return it.
    3) No 'arguments' exists or just appeared, so we keep building up until we have complete function name
    */
    rapidjson::Document newJson;
    try {
        if (!argumentsDelayWindow[0].empty()) {
            // Push delayed chunk to the JSON builder if we are processing arguments
            newJson = jsonBuilder.add(argumentsDelayWindow[0]);
        } else {
            // Otherwise just push the current chunk
            newJson = jsonBuilder.add(modifiedChunk);
        }
    } catch (const std::exception& e) {
        (void)e;  // Suppress unused variable warning on Windows
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call chunk partial parse failed: {}", e.what());
        // Throwing an error since at this point the JSON is broken and next chunks will not make it right.
        throw std::runtime_error("Generated tool call structure is not valid");
    }

    rapidjson::Document doc;
    // Case 1: 'arguments' has just appeared in the current chunk. If so, we return first delta.
    if (newJson.HasMember("arguments") && !lastJson.HasMember("arguments")) {
        std::string functionName;
        if (lastJson.HasMember("name") && lastJson["name"].IsString()) {
            functionName = lastJson["name"].GetString();
        } else if (newJson.HasMember("name") && newJson["name"].IsString()) {
            // We received big chunk with both full function name and arguments, so we get function name from the new JSON
            functionName = newJson["name"].GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call name has not been generated and arguments already started");
            throw std::runtime_error("Tool call name is missing in generated output");
        }
        // Wrap first delta in {"tool_calls":[{"id":<id>,"type":"function","index":<toolCallIndex>,"function":{"name": <functionName>}}]}
        doc = wrapFirstDelta(functionName, toolCallIndex);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        return doc;
        // Case 2: 'arguments' already exists in the last JSON, we compute delta and return it.
    } else if (lastJson.HasMember("arguments")) {
        rapidjson::Document delta = PartialJsonBuilder::computeDelta(lastJson, newJson);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        // If delta is empty or contains only null or empty string values, we don't stream anything.
        if (delta.ObjectEmpty()) {
            return std::nullopt;
        }
        for (auto it = delta.MemberBegin(); it != delta.MemberEnd(); ++it) {
            if (it->value.IsNull() || (it->value.IsString() && std::string(it->value.GetString()).empty())) {
                return std::nullopt;
            }
        }
        // Wrap delta in {"tool_calls":[{"index":<toolCallIndex>,"function":<delta>}]}
        doc = wrapDelta(delta, toolCallIndex);
        return doc;
        // Case 3: No 'arguments' exists or just appeared, so we keep building up until we have complete function name
    } else {
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
    }
    return std::nullopt;
}
}  // namespace ovms
