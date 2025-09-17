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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "../utils.hpp"

namespace ovms {

void Phi4ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;

    // Phi4 with vLLM template produces tool calls in the format:
    // functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]

    std::string toolsStartString = "functools";
    size_t toolsStartPos = 0;
    // If "functools" has been injected we assume the whole generated output is an array with tool calls,
    // otherwise we search for the "functools" tag in the content.
    if (!immediateParsingEnabled) {
        toolsStartPos = parsedOutput.content.find(toolsStartString);
    }

    if (toolsStartPos != std::string::npos) {
        // Extract the tools part, assuming it's all the remaining content after "functools" or entire content if immediate parsing is enabled
        std::string toolsString = immediateParsingEnabled ? parsedOutput.content : parsedOutput.content.substr(toolsStartPos + toolsStartString.length());
        rapidjson::Document toolsDoc;
        toolsDoc.Parse(toolsString.c_str());
        if (!toolsDoc.HasParseError() && toolsDoc.IsArray()) {
            for (auto& toolVal : toolsDoc.GetArray()) {
                if (!toolVal.IsObject()) {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call is not a valid JSON object");
                    continue;
                }
                ToolCall toolCall;
                toolCall.id = generateRandomId();  // Generate a random ID for the tool call
                if (toolVal.HasMember("name") && toolVal["name"].IsString()) {
                    toolCall.name = toolVal["name"].GetString();
                } else {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid name field");
                    continue;
                }

                if (toolVal.HasMember("arguments") && toolVal["arguments"].IsObject()) {
                    rapidjson::StringBuffer sb;
                    rapidjson::Writer<rapidjson::StringBuffer> toolWriter(sb);
                    toolVal["arguments"].Accept(toolWriter);
                    toolCall.arguments = sb.GetString();
                } else {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid parameters object");
                    continue;
                }
                parsedOutput.toolCalls.push_back(toolCall);
            }
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse functools content or extract tools array");
        }
        // Remove the tools part from the content
        parsedOutput.content.erase(toolsStartPos);
    }
}

std::optional<rapidjson::Document> Phi4ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
/* 
    Start and end tags in this phase modify state of the processing, but do not return any message.
    Otherwise we collect data until we have full function name - that's when we return the first delta.
    Every next delta contains next parts of the arguments. Phi4 generates arguments as JSON, but OpenAI API expects them in a string format.
    That's why once we reach 'arguments' key, we add double quote to force string type and escape all double quotes that come in next parts.
    To know when we reach the end of the arguments string, we return delta with a one-chunk delay. This way, when we reach end of tool call, we modify previous chunk to close
    arguments string properly and return such modified chunk.
    
    Assuming streamer will provide start/end tag either alone in the chunk or with whitespaces that can be dropped.
    */
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Phi4ToolParser::parseChunk called with chunk: '{}', finishReason: {}", chunk, static_cast<int>(finishReason));
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Phi4ToolParser");
        return std::nullopt;
    }

    if (internalState == AWAITING_START_TAG) {
        if (chunk.find(parsingStartTag) != std::string::npos) {
            lastJson.Clear();
            jsonBuilder.clear();
            toolCallIndex++;
            argumentsDelayWindow[0].clear();
            argumentsDelayWindow[1].clear();
            if (chunk.length() > parsingStartTag.length()) {
                // We have more content in the chunk after the start tag, so we process it as part of tool call processing
                internalState = AWAITING_TOOL_CALLS_OPENING_BRACKET;
                std::string remainingChunk = chunk.substr(chunk.find(parsingStartTag) + parsingStartTag.length());
                if (remainingChunk.empty()) {
                    return std::nullopt;  // Nothing more to process in this chunk
                } else {
                    return parseChunk(remainingChunk, finishReason);
                }
            } else { // chunk.length() == parsingStartTag.length() as at this state, chunk cannot be smaller
                internalState = AWAITING_TOOL_CALLS_OPENING_BRACKET;
                return std::nullopt;  // Nothing more to process in this chunk
            }
        }
        return std::nullopt;
    } else if (internalState == AWAITING_TOOL_CALLS_OPENING_BRACKET) {
        // Next chunk after "functools" should start with opening bracket of the array
        if (chunk[0] == '[') {
            internalState = PROCESSING_TOOL_CALLS;
            // We process the rest of the chunk after the opening bracket
            std::string remainingChunk = chunk.substr(1);
            if (remainingChunk.empty()) {
                return std::nullopt;  // Nothing more to process in this chunk
            } else {
                // Process the remaining chunk as part of tool call processing
                return parseChunk(remainingChunk, finishReason);
            }
        } else {
            // Still waiting for the opening bracket, ignore this chunk
            // TODO: we could get back to content phase since no tool calls can appear from now on. No API to that at the moment.
            return std::nullopt;
        }
    } else { // internalState == PROCESSING_TOOL_CALLS
        std::string modifiedChunk = unprocessedBuffer + chunk;
        unprocessedBuffer.clear();
        modifiedChunk.erase(std::remove(modifiedChunk.begin(), modifiedChunk.end(), '\n'), modifiedChunk.end());
        // JSON already contains 'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
        if (lastJson.HasMember("arguments")) {
            // Escaping double quotes in the arguments string
            for (size_t pos = 0; (pos = modifiedChunk.find("\"", pos)) != std::string::npos; pos += 2) {
                modifiedChunk.insert(pos, "\\");
            }

            // Handle the case when we are starting to collect arguments.
            // Force arguments string type and fill first element of the delay array.
            if (argumentsDelayWindow[0].empty()) {
                // If we are starting to collect arguments, we add opening quote before the first non-whitespace character
                size_t firstNonWhitespaceCharacter = modifiedChunk.find_first_not_of(" \t\n\r\f\v");
                if (firstNonWhitespaceCharacter != std::string::npos) {
                    modifiedChunk.insert(firstNonWhitespaceCharacter, "\"");
                } else {
                    // If the chunk is all whitespace, just insert at the end
                    modifiedChunk.append("\"");
                }
                argumentsDelayWindow[0] = modifiedChunk;
                return std::nullopt;  // We don't return anything yet, we need to collect next chunk
            }

            if (!argumentsDelayWindow[1].empty()) {
                // We already have two chunks, so we can move delay window forward
                argumentsDelayWindow[0] = argumentsDelayWindow[1];
            }
        
            if (jsonBuilder.isComplete()) {
                size_t lastClosingBrace = argumentsDelayWindow[0].find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    argumentsDelayWindow[0].insert(lastClosingBrace, "\"");
                }
                // If generation has stopped and we didn't get closing tag, we still look for the closing brace to close the string properly
                // Regardless of the closing brace presence we merge current chunk with the last one to make sure we don't miss returning the last part of model output.
            } else if (finishReason != ov::genai::GenerationFinishReason::NONE) {
                size_t lastClosingBrace = modifiedChunk.find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    modifiedChunk.insert(lastClosingBrace, "\"");
                }
                argumentsDelayWindow[0] += modifiedChunk;
                // Otherwise we just store the chunk in the second element of the delay array to be handled in the next call.
            } else {
                argumentsDelayWindow[1] = modifiedChunk;
            }
        } else { // no arguments yet, we need to make sure they are added only as a key
            // If 'arguments":' appears in the chunk, if there is any non-whitespace content after that is not string, we add double quote after colon to force string type
            size_t argumentsPos = modifiedChunk.find("arguments\":");
            if (argumentsPos != std::string::npos) {
                // Move everything after 'arguments":' to unprocessedBuffer
                size_t afterArgumentsPos = argumentsPos + std::string("arguments\":").length();
                if (afterArgumentsPos < modifiedChunk.length()) {
                    unprocessedBuffer = modifiedChunk.substr(afterArgumentsPos);
                    modifiedChunk.erase(afterArgumentsPos);
                }
            }
        }

        rapidjson::Document newJson;
        // Push delayed chunk to the JSON builder
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
    }
    return std::nullopt;
}
}  // namespace ovms
