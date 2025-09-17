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
    Due to the tool call format used by Phi4, we need to track the state of parsing more closely.
    We have four states:
    1) AWAITING_START_TAG - we are waiting for the "functools" tag to appear in the chunk
    2) AWAITING_TOOL_CALLS_OPENING_BRACKET - we have seen "functools" but are waiting for the opening bracket of the array
    3) AWAITING_TOOL_CALL_OPENING_BRACE - we have seen the opening bracket of the array but are waiting for the opening brace of the next tool call object
    4) PROCESSING_TOOL_CALL - we are processing the tool call object

    To avoid missing any generated content, we use unprocessedBuffer to store any output that is not used in the current state, but might be relevant in the next state.
    Since tools calls in the array are separated by commas we also need to track when the tool call object ends (no special tag for that).
    Next challenge, common for all parsers, is to return arguments as string even though model generates them as JSON.
    We address this by escaping double quotes and adding opening quote at the beginning of arguments and closing quote at the end of arguments.
    */
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Phi4ToolParser::parseChunk called with chunk: '{}', finishReason: {}", chunk, static_cast<int>(finishReason));
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Phi4ToolParser");
        return std::nullopt;
    }

    // We merge unprocessedBuffer from previous calls with the current chunk to avoid losing any content
    std::string modifiedChunk = unprocessedBuffer + chunk;
    unprocessedBuffer.clear();

    // Phase 1: Control the internal state and apply changes to the chunk if needed
    if (internalState == AWAITING_START_TAG) {
        // We did not see "functools" yet, so we look for it in the current chunk
        if (modifiedChunk.find(parsingStartTag) != std::string::npos) {
            // We found "functools", so we switch to the the state where we are waiting for the opening bracket of the array
            internalState = AWAITING_TOOL_CALLS_OPENING_BRACKET;
            if (modifiedChunk.length() > parsingStartTag.length()) {
                // We have more content in the chunk after "functools", so we process the rest of the chunk in the next state
                std::string remainingChunk = modifiedChunk.substr(modifiedChunk.find(parsingStartTag) + parsingStartTag.length());
                if (remainingChunk.empty()) {
                    return std::nullopt;  // Nothing more to process in this chunk
                } else {
                    return parseChunk(remainingChunk, finishReason);
                }
            } else {                  // modifiedChunk.length() == parsingStartTag.length() as at this state, chunk cannot be smaller
                return std::nullopt;  // Nothing more to process in this chunk
            }
        }
        return std::nullopt;
    } else if (internalState == AWAITING_TOOL_CALLS_OPENING_BRACKET) {
        // Next chunk after "functools" should start with opening bracket of the array
        if (modifiedChunk[0] == '[') {
            // We found the opening bracket, so we switch to waiting for the opening brace of the first tool call
            internalState = AWAITING_TOOL_CALL_OPENING_BRACE;

            // We process the rest of the chunk after the opening bracket
            std::string remainingChunk = modifiedChunk.substr(1);
            if (remainingChunk.empty()) {
                return std::nullopt;  // Nothing more to process in this chunk
            } else {
                // Process the remaining chunk as part of tool call processing
                return parseChunk(remainingChunk, finishReason);
            }
        } else {
            // Still waiting for the opening bracket, ignore this chunk
            return std::nullopt;
        }
    } else if (internalState == AWAITING_TOOL_CALL_OPENING_BRACE) {
        // We are waiting for the opening brace of the tool call object
        size_t firstOpeningBrace = modifiedChunk.find_first_of('{');
        if (firstOpeningBrace != std::string::npos) {
            internalState = PROCESSING_TOOL_CALL;
            // Clear state for the next tool call
            lastJson.Clear();
            jsonBuilder.clear();
            toolCallIndex++;
            argumentsQuotesOpened = false;
            openBracesCount = 1;  // Reset to 1 as we count just found opening brace of the tool call

            // Process the rest of the chunk after the opening brace (brace included) as part of tool call processing
            std::string remainingChunk = modifiedChunk.substr(firstOpeningBrace);
            if (remainingChunk.empty()) {
                return std::nullopt;  // Nothing more to process in this chunk
            } else {
                return parseChunk(remainingChunk, finishReason);
            }
        } else {
            // Still waiting for the opening brace, ignore this chunk
            return std::nullopt;
        }
    } else {  // internalState == PROCESSING_TOOL_CALL
        // Remove any newlines to avoid breaking JSON format
        modifiedChunk.erase(std::remove(modifiedChunk.begin(), modifiedChunk.end(), '\n'), modifiedChunk.end());

        // JSON already contains 'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
        if (lastJson.HasMember("arguments")) {
            // Escaping double quotes in the arguments string
            for (size_t pos = 0; (pos = modifiedChunk.find("\"", pos)) != std::string::npos; pos += 2) {
                modifiedChunk.insert(pos, "\\");
            }

            // Keep track of opened/closed braces to identify the end of the tool call object.
            // Note that this method can be fooled by unclosed braces in string values.
            // If turns out insufficient, we will need full JSON parsing to track opened/closed braces for arguments.
            for (char c : modifiedChunk) {
                if (c == '{') {
                    openBracesCount++;
                } else if (c == '}') {
                    openBracesCount--;
                }
            }

            // When we start collecting arguments, force string type by adding opening quote
            if (!argumentsQuotesOpened) {
                // Add opening quote before the first non-whitespace character
                size_t firstNonWhitespaceCharacter = modifiedChunk.find_first_not_of(" \t\n\r\f\v");
                if (firstNonWhitespaceCharacter != std::string::npos) {
                    modifiedChunk.insert(firstNonWhitespaceCharacter, "\"");
                } else {
                    // If the chunk is all whitespace, just insert quote at the end
                    modifiedChunk.append("\"");
                }
                argumentsQuotesOpened = true;
            }

            if (finishReason != ov::genai::GenerationFinishReason::NONE) {
                // If generation has stopped, we look for the closing brace to close the string properly
                size_t lastClosingBrace = modifiedChunk.find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    modifiedChunk.insert(lastClosingBrace, "\"");
                }
            } else if (openBracesCount == 0) {
                // If we balanced the braces, we are at the end of the tool call object, so we add closing quote before the last closing brace
                size_t lastClosingBrace = modifiedChunk.find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    modifiedChunk.insert(lastClosingBrace, "\"");
                } else {
                    // If there is no closing brace, we just add closing quote at the end
                    modifiedChunk.append("\"");
                }
            }
        } else {  // no arguments yet, we need to make sure they are added only as a key
            // If 'arguments":' appears in the chunk and there is any non-whitespace content after it, which is not string,
            // we add double quote after colon to force string type
            size_t argumentsPos = modifiedChunk.find("arguments\":");
            if (argumentsPos != std::string::npos) {
                // Move everything after 'arguments":' to unprocessedBuffer, so we can add opening quote at the beginning of arguments in the next call
                size_t afterArgumentsPos = argumentsPos + std::string("arguments\":").length();
                if (afterArgumentsPos < modifiedChunk.length()) {
                    unprocessedBuffer = modifiedChunk.substr(afterArgumentsPos);
                    modifiedChunk.erase(afterArgumentsPos);
                }
            }
        }

        // Phase 2: Parse the modified chunk with PartialJsonBuilder and return appropriate delta if possible
        rapidjson::Document newJson;
        try {
            // Otherwise just push the current chunk
            newJson = jsonBuilder.add(modifiedChunk);
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

            // Handle the case when tool call has finished - store unprocessed output and switch internal state
            if (jsonBuilder.isComplete()) {
                unprocessedBuffer = jsonBuilder.getUnprocessedBuffer();
                // Remove potential escape characters added in arguments processing logic from the unprocessedBuffer as we move to the next tool call
                unprocessedBuffer.erase(
                    std::remove(unprocessedBuffer.begin(), unprocessedBuffer.end(), '\\'),
                    unprocessedBuffer.end());
                // Switch to the state where we are waiting for the opening brace of the next tool call object
                internalState = AWAITING_TOOL_CALL_OPENING_BRACE;
            } else {
                lastJson.CopyFrom(newJson, lastJson.GetAllocator());
            }

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
