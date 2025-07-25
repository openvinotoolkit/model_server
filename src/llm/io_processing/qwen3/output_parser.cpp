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
#include "output_parser.hpp"
#include "../utils.hpp"

namespace ovms {
ParsedOutput Qwen3OutputParser::parse(const std::vector<int64_t>& generatedTokens) {
    ParsedOutput parsedOutput;

    auto reasoningStartIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningStartTokenId);
    auto reasoningEndIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningEndTokenId);

    if (reasoningStartIt != generatedTokens.end() && reasoningEndIt != generatedTokens.end() && reasoningStartIt < reasoningEndIt) {
        // Tokens between <think> and </think>, exclusive
        std::vector<int64_t> reasoningTokens(reasoningStartIt + 1, reasoningEndIt);
        parsedOutput.reasoning = tokenizer.decode(reasoningTokens);
        parsedOutput.reasoningTokenCount = reasoningTokens.size();
    } else {
        parsedOutput.reasoning.clear();
        parsedOutput.reasoningTokenCount = 0;
    }

    // If reasoning happened, we assume that the content starts after the reasoning end tag,
    // otherwise we assume that the content starts from the beginning of the generated_ids
    auto contentStartIt = reasoningEndIt != generatedTokens.end() ? reasoningEndIt + 1 : generatedTokens.begin();
    // Assuming content ends when tool calls start, so we find the first occurrence of <tool_call> after the content start
    auto contentEndIt = std::find(contentStartIt, generatedTokens.end(), toolCallStartTokenId);

    if (contentStartIt == contentEndIt) {
        // If tool call tag starts immediately after reasoning or at the beginning, we assume no content
        parsedOutput.content.clear();
    } else if (contentStartIt != generatedTokens.end() && contentEndIt != generatedTokens.end() && contentStartIt < contentEndIt) {
        // Tokens between start or reasoning end and the first <tool_call> tag, exclusive
        parsedOutput.content = tokenizer.decode(std::vector<int64_t>(contentStartIt, contentEndIt));
    } else {
        // If no tool call tags are found, we assume the content is the rest of the generated tokens
        parsedOutput.content = tokenizer.decode(std::vector<int64_t>(contentStartIt, generatedTokens.end()));
    }

    // Remove all leading whitespace from the content only if reasoning is present since they are separate reasoning part from the actual content
    if (parsedOutput.reasoningTokenCount > 0 && !parsedOutput.content.empty()) {
        size_t first_non_ws = parsedOutput.content.find_first_not_of(" \t\n\r\f\v");
        if (first_non_ws != std::string::npos) {
            parsedOutput.content = parsedOutput.content.substr(first_non_ws);
        } else {
            parsedOutput.content.clear();
        }
    }

    // Assuming tool calls are the last part of the output
    auto it = contentStartIt;
    std::vector<std::string> tools;
    while (it != generatedTokens.end()) {
        // Find the next <tool_call> tag
        auto toolCallStartIt = std::find(it, generatedTokens.end(), toolCallStartTokenId);
        if (toolCallStartIt == generatedTokens.end()) {
            break;
        }
        // Find the next </tool_call> tag after <tool_call>
        auto toolCallEndIt = std::find(toolCallStartIt + 1, generatedTokens.end(), toolCallEndTokenId);
        if (toolCallEndIt == generatedTokens.end()) {
            break;
        }
        // Extract tokens between <tool_call> and </tool_call>
        std::vector<int64_t> toolTokens(toolCallStartIt + 1, toolCallEndIt);
        std::string tool = tokenizer.decode(toolTokens);
        if (!tool.empty()) {
            tools.push_back(tool);
        }
        it = toolCallEndIt + 1;
    }

    for (const std::string& tool : tools) {
        ToolCall toolCall;
        toolCall.id = generateRandomId();  // Generate a random ID for the tool call
        rapidjson::Document toolDoc;
        toolDoc.Parse(tool.c_str());
        if (toolDoc.HasParseError()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse tool call as JSON");
            continue;
        }
        if (toolDoc.HasMember("name") && toolDoc["name"].IsString()) {
            toolCall.name = toolDoc["name"].GetString();
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
        parsedOutput.toolCalls.push_back(toolCall);
    }
    return parsedOutput;
}

std::optional<rapidjson::Document> Qwen3OutputParser::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3OutputParser");
        return std::nullopt;
    }

    rapidjson::Document doc;
    /* 
    Case 1: We are in CONTENT phase, from here we can switch to REASONING or TOOL_CALLS if we find starting tags.
    If we switch phase, we return nullopt, so no message is streamed back in this call. Otherwise we stream content delta.
    */
    if (processingPhase == ProcessingPhase::CONTENT) {
        if (chunk.find(reasoningStartTag) != std::string::npos) {
            processingPhase = ProcessingPhase::REASONING;
            return std::nullopt;
        } else if (chunk.find(toolCallStartTag) != std::string::npos) {
            processingPhase = ProcessingPhase::TOOL_CALLS;
            toolCallIndex++;
            return std::nullopt;
        } else {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            writer.String("content");
            writer.String(chunk.c_str());
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        }
        /* 
    Case 2: We are in REASONING phase, from here we can switch to CONTENT if we find ending tags.
    If we switch phase, we return nullopt, so no message is streamed back in this call. Otherwise we stream reasoning_content delta (temporarily streaming content).
    */
    } else if (processingPhase == ProcessingPhase::REASONING) {
        if (chunk.find(reasoningEndTag) != std::string::npos) {
            processingPhase = ProcessingPhase::CONTENT;
            return std::nullopt;
        } else {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            // writer.String("reasoning_content");
            writer.String("content");  // temporarily using "content" to make it work with agentic client
            writer.String(chunk.c_str());
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        }
        /* 
    Case 3: We are in TOOL_CALLS phase, which is the last phase of request processing.
    Start and end tags in this phase modify state of the processing, but do not return any message.
    Otherwise we collect data until we have full function name - that's when we return the first delta.
    Every next delta contains next parts of the arguments. Qwen3 generates arguments as JSON, but OpenAI API expects them in a string format.
    That's why once we reach 'arguments' key, we add double quote to force string type and escape all double quotes that come in next parts.
    To know when we reach the end of the arguments string, we return delta with a one chunk delay. This way, when we reach end of tool call, we modify previous chunk to close
    arguments string properly and return such modified chunk. 
    */
    } else if (processingPhase == ProcessingPhase::TOOL_CALLS) {
        // Assuming streamer will provide start/end tag either alone in the chunk or with whitespaces that can be dropped.
        if (chunk.find(toolCallStartTag) != std::string::npos) {
            lastJson.Clear();
            jsonBuilder.clear();
            toolCallIndex++;
            argumentsDelayWindow[0].clear();
            argumentsDelayWindow[1].clear();
            return std::nullopt;
        } else {
            // JSON already contains 'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
            if (lastJson.HasMember("arguments")) {
                std::string modifiedChunk = chunk;
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

                // If we are closing the tool call, we need to add closing quote after the last closing brace that we assume is present in the chunk processed in the last call.
                // Otherwise we just store the chunk in the second element of the delay array to be handled in the next call.
                if (modifiedChunk.find(toolCallEndTag) != std::string::npos) {
                    size_t lastClosingBrace = argumentsDelayWindow[0].find_last_of('}');
                    if (lastClosingBrace != std::string::npos) {
                        argumentsDelayWindow[0].insert(lastClosingBrace, "\"");
                    }
                } else {
                    argumentsDelayWindow[1] = modifiedChunk;
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
                    newJson = jsonBuilder.add(chunk);
                }
            } catch (const std::exception& e) {
                (void)e;  // Suppress unused variable warning on Windows
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call chunk partial parse failed: {}", e.what());
                // Throwing an error since at this point the JSON is broken and next chunks will not make it right.
                throw std::runtime_error("Generated tool call structure is not valid");
            }

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
    }
    return std::nullopt;
}
}  // namespace ovms
