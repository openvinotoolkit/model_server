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

#include "../../logging.hpp"
#include "qwen3_response_parser.hpp"
#include "utils.hpp"

namespace ovms {
ParsedResponse Qwen3ResponseParser::parse(const std::vector<int64_t>& generatedTokens) {
    ParsedResponse parsedResponse;

    auto reasoningStartIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningStartTokenId);
    auto reasoningEndIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningEndTokenId);

    if (reasoningStartIt != generatedTokens.end() && reasoningEndIt != generatedTokens.end() && reasoningStartIt < reasoningEndIt) {
        // Tokens between <think> and </think>, exclusive
        std::vector<int64_t> reasoningTokens(reasoningStartIt + 1, reasoningEndIt);
        parsedResponse.reasoning = tokenizer.decode(reasoningTokens);
        parsedResponse.reasoningTokenCount = reasoningTokens.size();
    } else {
        parsedResponse.reasoning.clear();
        parsedResponse.reasoningTokenCount = 0;
    }

    // If reasoning happened, we assume that the content starts after the reasoning end tag,
    // otherwise we assume that the content starts from the beginning of the generated_ids
    auto contentStartIt = reasoningEndIt != generatedTokens.end() ? reasoningEndIt + 1 : generatedTokens.begin();
    // Assuming content ends when tool calls start, so we find the first occurrence of <tool_call> after the content start
    auto contentEndIt = std::find(contentStartIt, generatedTokens.end(), toolCallStartTokenId);

    if (contentStartIt == contentEndIt) {
        // If tool call tag starts immediately after reasoning or at the beginning, we assume no content
        parsedResponse.content.clear();
    } else if (contentStartIt != generatedTokens.end() && contentEndIt != generatedTokens.end() && contentStartIt < contentEndIt) {
        // Tokens between start or reasoning end and the first <tool_call> tag, exclusive
        parsedResponse.content = tokenizer.decode(std::vector<int64_t>(contentStartIt, contentEndIt));
    } else {
        // If no tool call tags are found, we assume the content is the rest of the generated tokens
        parsedResponse.content = tokenizer.decode(std::vector<int64_t>(contentStartIt, generatedTokens.end()));
    }

    // Remove all leading whitespace from the content only if reasoning is present since they are separate reasoning part from the actual content
    if (parsedResponse.reasoningTokenCount > 0 && !parsedResponse.content.empty()) {
        size_t first_non_ws = parsedResponse.content.find_first_not_of(" \t\n\r\f\v");
        if (first_non_ws != std::string::npos) {
            parsedResponse.content = parsedResponse.content.substr(first_non_ws);
        } else {
            parsedResponse.content.clear();
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
        parsedResponse.toolCalls.push_back(toolCall);
    }
    return parsedResponse;
}

std::optional<rapidjson::Document> Qwen3ResponseParser::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3ResponseParser");
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
    To know when we reach the end of the arguments string, we track the nesting level of arguments (it's supposed to be valid JSON).
    When we reach the opening brace '{', we increase the nesting level, and when we reach the closing brace '}', we decrease it.
    When we reach the closing brace '}' and nesting level is zero, we add a closing quote '"' right after it to complete the string and keep the main JSON valid.
    */
    } else if (processingPhase == ProcessingPhase::TOOL_CALLS) {
        // Assuming streamer will provide start/end tag either alone in the chunk or with whitespaces that can be dropped.
        if (chunk.find(toolCallEndTag) != std::string::npos) {
            lastJson.Clear();
            jsonBuilder.clear();
            return std::nullopt;
        } else if (chunk.find(toolCallStartTag) != std::string::npos) {
            toolCallIndex++;
            argumentsNestingLevel = 0;
            return std::nullopt;
        } else {
            std::string modifiedChunk = chunk;
            // JSON already contains 'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
            if (lastJson.HasMember("arguments")) {
                // Escaping double quotes in the arguments string
                for (size_t pos = 0; (pos = modifiedChunk.find("\"", pos)) != std::string::npos; pos += 2) {
                    modifiedChunk.insert(pos, "\\");
                }
                // Tracking nesting level and applying closure after arguments are complete
                for (size_t i = 0; i < modifiedChunk.size(); ++i) {
                    char c = modifiedChunk[i];
                    if (c == '{') {
                        argumentsNestingLevel++;
                    } else if (c == '}') {
                        argumentsNestingLevel--;
                        if (argumentsNestingLevel == 0) {
                            modifiedChunk.insert(i + 1, "\"");
                            break;
                        }
                    }
                }
            }

            rapidjson::Document newJson;
            // Push modified chunk to the JSON builder and collect new partial JSON
            try {
                newJson = jsonBuilder.add(modifiedChunk);
            } catch (const std::exception& e) {
                std::cout << "Failed to parse tool call arguments: " << e.what() << std::endl;
                return std::nullopt;  // If parsing fails, we return nullopt
            }

            // Case 1: 'arguments' has just appeared in the current chunk. If so, we return first delta.
            if (newJson.HasMember("arguments") && !lastJson.HasMember("arguments")) {
                // If 'arguments' is null we add double quote to force string data type.
                if (newJson["arguments"].IsNull()) {
                    jsonBuilder.add("\"");
                }
                // Wrap first delta in {"tool_calls":[{"id":<id>,"type":"function","index":<toolCallIndex>,"function":<delta>}]}
                doc = wrapFirstDelta(lastJson["name"].GetString(), toolCallIndex);
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
