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

rapidjson::Document Qwen3ResponseParser::parseChunk(const std::string& chunk) {
    if (chunk.empty()) {
        // TODO: decide how to handle that
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3ResponseParser");
        return rapidjson::Document();
    }

    rapidjson::Document doc;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    std::string content, reasoningContent, toolCall;
    if (processingPhase == ProcessingPhase::CONTENT) {
        if (chunk.find(reasoningStartTag) != std::string::npos) {
            // If we find reasoning start tag, we switch to REASONING phase
            processingPhase = ProcessingPhase::REASONING;
            size_t tagPos = chunk.find(reasoningStartTag);
            content = (tagPos > 0) ? chunk.substr(0, tagPos) : "";
            reasoningContent = (tagPos != std::string::npos && tagPos + reasoningStartTag.length() < chunk.size())
                ? chunk.substr(tagPos + reasoningStartTag.length())
                : "";

            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            if (!content.empty()) {
                writer.String("content");
                writer.String(content.c_str());
            }
            if (!reasoningContent.empty()) {
                writer.String("reasoning_content");
                writer.String(reasoningContent.c_str());
            }
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        } else if (chunk.find(toolCallStartTag) != std::string::npos) {
            // If we find tool call start tag, we switch to TOOL_CALLS phase
            processingPhase = ProcessingPhase::TOOL_CALLS;
            size_t tagPos = chunk.find(toolCallStartTag);
            content = (tagPos > 0) ? chunk.substr(0, tagPos) : "";
            toolCall = (tagPos != std::string::npos && tagPos + toolCallStartTag.length() < chunk.size())
                ? chunk.substr(tagPos + toolCallStartTag.length())
                : "";
            toolCallBuffer += toolCall; // we are not processing tool calls in this step, but need to keep tool call content for later
            jsonBuilder.partialParseToJson("{\"tool_calls\": ["); // initialize JSON builder for tool calls
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            if (!content.empty()) {
                writer.String("content");
                writer.String(content.c_str());
            }
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        } else {
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
    } else if (processingPhase == ProcessingPhase::REASONING) {
        if (chunk.find(reasoningEndTag) != std::string::npos) {
            // If we find reasoning end tag, we switch to CONTENT phase
            processingPhase = ProcessingPhase::CONTENT;
            size_t tagPos = chunk.find(reasoningEndTag);
            reasoningContent = (tagPos > 0) ? chunk.substr(0, tagPos) : "";
            std::string content;
            if (tagPos != std::string::npos && tagPos + reasoningEndTag.length() < chunk.size()) {
                content = chunk.substr(tagPos + reasoningEndTag.length());
            }
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            if (!reasoningContent.empty()) {
                writer.String("reasoning_content");
                writer.String(reasoningContent.c_str());
            }
            if (!content.empty()) {
                writer.String("content");
                writer.String(content.c_str());
            }
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        } else {
            reasoningContent = chunk;
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            writer.String("reasoning_content");
            writer.String(chunk.c_str());
            writer.EndObject();
            writer.EndObject();
            doc.Parse(buffer.GetString());
            return doc;
        }
    }

    if (processingPhase == ProcessingPhase::TOOL_CALLS) {
        if (chunk.find(toolCallEndTag) != std::string::npos) {
            // If we find tool call end tag, we process the tool call buffer
            size_t tagPos = chunk.find(toolCallEndTag);
            std::string toolCallContent = (tagPos > 0) ? chunk.substr(0, tagPos) : "";
            if (!toolCallContent.empty()) {
                rapidjson::Document currentDoc = jsonBuilder.partialParseToJson(toolCallContent);
                rapidjson::Document delta = computeDelta(lastJson, currentDoc);
                doc.SetObject();
                doc.AddMember("delta", delta, doc.GetAllocator());
                lastJson.CopyFrom(currentDoc, doc.GetAllocator());
                return doc;
            }
        } else if (chunk.find(toolCallStartTag) != std::string::npos) {
            // If we find another tool call start tag, we process the previous tool call buffer
            size_t tagPos = chunk.find(toolCallStartTag);
            std::string toolCallContent;
            if (tagPos != std::string::npos && tagPos + toolCallStartTag.length() < chunk.size()) {
                toolCallContent = chunk.substr(tagPos + toolCallStartTag.length());
            } else {
                toolCallContent = "";
            }

            if (!toolCallContent.empty()) {
                rapidjson::Document currentDoc = jsonBuilder.partialParseToJson(toolCallContent);
                rapidjson::Document delta = computeDelta(lastJson, currentDoc);
                doc.SetObject();
                doc.AddMember("delta", delta, doc.GetAllocator());
                lastJson.CopyFrom(currentDoc, doc.GetAllocator());
                return doc;
            }
        } else {
                rapidjson::Document currentDoc = jsonBuilder.partialParseToJson(chunk);
                rapidjson::Document delta = computeDelta(lastJson, currentDoc);
                doc.SetObject();
                doc.AddMember("delta", delta, doc.GetAllocator());
                lastJson.CopyFrom(currentDoc, doc.GetAllocator());
                return doc;
        }
    }
    // If none of the above conditions are met, return an empty document
    // todo change type to optional
    return rapidjson::Document();
}
}  // namespace ovms
