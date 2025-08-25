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
#include <utility>

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
void Llama3ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // TODO: check if we can rely on decoded <|python_tag|> token to be present in the content, so we can drop multiple detokenizations and copies
    // and just extract substrings from the content and modify content in-place

    // If immediate trigger parsing is enabled, we assume botTokenId has been injected into the prompt and whole output are tool calls,
    // otherwise we search for botTokenId in the generatedTokens to find tool calls start or check if the content starts with "{" (llama3 sometimes does not generate botTokenId)
    auto toolCallsStartPosition = generatedTokens.begin();
    if (!immediateParsingEnabled) {
        toolCallsStartPosition = generatedTokens.end();
        // Find botTokenId in generated_ids
        auto botTokenIt = std::find(generatedTokens.begin(), generatedTokens.end(), botTokenId);

        if (botTokenIt != generatedTokens.end()) {
            // Decode the content before botTokenId
            std::vector<int64_t> contentTokens(generatedTokens.begin(), botTokenIt);
            parsedOutput.content = tokenizer.decode(contentTokens);
            // Tokens after botTokenId will be treated as tool calls
            toolCallsStartPosition = botTokenIt + 1;
        } else {
            // If botTokenId is not found, check if model output starts with "{" and if so, assume it's a tool call"
            if (!parsedOutput.content.empty() && parsedOutput.content[0] == '{') {
                // If model output starts with "{", treat it as a tool call
                toolCallsStartPosition = generatedTokens.begin();
                parsedOutput.content.clear();
            }
        }
    } else {
        parsedOutput.content.clear();
    }

    if (toolCallsStartPosition != generatedTokens.end()) {
        std::vector<int64_t> toolCallsTokens(toolCallsStartPosition, generatedTokens.end());
        std::string toolsResponse = tokenizer.decode(toolCallsTokens);

        std::vector<std::string> tools;
        size_t start = 0;
        size_t end = 0;
        while ((end = toolsResponse.find(separator, start)) != std::string::npos) {
            std::string tool = toolsResponse.substr(start, end - start);
            if (!tool.empty()) {
                tools.push_back(tool);
            }
            start = end + separator.length();
        }
        std::string lastTool = toolsResponse.substr(start);
        if (!lastTool.empty()) {
            tools.push_back(lastTool);
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

            if (toolDoc.HasMember("parameters") && toolDoc["parameters"].IsObject()) {
                rapidjson::StringBuffer sb;
                rapidjson::Writer<rapidjson::StringBuffer> toolWriter(sb);
                toolDoc["parameters"].Accept(toolWriter);
                toolCall.arguments = sb.GetString();
            } else {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid parameters object");
                continue;
            }
            toolCall.id = generateRandomId();  // Generate a random ID for the tool call
            parsedOutput.toolCalls.push_back(toolCall);
        }
    }
}

void Llama3ToolParser::startNextToolCall() {
    lastJson.Clear();
    jsonBuilder.clear();
    toolCallIndex++;
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
}

static inline bool jsonHasArgumentsOrParameters(const rapidjson::Document& json) {
    return json.HasMember("arguments") || json.HasMember("parameters");
}

static inline void ensureArgumentsInJson(rapidjson::Document& json) {
    if (json.HasMember("parameters")) {
        // change key to "arguments"
        json.AddMember("arguments", json["parameters"], json.GetAllocator());
        json.RemoveMember("parameters");
    }
}

std::optional<rapidjson::Document> Llama3ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_INFO("CHUUNK: {}", chunk);

    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Llama3ToolParser");
        return std::nullopt;
    }

    // <|python_tag|> appears
    if (chunk.find(parsingStartTag) != std::string::npos) {
        this->startNextToolCall();
        return std::nullopt;  // ignoring the special tag
    }

    // -1 means not started, we need to start it
    if (toolCallIndex < 0) {
        this->startNextToolCall();
    }

    // Cases to handle:
    // <|python_tag|>{ ... parameters ... } ; { ... parameters ... }
    // <|python_tag|>{ ... arguments ... } ; { ... arguments ... }
    // { ... parameters ... } ; { ... parameters ... }
    // { ... arguments ... } ; { ... arguments ... }

    bool isCurrentToolCallParsingFinished = false;

    // JSON already contains 'parameters'/'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
    if (jsonHasArgumentsOrParameters(lastJson)) {
        std::string modifiedChunk = chunk;
        // Escaping all double quotes in the parameters/arguments string
        for (size_t pos = 0; (pos = modifiedChunk.find("\"", pos)) != std::string::npos; pos += 2) {
            modifiedChunk.insert(pos, "\\");
        }

        // Handle the case when we are starting to collect parameters/arguments.
        // Force parameters/arguments string type and fill first element of the delay array.
        if (argumentsDelayWindow[0].empty()) {
            // If we are starting to collect parameters/arguments, we add opening quote before the first non-whitespace character
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

        // If this is end of streaming, we need to manually add closing quote "
        // We need to place it right before last closing brace
        if (finishReason == ov::genai::GenerationFinishReason::STOP) {
            isCurrentToolCallParsingFinished = true;
            size_t lastClosingBrace = modifiedChunk.find_last_of('}');
            if (lastClosingBrace != std::string::npos) {
                modifiedChunk.insert(lastClosingBrace, "\"");
                argumentsDelayWindow[0] += modifiedChunk;
            }
            // If this is end of one of the tool calls "in the middle" (; has been found), we need to manually add closing quote "
            // We need to place it right before last closing brace
        } else if (modifiedChunk.find(separator) != std::string::npos) {
            isCurrentToolCallParsingFinished = true;
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
            // Push delayed chunk to the JSON builder if we are processing parameters
            newJson = jsonBuilder.add(argumentsDelayWindow[0]);
        } else {
            // Otherwise just push the current chunk
            newJson = jsonBuilder.add(chunk);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call chunk partial parse failed: {}", e.what());
        // Throwing an error since at this point the JSON is broken and next chunks will not make it right.
        throw std::runtime_error("Generated tool call structure is not valid");  // re-throw
    }

    rapidjson::Document doc;
    // Case 1: 'parameters'/'arguments' has just appeared in the current chunk. If so, we return first delta.
    if (jsonHasArgumentsOrParameters(newJson) && !jsonHasArgumentsOrParameters(lastJson)) {
        std::string functionName;
        ensureArgumentsInJson(newJson);
        if (lastJson.HasMember("name") && lastJson["name"].IsString()) {
            functionName = lastJson["name"].GetString();
        } else if (newJson.HasMember("name") && newJson["name"].IsString()) {
            // We received big chunk with both full function name and parameters, so we get function name from the new JSON
            functionName = newJson["name"].GetString();
        } else {
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Tool call name has not been generated and parameters already started");
            throw std::runtime_error("Tool call name is missing in generated output");
        }
        // Wrap first delta in {"tool_calls":[{"id":<id>,"type":"function","index":<toolCallIndex>,"function":{"name": <functionName>}}]}
        doc = wrapFirstDelta(functionName, toolCallIndex);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        return doc;
        // Case 2: 'parameters' already exists in the last JSON, we compute delta and return it.
    } else if (lastJson.HasMember("arguments") || lastJson.HasMember("parameters")) {
        ensureArgumentsInJson(newJson);
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
        if (isCurrentToolCallParsingFinished) {
            this->startNextToolCall();
        }
        return doc;
        // Case 3: No 'parameters' exists or just appeared, so we keep building up until we have complete function name
    } else {
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
    }
    return std::nullopt;
}

}  // namespace ovms
