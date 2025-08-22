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

    auto toolCallsStartPosition = generatedTokens.end();

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

void Llama3ToolParser::next() {
    lastJson.Clear();
    jsonBuilder.clear();
    toolCallIndex++;
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
}

std::optional<rapidjson::Document> Llama3ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    // todo: sometimes there is 'parameters' and sometimes there is 'arguments', however, 'parameters' more often

    SPDLOG_INFO("Hello: [{}]", chunk);
    
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Llama3ToolParser");
        return std::nullopt;
    }

    // <|python_tag|> appears
    if (chunk.find(parsingStartTag) != std::string::npos) {
        this->next();
        return std::nullopt;
    }

    // we start from {. therefore we need to begin
    if (toolCallIndex < 0) {
        this->next();
    }

    // <|python_tag|>{   }  ;   {    }
    // <tool_call> </tool_call><tool_call> </tool_call><tool_call> </tool_call><tool_call> </tool_call><tool_call> </tool_call>

    bool end=false;

    // JSON already contains 'parameters' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
    if (lastJson.HasMember("arguments") || lastJson.HasMember("parameters")) {
        std::string modifiedChunk = chunk;
        // Escaping double quotes in the parameters string
        for (size_t pos = 0; (pos = modifiedChunk.find("\"", pos)) != std::string::npos; pos += 2) {
            modifiedChunk.insert(pos, "\\");
        }

        // Handle the case when we are starting to collect parameters.
        // Force parameters string type and fill first element of the delay array.
        if (argumentsDelayWindow[0].empty()) {
            // If we are starting to collect parameters, we add opening quote before the first non-whitespace character
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

        if (static_cast<int>(finishReason) == 1) {
            end=true;
            size_t lastClosingBrace = modifiedChunk.find_last_of('}');
            if (lastClosingBrace != std::string::npos) {
                modifiedChunk.insert(lastClosingBrace, "\"");
                argumentsDelayWindow[0] += modifiedChunk;
            }
        }

        // If we are closing the tool call, we need to add closing quote after the last closing brace that we assume is present in the chunk processed in the last call.
        // Otherwise we just store the chunk in the second element of the delay array to be handled in the next call.
        else if (modifiedChunk.find(separator) != std::string::npos) {
            end=true;
            size_t lastClosingBrace = argumentsDelayWindow[0].find_last_of('}');
            if (lastClosingBrace != std::string::npos) {
                argumentsDelayWindow[0].insert(lastClosingBrace, "\"");
            }
            //SPDLOG_INFO("This is the end. We are closing the tool call. [{}] [{}] [{}]", argumentsDelayWindow[0], argumentsDelayWindow[1], modifiedChunk);
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
        (void)e;  // Suppress unused variable warning on Windows
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Tool call chunk partial parse failed: {}", e.what());
        // Throwing an error since at this point the JSON is broken and next chunks will not make it right.
        throw std::runtime_error("Generated tool call structure is not valid");
    }

    rapidjson::Document doc;
    // Case 1: 'parameters' has just appeared in the current chunk. If so, we return first delta.
    if ((newJson.HasMember("parameters") || newJson.HasMember("arguments")) && !(lastJson.HasMember("arguments") || lastJson.HasMember("parameters"))) {
        std::string functionName;
        if (newJson.HasMember("parameters")) {
            SPDLOG_INFO("HAD PARAMETERS, COPYING TO ARGUMENTS");
            // change key to "arguments"
            newJson.AddMember("arguments", newJson["parameters"], newJson.GetAllocator());
            newJson.RemoveMember("parameters");

        }
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
        if (newJson.HasMember("parameters")) {
            SPDLOG_INFO("HAD PARAMETERS, COPYING TO ARGUMENTS");
            // change key to "arguments"
            newJson.AddMember("arguments", newJson["parameters"], newJson.GetAllocator());
            newJson.RemoveMember("parameters");

        }
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
        if (end)
            this->next();
        return doc;
        // Case 3: No 'parameters' exists or just appeared, so we keep building up until we have complete function name
    } else {
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
    }
    return std::nullopt;
}

}  // namespace ovms
