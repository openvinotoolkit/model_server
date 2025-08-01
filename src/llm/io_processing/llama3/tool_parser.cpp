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

std::optional<rapidjson::Document> Llama3ToolParser::parseChunk(const std::string& chunk) {
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3ToolParser::parseChunk is not implemented");
    return std::nullopt;
}

}  // namespace ovms
