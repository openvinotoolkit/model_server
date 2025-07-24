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

ParsedOutput Hermes3OutputParser::parse(const std::vector<int64_t>& generatedTokens) {
    ParsedOutput parsedOutput;

    // Assuming content ends when tool calls start, so we find the first occurrence of <tool_call> after the content start
    auto contentEndIt = std::find(generatedTokens.begin(), generatedTokens.end(), toolCallStartTokenId);

    if (contentEndIt != generatedTokens.end()) {
        parsedOutput.content = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin(), contentEndIt));
    } else {
        parsedOutput.content = tokenizer.decode(generatedTokens);
    }

    // Assuming tool calls are the last part of the output
    auto it = generatedTokens.begin();
    std::vector<std::string> tools;
    while (it != generatedTokens.end()) {
        // Find the next <tool_call> tag
        auto toolCallStartIt = std::find(it, generatedTokens.end(), toolCallStartTokenId);
        if (toolCallStartIt == generatedTokens.end()) {
            break;
        }
        // Find the next </tool_call> tag after <tool_call>
        auto toolCallEndIt = std::find(toolCallStartIt + 1, generatedTokens.end(), toolCallEndTokenId);

        std::vector<int64_t> toolTokens;
        if (toolCallEndIt != generatedTokens.end()) {
            // Extract tokens between <tool_call> and </tool_call>
            toolTokens.assign(toolCallStartIt + 1, toolCallEndIt);
            it = toolCallEndIt + 1;
        } else {
            // No closing tag, take everything until the end
            toolTokens.assign(toolCallStartIt + 1, generatedTokens.end());
            it = generatedTokens.end();
        }

        std::string tool = tokenizer.decode(toolTokens);
        if (!tool.empty()) {
            tools.push_back(tool);
        }
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

std::optional<rapidjson::Document> Hermes3OutputParser::parseChunk(const std::string& chunk) {
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Hermes3OutputParser::parseChunk is not implemented");
    return std::nullopt;
}
}  // namespace ovms
