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
#pragma once

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "../../logging.hpp"
#include "base_response_parser.hpp"
#include "utils.hpp"

namespace ovms {
class Qwen3ResponseParser : public BaseResponseParser {
protected:
    // Tool calls are wrapped in <tool_call> and </tool_call> tags
    std::string toolCallStartTag = "<tool_call>";
    int64_t toolCallStartTokenId = 151657;  // This is the token ID for <tool_call> in Qwen3 tokenizer
    std::string toolCallEndTag = "</tool_call>";
    int64_t toolCallEndTokenId = 151658;  // This is the token ID for </tool_call> in Qwen3 tokenizer

    std::string reasoningStartTag = "<think>";
    int64_t reasoningStartTokenId = 151667;  // This is the token ID for <think> in Qwen3 tokenizer
    std::string reasoningEndTag = "</think>";
    int64_t reasoningEndTokenId = 151668;  // This is the token ID for </think> in Qwen3 tokenizer

public:
    Qwen3ResponseParser() = delete;
    explicit Qwen3ResponseParser(ov::genai::Tokenizer& tokenizer) :
        BaseResponseParser(tokenizer) {}

    ParsedResponse parse(const std::vector<int64_t>& generatedTokens) override {
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

        if (contentStartIt != generatedTokens.end() && contentEndIt != generatedTokens.end() && contentStartIt < contentEndIt) {
            parsedResponse.content = tokenizer.decode(std::vector<int64_t>(contentStartIt, contentEndIt));
        } else {
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
};
}  // namespace ovms
