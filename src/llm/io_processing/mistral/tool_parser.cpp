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

void MistralToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;

    if (parsedOutput.content.empty() || generatedTokens.size() <= 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No content to parse for tool calls");
        return;
    }

    // In regular parsing, the parser will consume entire model output only if the first generated token is the beginning of tools token.
    // In immediate parsing, the parser will consume entire model output regardless of the first token.
    if (generatedTokens[0] != this->botTokenId && !immediateParsingEnabled) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Begin of tools token has not been found in the model output. Exiting parser.");
        return;
    }

    rapidjson::Document toolsDoc;
    toolsDoc.Parse(parsedOutput.content.c_str());

    if (!toolsDoc.HasParseError() && toolsDoc.IsArray()) {
        for (auto& toolVal : toolsDoc.GetArray()) {
            if (!toolVal.IsObject()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call is not a valid JSON object");
                continue;
            }
            ToolCall toolCall;
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
            toolCall.id = generateRandomId();  // Generate a random ID for the tool call
            parsedOutput.toolCalls.push_back(toolCall);
        }
        parsedOutput.content.clear();
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse functools content or extract tools array");
    }
}

std::optional<rapidjson::Document> MistralToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "MistralToolParser::parseChunk is not implemented");
    return std::nullopt;
}
}  // namespace ovms
