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
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Phi4OutputParser::parseChunk is not implemented");
    return std::nullopt;
}
}  // namespace ovms
