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

#include "../../logging.hpp"
#include "phi4_response_parser.hpp"
#include "utils.hpp"

namespace ovms {

ParsedResponse Phi4ResponseParser::parse(const std::vector<int64_t>& generatedTokens) {
    ParsedResponse parsedResponse;
    std::vector<std::string> tools;

    // Phi4 with vLLM template produces tool calls in the format:
    // functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
    std::string decoded = tokenizer.decode(generatedTokens);
    std::string toolsStartString = "functools";
    size_t toolsStartPos = decoded.find(toolsStartString);
    if (toolsStartPos != std::string::npos) {
        // Extract the content before the tools part
        parsedResponse.content = decoded.substr(0, toolsStartPos);
        // Extract the tools part, assuming it's all the remaining content after "functools"
        std::string toolsString = decoded.substr(toolsStartPos + toolsStartString.length());
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
                parsedResponse.toolCalls.push_back(toolCall);
            }
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse functools content or extract tools array");
            parsedResponse.content = decoded;  // If parsing fails, return the whole decoded content
        }
    } else {
        parsedResponse.content = decoded;
    }
    return parsedResponse;
}
ov::genai::StructuredOutputConfig Phi4ResponseParser::prepareStructuredOutputConfig(const std::map<std::string, std::string>& toolNameSchemaMap) {
    ov::genai::StructuralTagsConfig structuralTagsConfig;
    std::string beginOfToolsString = "functools";
    structuralTagsConfig.triggers.push_back(beginOfToolsString);
    ov::genai::StructuralTagItem tagItem;
    tagItem.begin = beginOfToolsString;

    // Build the "anyOf" array for each tool
    std::string anyOfArray = "[";
    bool first = true;
    std::cout << "tools size: " << toolNameSchemaMap.size() << std::endl;
    for (const auto& [toolName, toolSchema] : toolNameSchemaMap) {
        if (!first) {
            anyOfArray += ",";
        }
        first = false;
        anyOfArray += R"({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [")" + toolName + R"("]
                },
                "arguments": )" + toolSchema + R"(
            },
            "required": [
                "name",
                "arguments"
            ]
        })";
    }
    anyOfArray += "]";

    tagItem.schema = R"({
        "type": "array",
        "items": {
            "anyOf": )" + anyOfArray + R"(
        }
    })";

    std::cout << "Created schema: " << tagItem.schema << std::endl;
    structuralTagsConfig.structural_tags.push_back(tagItem);

    ov::genai::StructuredOutputConfig structuredOutputConfig;
    structuredOutputConfig.structural_tags_config = structuralTagsConfig;
    return structuredOutputConfig;
}
}  // namespace ovms
