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
    std::regex toolRegex(R"(functools\[(.*?)\])");
    std::sregex_iterator begin(decoded.begin(), decoded.end(), toolRegex);
    std::sregex_iterator end;
    size_t matchCount = std::distance(begin, end);

    if (matchCount == 0) {
        parsedResponse.content = decoded;
    } else if (matchCount == 1) {
        std::smatch match = *begin;
        // Put everything, but functools[...] part into the response content
        parsedResponse.content = decoded.substr(0, match.position()) +
                                    decoded.substr(match.position() + match.length());

        std::string toolsStr = match[1].str();
        std::string toolsJson = "{\"functools\": [" + toolsStr + "]}";  // Wrap in JSON array

        rapidjson::Document toolsDoc;
        toolsDoc.Parse(toolsJson.c_str());
        if (!toolsDoc.HasParseError() && toolsDoc.IsObject() && toolsDoc.HasMember("functools") && toolsDoc["functools"].IsArray()) {
            const rapidjson::Value& toolsArray = toolsDoc["functools"];
            for (auto& toolVal : toolsArray.GetArray()) {
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
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse toolsJson or extract tools array");
        }
    } else {
        throw std::runtime_error("Multiple 'functools[...]' matches found in the response.");
    }
    return parsedResponse;
}
}  // namespace ovms
