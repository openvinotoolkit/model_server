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

void GptToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;

    if (parsedOutput.content.empty() || generatedTokens.size() <= 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No content to parse for tool calls");
        return;
    }

    
    auto gt = generatedTokens;

    // find all sequences of 200005, 12606 in generatedTokens
    // after those tokens, find 200012
    // extract to new vector, remove from original
    std::vector<int64_t> toolCallTokens;
    bool startFound = false;
    size_t startIndex = 0;
    for (size_t i = 0; i < gt.size(); i++) {
        if (!startFound && gt[i] == 200005) {
            if (i + 2 < gt.size() && gt[i + 1] == 12606 && gt[i + 2] == 815) {
                startFound = true;
                startIndex = i;
                i += 1; // skip next token
                for (; i < gt.size(); i++) {
                    if (gt[i] == 200012) {
                        auto toolStr = tokenizer.decode(std::vector<int64_t>(gt.begin() + startIndex, gt.begin() + i + 1), ov::AnyMap{ov::genai::skip_special_tokens(false)});

                            


                        SPDLOG_INFO("HEADER: [{}]", toolStr);

                        std::string marker = "to=functions.";
                        size_t start = toolStr.find(marker);
                        std::string functionName;
                        if (start != std::string::npos) {
                            start += marker.length(); // move past "to=functions."
                            size_t end = toolStr.find(' ', start); // find next space
                            std::string result = toolStr.substr(start, end - start);
                            SPDLOG_INFO("Extracted function name: [{}]", result);
                            functionName = result;
                            
                        } else {
                            break;
                        }

                        marker = "<|message|>";
                        start = toolStr.find(marker);
                        if (start == std::string::npos) {
                            break;
                        }

                        start += marker.length(); // Move past the marker
                        // Find the position of <|call|>
                        std::size_t end = toolStr.find("<|call|>", start);
                        if (end == std::string::npos) {
                            break;
                        }

                        // Extract substring
                        std::string functionArgs = toolStr.substr(start, end - start);
                        SPDLOG_INFO("Content: [{}]", functionArgs);

                        ToolCall toolCall;
                        toolCall.name = functionName;
                        toolCall.arguments = functionArgs;
                        parsedOutput.toolCalls.push_back(toolCall);
                        
                        startFound = false;
                        parsedOutput.content = "";
                        break;  // exit inner loop
                    }
                }
            }
        }
    }

    SPDLOG_INFO("DDDDD: [{}]", toolCallTokens);
    SPDLOG_INFO("EEEEE: [{}]", tokenizer.decode(toolCallTokens));
    //parsedOutput.tool_calls = tokenizer.decode(toolCallTokens);

    SPDLOG_INFO("GTRemaining: [{}]", gt);
    SPDLOG_INFO("Remaining: [{}]", tokenizer.decode(gt));
}

std::optional<rapidjson::Document> GptToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "GptToolParser::parseChunk is not implemented");
    return std::nullopt;
}
}  // namespace ovms
