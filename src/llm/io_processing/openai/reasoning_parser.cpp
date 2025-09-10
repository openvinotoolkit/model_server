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
#include "reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {
void GptReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {

    SPDLOG_INFO("BBBBB: [{}]", parsedOutput.content);
    SPDLOG_INFO("CCCCC: [{}]", generatedTokens);
    SPDLOG_INFO("KKKK: [{}]", tokenizer.decode(generatedTokens, ov::AnyMap{ov::genai::skip_special_tokens(false)}));

    if (generatedTokens.size() == 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No generated tokens to parse for reasoning");
        return;
    }

    auto gt = generatedTokens;

    // find sequence 200005, X, 200008 in generatedTokens
    // find first 200007 after that
    // extract to new vector, remove from original
    std::vector<int64_t> reasoningTokens;
    bool startFound = false;
    size_t startIndex = 0;
    for (size_t i = 0; i < gt.size(); i++) {
        if (!startFound && gt[i] == 200005) {
            if (i + 2 < gt.size() && gt[i + 2] == 200008) {
                startFound = true;
                startIndex = i;
                int64_t channel = gt[i + 1];
                i += 2; // skip next two tokens
                for (; i < gt.size(); i++) {
                    if (gt[i] == 200007) {
                        if (channel == 35644) {
                            SPDLOG_INFO("Found reasoning channel: analysis");
                        } else if (channel == 35645) {
                            SPDLOG_INFO("Found reasoning channel: tool use");
                        } else {
                            SPDLOG_INFO("Unknown reasoning channel: {}", channel);
                        }
                        // found 200007, extract reasoning tokens
                        reasoningTokens.insert(reasoningTokens.end(), gt.begin() + startIndex + 3, gt.begin() + i);
                        gt.erase(gt.begin() + startIndex, gt.begin() + i + 3);
                        break;
                    }
                }
            }
        }
    }

    SPDLOG_INFO("DDDDD: [{}]", reasoningTokens);
    SPDLOG_INFO("EEEEE: [{}]", tokenizer.decode(reasoningTokens));
    parsedOutput.reasoning = tokenizer.decode(reasoningTokens);

    SPDLOG_INFO("GTRemaining: [{}]", gt);
    SPDLOG_INFO("Remaining: [{}]", tokenizer.decode(gt));
    parsedOutput.content = tokenizer.decode(gt);
}

std::optional<rapidjson::Document> GptReasoningParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3ReasoningParser");
        return std::nullopt;
    }

    if (chunk.find(getParsingStartTag()) != std::string::npos || chunk.find(getParsingEndTag()) != std::string::npos) {
        return std::nullopt;
    } else {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("delta");
        writer.StartObject();
        writer.String("reasoning_content");
        writer.String(chunk.c_str());
        writer.EndObject();
        writer.EndObject();
        rapidjson::Document doc;
        doc.Parse(buffer.GetString());
        return doc;
    }
    return std::nullopt;
}
}  // namespace ovms
