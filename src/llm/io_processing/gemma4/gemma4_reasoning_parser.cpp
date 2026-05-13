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

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "gemma4_reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {
void Gemma4ReasoningParser::skipToken(const std::vector<int64_t>& generatedTokens, size_t& pos, int64_t tokenId) {
    if (pos < generatedTokens.size() && generatedTokens[pos] == tokenId) {
        pos++;
    }
}

void Gemma4ReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    auto startPos = std::string::npos;
    auto endPos = std::string::npos;

    auto startIt = std::find(generatedTokens.begin(), generatedTokens.end(), channelStartTokenId);
    auto endIt = std::find(generatedTokens.begin(), generatedTokens.end(), channelEndTokenId);

    if (startIt != generatedTokens.end() && endIt != generatedTokens.end() && startIt < endIt) {
        startPos = std::distance(generatedTokens.begin(), startIt);
        endPos = std::distance(generatedTokens.begin(), endIt);
    }

    if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
        skipToken(generatedTokens, startPos, channelStartTokenId);
        std::string reasoningText = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + startPos, generatedTokens.begin() + endPos), ov::genai::skip_special_tokens(true));
        if (reasoningText.find(reasoningStrIndicator) == 0) {
            reasoningText = reasoningText.substr(reasoningStrIndicator.size());
        }
        parsedOutput.reasoning = reasoningText;
        // Remove reasoning from content
        std::string contentWithoutReasoning = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()), ov::genai::skip_special_tokens(true));  // content MUST never appear before reasoning
        parsedOutput.content = contentWithoutReasoning;
    }
}
std::optional<rapidjson::Document> Gemma4ReasoningParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Gemma4ReasoningParser");
        return std::nullopt;
    }

    if (chunk.find(getParsingStartTags()[0]) != std::string::npos || chunk.find(getParsingEndTag()) != std::string::npos) {
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
