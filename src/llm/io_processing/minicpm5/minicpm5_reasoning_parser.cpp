//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "minicpm5_reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {
void Minicpm5ReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    auto startReasoningIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningStartTokenId);
    auto endReasoningIt = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningEndTokenId);

    if ((startReasoningIt == generatedTokens.end() && endReasoningIt == generatedTokens.end())) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Minicpm5ReasoningParser: Reasoning start or end token not found in the generated tokens. Start token found: {}, End token found: {}, Start position: {}, End position: {}",
            startReasoningIt != generatedTokens.end(), endReasoningIt != generatedTokens.end(), std::distance(generatedTokens.begin(), startReasoningIt), std::distance(generatedTokens.begin(), endReasoningIt));
        return;
    }

    auto startPos = 0;
    if (startReasoningIt != generatedTokens.end()) {
        startPos = std::distance(generatedTokens.begin(), startReasoningIt) + 1;
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Minicpm5ReasoningParser: Reasoning start token not found in the generated tokens. Start position: {}", startPos);
    }
    auto endPos = std::distance(generatedTokens.begin(), endReasoningIt);

    std::string reasoningContent = tokenizer.decode(std::vector<int64_t>(startPos + generatedTokens.begin(), endPos + generatedTokens.begin()), ov::genai::skip_special_tokens(true));

    parsedOutput.reasoning = reasoningContent;

    if (endReasoningIt != generatedTokens.end()) {
        endPos += 1; 
    }
    
    std::string contentWithoutReasoning = tokenizer.decode(std::vector<int64_t>(endPos + generatedTokens.begin(), generatedTokens.end()), ov::genai::skip_special_tokens(true));
    parsedOutput.content = contentWithoutReasoning;
}

std::optional<rapidjson::Document> Minicpm5ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) {
    if (tokens.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty tokens for Minicpm5ReasoningParser");
        return std::nullopt;
    }

    if (std::find(tokens.begin(), tokens.end(), reasoningStartTokenId) != tokens.end() ||
        std::find(tokens.begin(), tokens.end(), reasoningEndTokenId) != tokens.end()) {
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
}
}  // namespace ovms
