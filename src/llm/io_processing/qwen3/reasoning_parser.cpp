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
void Qwen3ReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::string startReasoningTag = getParsingStartTag();
    std::string endReasoningTag = getParsingEndTag();
    size_t startPos = parsedOutput.content.find(startReasoningTag);
    size_t endPos = parsedOutput.content.find(endReasoningTag);

    if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
        // Extract reasoning between <think> and </think>
        size_t reasoningStart = startPos + startReasoningTag.length();
        std::string reasoningText = parsedOutput.content.substr(reasoningStart, endPos - reasoningStart);
        parsedOutput.reasoning = reasoningText;
        // Remove reasoning from content
        parsedOutput.content.erase(startPos, endPos - startPos + endReasoningTag.length());
    }
}

std::optional<rapidjson::Document> Qwen3ReasoningParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
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
