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
#include "harmony.hpp"
#include "../utils.hpp"

#include "harmony.hpp"
namespace ovms {
void GptReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    SPDLOG_INFO("Reasoning parsing with GPT format");

    openai::Harmony harmony(tokenizer, generatedTokens);
    if (!harmony.parse()) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Harmony parsing failed");
    } else {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed with harmony");
    }

    parsedOutput.content = harmony.getContent();  // what if someone has only tool parser and no reasoning parsers?
    parsedOutput.reasoning = harmony.getReasoning();
}

std::optional<rapidjson::Document> GptReasoningParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_INFO("REASONING CHUNK [{}]", chunk);
    
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for GptReasoningParser");
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
