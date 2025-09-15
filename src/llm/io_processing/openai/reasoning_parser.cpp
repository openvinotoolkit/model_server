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
#include "../../../stringutils.hpp"
#include "reasoning_parser.hpp"
#include "harmony.hpp"
#include "../utils.hpp"

#include "harmony.hpp"
namespace ovms {
void GptReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    //SPDLOG_INFO("Reasoning parsing with GPT format");

    openai::Harmony harmony(tokenizer, generatedTokens);
    if (!harmony.parse()) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Harmony parsing failed");
    } else {
        //SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed with harmony");
    }

    // Loggin assuming both parser are always on
    parsedOutput.content = harmony.getContent();  // what if someone has only tool parser and no reasoning parsers?
    SPDLOG_INFO("DEBUG Unary | GPT Content | [{}]", parsedOutput.content);
    parsedOutput.reasoning = harmony.getReasoning();
    SPDLOG_INFO("DEBUG Unary | GPT Reasoning | [{}]", parsedOutput.reasoning);
}

std::optional<rapidjson::Document> GptReasoningParser::parseChunk(const std::string& c, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_INFO("DEBUG Streaming | GPT Reason | Chunk [{}]", c);
    
    if (c.empty()) {
        //SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for GptReasoningParser");
        return std::nullopt;
    }

    std::string chunk = c;

    int lastState = state;

    if (startsWith(chunk, getParsingStartTag())) {
        state = 1;
        // remove the start tag
        chunk = chunk.substr(getParsingStartTag().size());
    } else if (startsWith(chunk, "<|start|>assistant<|channel|>final<|message|>")) {
        state = 2;
        // remove the tag
        chunk = chunk.substr(std::strlen("<|start|>assistant<|channel|>final<|message|>"));
    } else if (endsWith(chunk, getParsingEndTag())) {
        state = 0;
        // remove the end tag
        chunk = chunk.substr(0, chunk.size() - getParsingEndTag().size());
    }

    if (chunk.size() == 0)
        return std::nullopt;

    switch (lastState)
    {
        case 1:
        case 2:
        {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.StartObject();
            writer.String("delta");
            writer.StartObject();
            if (state == 1)
                writer.String("reasoning_content");
            else
                writer.String("content");
            writer.String(chunk.c_str());
            writer.EndObject();
            writer.EndObject();
            rapidjson::Document doc;
            doc.Parse(buffer.GetString());

            if (state == 1)
                SPDLOG_INFO("DEBUG Streaming | GPT Reason-Think | Send [{}]", chunk);
            else
                SPDLOG_INFO("DEBUG Streaming | GPT Reason-Content | Send [{}]", chunk);
            return doc;
        }
    }

    return std::nullopt;
}
}  // namespace ovms
