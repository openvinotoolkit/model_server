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

namespace ovms {
void GptOssReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    openai::Harmony harmony(tokenizer, generatedTokens);
    if (!harmony.parse()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Harmony parsing failed");
        return;
    }

    parsedOutput.content = harmony.getContent();
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unary | GPT Content | [{}]", parsedOutput.content);
    parsedOutput.reasoning = harmony.getReasoning();
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unary | GPT Reasoning | [{}]", parsedOutput.reasoning);
}

std::optional<rapidjson::Document> GptOssReasoningParser::parseChunk(const std::string& newChunk, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Processing Chunk [{}]", newChunk);

    if (newChunk.empty()) {
        return std::nullopt;
    }

    std::string chunk = newChunk;

    StreamState lastState = state;

    if (startsWith(chunk, getParsingStartTag())) {
        // Final content
        state = StreamState::READING_REASONING;
        chunk = chunk.substr(getParsingStartTag().size());
    } else if (startsWith(chunk, "<|start|>assistant<|channel|>final<|message|>")) {
        // Final content
        state = StreamState::READING_CONTENT;
        chunk = chunk.substr(std::strlen("<|start|>assistant<|channel|>final<|message|>"));
    } else if (startsWith(chunk, "<|channel|>final<|message|>")) {
        // Final content
        state = StreamState::READING_CONTENT;
        chunk = chunk.substr(std::strlen("<|channel|>final<|message|>"));
    } else if (startsWith(chunk, "<|channel|>commentary<|message|>")) {
        // Preamble
        state = StreamState::READING_CONTENT;
        chunk = chunk.substr(std::strlen("<|channel|>commentary<|message|>"));
    } else if (endsWith(chunk, getParsingEndTag())) {
        // End
        state = StreamState::UNKNOWN;
        chunk = chunk.substr(0, chunk.size() - getParsingEndTag().size());
    }

    if (chunk.size() == 0)
        return std::nullopt;

    switch (lastState) {
    case StreamState::READING_REASONING:
    case StreamState::READING_CONTENT: {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("delta");
        writer.StartObject();
        if (state == StreamState::READING_REASONING)
            writer.String("reasoning_content");
        else
            writer.String("content");
        writer.String(chunk.c_str());
        writer.EndObject();
        writer.EndObject();
        rapidjson::Document doc;
        doc.Parse(buffer.GetString());

        if (state == StreamState::READING_REASONING)
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Sending Reasoning [{}]", chunk);
        else
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Sending Content [{}]", chunk);
        return doc;
    }
    case StreamState::UNKNOWN:
        break;
    }

    return std::nullopt;
}
}  // namespace ovms
