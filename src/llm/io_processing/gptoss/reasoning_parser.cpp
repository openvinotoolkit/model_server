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

#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "reasoning_parser.hpp"
#include "harmony.hpp"

namespace ovms {

std::optional<Delta> GptOssReasoningParser::parseChunk(const std::string& newChunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Processing Chunk [{}]", newChunk);

    if (newChunk.empty()) {
        return std::nullopt;
    }

    std::string chunk = newChunk;

    StreamState lastState = state;

    if (startsWith(chunk, parsingConfig.startTags[0])) {
        // Final content
        state = StreamState::READING_REASONING;
        chunk = chunk.substr(parsingConfig.startTags[0].size());
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
    } else if (endsWith(chunk, parsingConfig.endTag)) {
        // End
        state = StreamState::UNKNOWN;
        chunk = chunk.substr(0, chunk.size() - parsingConfig.endTag.size());
    } else if (endsWith(chunk, "<|return|>")) {
        // End
        state = StreamState::UNKNOWN;
        chunk = chunk.substr(0, chunk.size() - std::strlen("<|return|>"));
    }

    if (chunk.size() == 0)
        return std::nullopt;

    switch (lastState) {
    case StreamState::READING_REASONING:
    case StreamState::READING_CONTENT: {
        if (state == StreamState::READING_REASONING) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Sending Reasoning [{}]", chunk);
            return ReasoningDelta{chunk};
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Sending Content [{}]", chunk);
            return ContentDelta{chunk};
        }
    }
    case StreamState::UNKNOWN:
        break;
    }

    return std::nullopt;
}
}  // namespace ovms
