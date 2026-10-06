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

#include <algorithm>
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

    const std::size_t startPos = chunk.find(parsingConfig.startTags[0]);
    if (startPos != std::string::npos) {
        state = StreamState::READING_REASONING;
        chunk = chunk.substr(startPos + parsingConfig.startTags[0].size());
    }

    const std::size_t endPos = chunk.find(parsingConfig.endTag);
    const std::size_t returnPos = chunk.find("<|return|>");
    const std::size_t closingPos = std::min(endPos, returnPos);
    if (closingPos != std::string::npos) {
        chunk = chunk.substr(0, closingPos);
    }

    if (state == StreamState::READING_REASONING && !chunk.empty()) {
        if (closingPos != std::string::npos) {
            state = StreamState::UNKNOWN;
        }
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Reason | Sending Reasoning [{}]", chunk);
        return ReasoningDelta{chunk};
    }

    if (closingPos != std::string::npos) {
        state = StreamState::UNKNOWN;
    }
    return std::nullopt;
}
}  // namespace ovms
