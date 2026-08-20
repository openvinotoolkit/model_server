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
#include "reasoning_parser.hpp"

namespace ovms {

std::optional<Delta> Qwen3ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3ReasoningParser");
        return std::nullopt;
    }

    // Strip the end tag and keep only the text that precedes it.
    // This handles the case where the end tag token is decoded in the same
    // streamer flush as preceding reasoning text (FOUND_INCOMPLETE hold-back
    // accumulates e.g. "...ing</think>" in the cache).
    std::string text = chunk;
    const std::string& endTag = parsingConfig.endTag;
    const size_t endTagPos = text.rfind(endTag);
    if (endTagPos != std::string::npos) {
        text = text.substr(0, endTagPos);
    }

    // On the very first call, consume the start tag if it begins the text
    // (explicit phase-entry case) or mark it consumed immediately if no start
    // tag is present (implicit reasoning start — the prompt already ended with
    // <think> so the model never emits it again).
    // After the first call, any <think> that appears in the stream is literal
    // reasoning content produced by the model and is emitted as-is.
    if (!phaseEntryTagConsumed_) {
        const std::string& startTag = parsingConfig.startTags[0];
        const size_t startTagPos = text.find(startTag);
        if (startTagPos != std::string::npos) {
            text = text.substr(startTagPos + startTag.size());
        }
        phaseEntryTagConsumed_ = true;
    }

    if (text.empty()) {
        return std::nullopt;
    }

    return ReasoningDelta{text};
}
}  // namespace ovms
