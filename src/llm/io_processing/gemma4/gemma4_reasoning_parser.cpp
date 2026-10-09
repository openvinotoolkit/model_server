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
#include "gemma4_reasoning_parser.hpp"

namespace ovms {

std::optional<Delta> Gemma4ReasoningParser::parseChunk(
    const std::string& chunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {
    if (chunk.empty()) {
        return std::nullopt;
    }

    std::string text = chunk;

    // The generic OutputParser normally splits at <channel|>, but keep this
    // defensive stripping for direct parser use and boundary-sharing chunks.
    const size_t endTagPos = text.rfind(parsingConfig.endTag);
    if (endTagPos != std::string::npos) {
        text = text.substr(0, endTagPos);
    }

    // Gemma4's opener is a channel marker plus a role label. Strip it exactly
    // once at phase entry. On post-tool continuation the chat template may have
    // already placed the opener in the prompt, so generated text starts directly
    // inside reasoning and there is nothing to strip.
    if (!phaseEntryTagConsumed) {
        const std::string& startTag = parsingConfig.startTags.front();
        const size_t startTagPos = text.find(startTag);
        if (startTagPos != std::string::npos) {
            text = text.substr(startTagPos + startTag.size());
        }
        phaseEntryTagConsumed = true;
    }

    if (text.empty()) {
        return std::nullopt;
    }

    return ReasoningDelta{text};
}

}  // namespace ovms
