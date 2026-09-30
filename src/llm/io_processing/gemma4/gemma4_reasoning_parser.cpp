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
#include <cctype>
#include <string>
#include <vector>

#include "../../../logging.hpp"
#include "gemma4_reasoning_parser.hpp"

namespace ovms {
std::optional<Delta> Gemma4ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty() && finishReason == ov::genai::GenerationFinishReason::NONE) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Gemma4ReasoningParser");
        return std::nullopt;
    }

    std::string text = chunk;

    if (phase == Phase::AwaitingOpener) {
        if (!isImplicitStart()) {
            const std::string& opener = parsingConfig.startTags[0];  // "<|channel>"
            const size_t openerPos = text.find(opener);
            if (openerPos != std::string::npos) {
                text = text.substr(openerPos + opener.size());
            }
        }
        // Implicit start means the prompt already supplied the opener, not the keyword line -
        // the model may still generate its own (possibly corrupted) header, so header detection
        // must run either way.
        phase = Phase::AwaitingChannelHeader;
    }

    if (phase == Phase::AwaitingChannelHeader) {
        pendingChannelHeaderText += text;
        const size_t newlinePos = pendingChannelHeaderText.find('\n');
        const size_t endTagPos = pendingChannelHeaderText.find(parsingConfig.endTag);
        // A first line with an internal space is a real multi-word reasoning sentence, not a
        // discardable single-token keyword - it must be kept, otherwise a keyword-omitted
        // multiline body would silently lose its first line.
        bool firstLineIsMultiWord = false;
        if (newlinePos != std::string::npos) {
            const std::string firstLine = pendingChannelHeaderText.substr(0, newlinePos);
            size_t wordStart = 0;
            while (wordStart < firstLine.size() && std::isspace(static_cast<unsigned char>(firstLine[wordStart])) != 0) {
                ++wordStart;
            }
            for (size_t k = wordStart; k < firstLine.size(); ++k) {
                if (std::isspace(static_cast<unsigned char>(firstLine[k])) != 0) {
                    firstLineIsMultiWord = true;
                    break;
                }
            }
        }
        if (newlinePos != std::string::npos && !firstLineIsMultiWord &&
            (endTagPos == std::string::npos || newlinePos < endTagPos)) {
            text = pendingChannelHeaderText.substr(newlinePos + 1);
        } else if (endTagPos != std::string::npos || finishReason != ov::genai::GenerationFinishReason::NONE) {
            // No header line ever materialized (e.g. keyword missing entirely, or opener directly
            // followed by body text with no separating newline) - the whole span is reasoning body.
            text = pendingChannelHeaderText;
        } else {
            return std::nullopt;
        }
        pendingChannelHeaderText.clear();
        phase = Phase::Body;
    }

    const size_t endTagPos = text.find(parsingConfig.endTag);
    if (endTagPos != std::string::npos) {
        text = text.substr(0, endTagPos);
    }

    if (text.empty()) {
        return std::nullopt;
    }
    return ReasoningDelta{text};
}
}  // namespace ovms
