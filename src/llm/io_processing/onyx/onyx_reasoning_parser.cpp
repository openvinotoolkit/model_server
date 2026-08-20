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

#include "src/logging.hpp"
#include "onyx_reasoning_parser.hpp"

namespace ovms {

std::optional<Delta> OnyxReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason /*finishReason*/) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for OnyxReasoningParser");
        return std::nullopt;
    }

    // Buffer content until the <|message|> separator has been fully consumed.
    // The tag arrives as individual tokens so we must accumulate to detect the full string.
    if (!headerConsumed) {
        headerBuffer += chunk;
        size_t msgPos = headerBuffer.find(messageTag);
        if (msgPos == std::string::npos) {
            return std::nullopt;
        }
        headerConsumed = true;
        std::string afterMsg = headerBuffer.substr(msgPos + messageTag.size());
        headerBuffer.clear();
        if (afterMsg.empty()) {
            return std::nullopt;
        }
        // Fall through with the content that follows the separator.
        return parseChunk(afterMsg, {}, ov::genai::GenerationFinishReason::NONE);
    }

    // Strip the end tag if it arrives bundled with the last reasoning text.
    std::string text = chunk;
    const size_t endTagPos = text.rfind(continuationEndTag);
    if (endTagPos != std::string::npos) {
        text = text.substr(0, endTagPos);
    }
    if (text.empty()) {
        return std::nullopt;
    }

    return ReasoningDelta{text};
}
}  // namespace ovms
