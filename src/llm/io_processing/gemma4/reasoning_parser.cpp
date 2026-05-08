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

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {
void Gemma4ReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::string contentWithSpecialTokens = tokenizer.decode(generatedTokens, ov::genai::skip_special_tokens(false));
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsing reasoning with Gemma4ReasoningParser. Content with special tokens: {}", contentWithSpecialTokens);
    std::string startReasoningTag = getParsingStartTags()[0];
    std::string endReasoningTag = getParsingEndTag();
    size_t startPos = contentWithSpecialTokens.find(startReasoningTag);
    size_t endPos = contentWithSpecialTokens.find(endReasoningTag);

    if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
        size_t reasoningStart = startPos + startReasoningTag.length();
        std::string reasoningText = contentWithSpecialTokens.substr(reasoningStart, endPos - reasoningStart);
        parsedOutput.reasoning = reasoningText;
        // Remove reasoning from content
        std::string contentWithoutReasoning = contentWithSpecialTokens.substr(0, startPos) + contentWithSpecialTokens.substr(endPos + endReasoningTag.length());
        parsedOutput.content = contentWithoutReasoning;
    }
}
}  // namespace ovms
