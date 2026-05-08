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
    size_t startPos = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningTokenId) - generatedTokens.begin();
    size_t endPos = std::find(generatedTokens.begin(), generatedTokens.end(), reasoningEndTokenId) - generatedTokens.begin();

    if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
        size_t reasoningStart = startPos + 3;  // deleting "<|channel>thought\n"
        std::string reasoningText = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + reasoningStart, generatedTokens.begin() + endPos), ov::genai::skip_special_tokens(true));
        parsedOutput.reasoning = reasoningText;
        // Remove reasoning from content
        std::string contentWithoutReasoning = tokenizer.decode(std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()), ov::genai::skip_special_tokens(true));
        parsedOutput.content = contentWithoutReasoning;
    }
}
}  // namespace ovms
