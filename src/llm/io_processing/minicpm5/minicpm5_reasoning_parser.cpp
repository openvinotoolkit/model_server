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
#include <algorithm>
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "minicpm5_reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {

void Minicpm5ReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    auto endIt = std::find(generatedTokens.begin(), generatedTokens.end(), thinkEndTokenId);

    // Implicit-start mode: the chat template already emitted the <think> token as the prompt
    // suffix, so generatedTokens (the model's own continuation) begins inside the reasoning
    // segment and will not contain thinkStartTokenId. Everything up to the first </think> is
    // reasoning, even if unexpected extra thinkStartTokenId tokens appear in that span.
    if (implicitStart) {
        if (endIt != generatedTokens.end()) {
            size_t endPos = std::distance(generatedTokens.begin(), endIt);
            parsedOutput.reasoning = tokenizer.decode(
                std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + endPos),
                ov::genai::skip_special_tokens(true));
            parsedOutput.content = tokenizer.decode(
                std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()),
                ov::genai::skip_special_tokens(true));
        } else {
            parsedOutput.reasoning = tokenizer.decode(generatedTokens, ov::genai::skip_special_tokens(true));
            parsedOutput.content.clear();
        }
        return;
    }

    auto startIt = std::find(generatedTokens.begin(), generatedTokens.end(), thinkStartTokenId);
    if (startIt == generatedTokens.end() || endIt == generatedTokens.end() || startIt >= endIt) {
        // No well-formed <think>...</think> segment in the raw tokens; leave parsedOutput.content
        // (already decoded by the caller) untouched -- there is no reasoning to extract.
        return;
    }

    size_t startPos = std::distance(generatedTokens.begin(), startIt);
    size_t endPos = std::distance(generatedTokens.begin(), endIt);

    parsedOutput.reasoning = tokenizer.decode(
        std::vector<int64_t>(generatedTokens.begin() + startPos + 1, generatedTokens.begin() + endPos),
        ov::genai::skip_special_tokens(true));

    // Content is everything outside the [start, end] token span: before <think> and after
    // </think>, concatenated. Decoding the two spans separately (rather than erasing a
    // substring from the already-decoded text) avoids any ambiguity from special-token
    // markers that may appear in the decoded text around the reasoning segment.
    std::string beforeReasoning = tokenizer.decode(
        std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + startPos),
        ov::genai::skip_special_tokens(true));
    std::string afterReasoning = tokenizer.decode(
        std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()),
        ov::genai::skip_special_tokens(true));
    parsedOutput.content = beforeReasoning + afterReasoning;
}

}  // namespace ovms
