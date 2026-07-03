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
            // Content must keep special tokens: MiniCPM5's tool-call tags (<function>, <param>,
            // ...) are special tokens too, and Minicpm5ToolParserImpl matches them as literal
            // substrings in decoded content, so skip_special_tokens(true) here would silently
            // erase them before the tool parser ever sees them.
            parsedOutput.content = tokenizer.decode(
                std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()),
                ov::genai::skip_special_tokens(false));
        } else {
            parsedOutput.reasoning = tokenizer.decode(generatedTokens, ov::genai::skip_special_tokens(true));
            parsedOutput.content.clear();
        }
        return;
    }

    // NOTE: endIt is the FIRST </think> token in the whole sequence. A stray/malformed
    // </think> appearing before any <think> would make startIt >= endIt below and cause a
    // later, well-formed <think>...</think> pair to be skipped entirely. This mirrors an
    // existing limitation in Gemma4ReasoningParser's identical find-first approach and is
    // considered acceptable for now, since models are not expected to emit an unpaired
    // closing tag before ever opening one.
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
    //
    // Content must keep special tokens: MiniCPM5's tool-call tags (<function>, <param>, ...)
    // are special tokens too, and Minicpm5ToolParserImpl matches them as literal substrings
    // in decoded content, so skip_special_tokens(true) here would silently erase them before
    // the tool parser ever sees them. Only the extracted reasoning text (above) is safe to
    // decode with special tokens skipped, since it is a human-facing string, not matched
    // against tags downstream.
    std::string beforeReasoning = tokenizer.decode(
        std::vector<int64_t>(generatedTokens.begin(), generatedTokens.begin() + startPos),
        ov::genai::skip_special_tokens(false));
    std::string afterReasoning = tokenizer.decode(
        std::vector<int64_t>(generatedTokens.begin() + endPos + 1, generatedTokens.end()),
        ov::genai::skip_special_tokens(false));
    parsedOutput.content = beforeReasoning + afterReasoning;
}

}  // namespace ovms
