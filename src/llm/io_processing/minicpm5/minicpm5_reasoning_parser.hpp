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
#pragma once

#include <openvino/genai/tokenizer.hpp>
#include <vector>

#include "src/llm/io_processing/qwen3/reasoning_parser.hpp"

namespace ovms {

// MiniCPM5's <think> / </think> reasoning tags are dedicated special tokens in its tokenizer
// (added_tokens_decoder ids 8 and 9 respectively), not incidental text. This mirrors
// Gemma4ReasoningParser: parse() locates the reasoning segment by scanning generatedTokens
// for these token ids directly, instead of searching decoded text for the "<think>"/"</think>"
// substrings the way the inherited Qwen3ReasoningParser::parse() does.
//
// getParsingStartTags()/getParsingEndTag() (used by the streaming parseChunk() path, inherited
// from Qwen3ReasoningParser) still report the literal "<think>"/"</think>" strings, since that
// is what the streamed, special-tokens-preserved text actually contains.
//
// requiresStreamingWithSpecialTokens() is true because MiniCPM5's tool-call tags (<function>,
// <param>, etc. -- see Minicpm5ToolParser) are also special tokens, and the framework requires
// the paired tool/reasoning parser to agree on this setting.
class Minicpm5ReasoningParser : public Qwen3ReasoningParser {
protected:
    static constexpr int64_t thinkStartTokenId = 8;  // <think>
    static constexpr int64_t thinkEndTokenId = 9;    // </think>

public:
    Minicpm5ReasoningParser() = delete;
    explicit Minicpm5ReasoningParser(ov::genai::Tokenizer& tokenizer) :
        Qwen3ReasoningParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};

}  // namespace ovms
