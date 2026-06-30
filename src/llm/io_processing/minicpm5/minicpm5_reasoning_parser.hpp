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

#include "src/llm/io_processing/qwen3/reasoning_parser.hpp"

namespace ovms {

// MiniCPM5 uses the same <think>...</think> reasoning tags as Qwen3, so the reasoning logic is
// reused verbatim. The only difference is that the MiniCPM5 tool parser requires the stream to
// retain special tokens (its tool tags are special tokens); the framework requires the paired
// reasoning parser to agree on requiresStreamingWithSpecialTokens(), so this subclass returns
// true. The <think> tags themselves are ordinary (non-special) tokens, so the inherited
// text-based reasoning parsing is unaffected by preserving special tokens.
class Minicpm5ReasoningParser : public Qwen3ReasoningParser {
public:
    Minicpm5ReasoningParser() = delete;
    explicit Minicpm5ReasoningParser(ov::genai::Tokenizer& tokenizer) :
        Qwen3ReasoningParser(tokenizer) {}

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};

}  // namespace ovms
