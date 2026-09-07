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
#include "../qwen3/reasoning_parser.hpp"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ovms {
// LFM2.5 reasoning uses the same <think>/<\think> grammar as Qwen3 but both delimiters
// are registered special tokens (not regular vocabulary), so tokenIdStartTags is set and
// needsSpecialTokens/defaultDecodingWithSpecialTokens are both true.
class Lfm25ReasoningParser : public Qwen3ReasoningParser {
public:
    Lfm25ReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<think>"};
        cfg.tokenIdStartTags = {"<think>"};
        cfg.endTag = "</think>";
        cfg.needsSpecialTokens = true;
        cfg.defaultDecodingWithSpecialTokens = true;
        return cfg;
    }

    explicit Lfm25ReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        Qwen3ReasoningParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}
};
}  // namespace ovms
