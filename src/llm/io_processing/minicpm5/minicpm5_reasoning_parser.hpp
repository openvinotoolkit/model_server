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
#include "src/llm/io_processing/base_output_parser.hpp"
#include <string>
#include <utility>
#include <vector>

namespace ovms {
class Minicpm5ReasoningParser : public BaseOutputParser {
public:
    static constexpr int64_t reasoningStartTokenId = 8;
    static constexpr int64_t reasoningEndTokenId = 9;

public:
    Minicpm5ReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<think>"};
        cfg.tokenIdStartTags = {"<think>"};
        cfg.endTag = "</think>";
        cfg.needsSpecialTokens = true;
        return cfg;
    }

    explicit Minicpm5ReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
