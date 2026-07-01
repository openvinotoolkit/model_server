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
#pragma once

#include <openvino/genai/tokenizer.hpp>
#include <vector>
#include <string>

#include "../qwen3/reasoning_parser.hpp"

namespace ovms {
class Gemma4ReasoningParser : public Qwen3ReasoningParser {
protected:
    const int64_t channelStartTokenId = 100;  // <|channel>
    const int64_t channelEndTokenId = 101;    // <channel|>

    const std::string reasoningStrIndicator = "thought\n";
    const std::string parsingStartTag = "<|channel>" + reasoningStrIndicator;
    const std::string parsingEndTag = "<channel|>";

    void skipToken(const std::vector<int64_t>& generatedTokens, size_t& pos, int64_t tokenId);

public:
    Gemma4ReasoningParser() = delete;
    explicit Gemma4ReasoningParser(ov::genai::Tokenizer& tokenizer,
                                    std::optional<ParsingConfig> configOverride = std::nullopt) :
        Qwen3ReasoningParser(tokenizer, [&]() -> std::optional<ParsingConfig> {
            if (configOverride.has_value()) return configOverride;
            ParsingConfig cfg;
            cfg.startTags                = {"<|channel>thought\n"};
            cfg.specialTokenStartTags    = {"<|channel>"};
            cfg.endTag                   = "<channel|>";
            cfg.alwaysNeedsSpecialTokens = true;
            return cfg;
        }()) {
        resolveSpecialTokenIds();
    }
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
