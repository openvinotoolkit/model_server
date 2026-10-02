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

#include "src/llm/io_processing/qwen3/reasoning_parser.hpp"

namespace ovms {
// Delimiter-driven, not keyword-driven: the keyword after the opener is never validated.
class Gemma4ReasoningParser : public Qwen3ReasoningParser {
protected:
    enum class Phase {
        AwaitingOpener,
        AwaitingChannelHeader,
        Body
    };
    Phase phase = Phase::AwaitingOpener;
    std::string pendingChannelHeaderText;

public:
    Gemma4ReasoningParser() = delete;
    explicit Gemma4ReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        Qwen3ReasoningParser(tokenizer, [&]() -> std::optional<OutputParsingConfig> {
            if (configOverride.has_value())
                return configOverride;
            OutputParsingConfig cfg;
            cfg.startTags = {"<|channel>"};
            cfg.preambleStartTags = {"thought\n"};
            cfg.tokenIdStartTags = {"<|channel>"};
            cfg.endTag = "<channel|>";
            cfg.needsSpecialTokens = true;
            return cfg;
        }()) {
        resolveSpecialTokenIds();
    }

    void resetState() override {
        phase = Phase::AwaitingOpener;
        pendingChannelHeaderText.clear();
    }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
