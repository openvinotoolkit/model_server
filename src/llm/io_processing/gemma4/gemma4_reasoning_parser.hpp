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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../base_output_parser.hpp"

namespace ovms {

class Gemma4ReasoningParser : public BaseOutputParser {
    bool phaseEntryTagConsumed{false};

public:
    Gemma4ReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<|channel>thought\n"};
        // <|channel> is a single special token. The generic streamer can hold it
        // until the following `thought\n` role label completes the semantic opener.
        cfg.tokenIdStartTags = {"<|channel>"};
        cfg.endTag = "<channel|>";
        cfg.needsSpecialTokens = true;
        // Google's canonical Gemma4 tool sequence explicitly closes the thought
        // channel with <channel|> before <|tool_call>. Keep tool-start takeover as
        // a tolerant recovery boundary only: if malformed/edge output or streaming
        // state presents a complete native tool opener while reasoning still owns
        // the stream, preserve the reasoning prefix and hand the opener intact to
        // the tool parser instead of swallowing it as reasoning.
        cfg.toolStartTerminatesReasoning = true;
        return cfg;
    }

    explicit Gemma4ReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override {
        phaseEntryTagConsumed = false;
    }

    std::optional<Delta> parseChunk(const std::string& chunk,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;
};

}  // namespace ovms
