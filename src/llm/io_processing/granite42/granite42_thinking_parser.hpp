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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "../base_output_parser.hpp"

namespace ovms {

// Faithful OVMS port of IBM's granite_thinking_parser.py reasoning behavior.
// The constructor flag is derived from the effective chat_template_kwargs after
// OpenAI reasoning-effort processing, keeping RapidJSON request handling out of
// the parser itself.
class Granite42ThinkingParser final : public BaseOutputParser {
    bool promoteReasoningToContent_ = false;
    bool phaseEntryTagConsumed_ = false;
    // Streaming keeps a fused <think> marker in the reasoning delta. Unary
    // extraction removes the first generated marker with str.partition();
    // retain enough private state to reproduce that later without changing
    // ParsedOutput or the public parser API.
    bool generatedStartTagSeen_ = false;
    // True when the first generated start marker was consumed as structural
    // input (a standalone marker, or a same-delta start/end pair). In that
    // case unary finalization must discard only the pre-marker reasoning
    // prefix and must not search for a later literal <think> marker.
    bool generatedStartTagConsumed_ = false;
    // Streaming suppresses every standalone start-token delta. Unary
    // str.partition() removes only the first generated marker, so retain the
    // later standalone-marker positions for restoration during finalization.
    bool firstGeneratedStartTagWasStandalone_ = false;
    std::vector<size_t> standaloneStartTagEventReasoningOrdinals_;
    std::optional<size_t> generatedStartTagEventReasoningOrdinal_;
    size_t emittedReasoningDeltaCount_ = 0;

public:
    Granite42ThinkingParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<think>"};
        // Granite 4.2 declares <think> as a single added (non-special) token.
        // Register it for zero-delay phase entry while retaining text matching
        // for ordinary streamed chunks and test tokenizers.
        cfg.tokenIdStartTags = {"<think>"};
        cfg.endTag = "</think>";
        // Granite inherits DeepSeek-R1's one-way grammar: the first closing
        // marker ends reasoning and all later bytes are final content.
        cfg.allowReasoningReentry = false;
        // The upstream DeepSeek streaming parser classifies initial output as
        // reasoning even when the model omitted its opening marker.
        cfg.reasoningStartsWithoutTag = true;
        return cfg;
    }

    explicit Granite42ThinkingParser(ov::genai::Tokenizer& tokenizer,
        bool promoteReasoningToContent = false) :
        BaseOutputParser(tokenizer, defaultParsingConfig()),
        promoteReasoningToContent_(promoteReasoningToContent) {}

    void resetState() override {
        phaseEntryTagConsumed_ = implicitStart;
        generatedStartTagSeen_ = false;
        generatedStartTagConsumed_ = false;
        firstGeneratedStartTagWasStandalone_ = false;
        standaloneStartTagEventReasoningOrdinals_.clear();
        generatedStartTagEventReasoningOrdinal_.reset();
        emittedReasoningDeltaCount_ = 0;
    }

    std::optional<Delta> parseChunk(const std::string& chunk,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;

    void finalizeUnaryDeltas(std::vector<Delta>& deltas, bool finalContentWasPresent) const override;
};

}  // namespace ovms
