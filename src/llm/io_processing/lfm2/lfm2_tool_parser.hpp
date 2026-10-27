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
#include <optional>
#include <deque>
#include <map>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {

// Streaming state machine states for LFM2 / LFM2.5 tool-call parsing.
enum class Lfm2ParseState {
    Content,
    ToolCallStarted,
    ToolCallParameters,
    ToolCallEnded,
    AfterToolCall
};

// Unified tool parser for both LFM2 and LFM2.5.
// The two model families share identical tool-call grammar; the only differences
// are the token IDs assigned by their respective tokenizers and whether the
// chat template appends <|im_end|> after tool calls (LFM2.5 only).
// The correct OutputParsingConfig variant is chosen automatically via configForTokenizer().
class Lfm2ToolParser : public BaseOutputParser {
public:
    Lfm2ToolParser() = delete;

    // OutputParsingConfig for LFM2 and LFM2.5. Both model families use the same
    // tool-call grammar and token-boundary strings; the only model-specific
    // behaviour (stripping <|im_end|> from content) is a no-op on LFM2 since
    // that model's chat template never emits <|im_end|> in tool-call context.
    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<|tool_call_start|>"};
        cfg.tokenIdStartTags = {"<|tool_call_start|>"};
        cfg.endTags = {"<|tool_call_end|>"};
        cfg.needsSpecialTokens = true;
        return cfg;
    }

    // Low-level compatibility constructor without executable-tool validation.
    // Production OutputParser must use the registry-taking overload below.
    explicit Lfm2ToolParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    // Executable publication path. The registry is copied so retained chunks
    // never depend on request-owned RapidJSON values or their lifetime.
    explicit Lfm2ToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& schemas, bool parallelToolCalls = true,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()),
        registryEnabled(true), parallelToolCalls(parallelToolCalls) {
        for (const auto& entry : schemas) registeredSchemas.emplace(entry.first, entry.second.stringRepr);
    }

    void resetState() override {
        streamingContent.clear();
        streamingPosition = 0;
        currentState = Lfm2ParseState::Content;
        toolCall = {};
        toolCallIndex = -1;
        pendingDeltas.clear();
        publishedCallIds.clear();
        retainedEnvelope = false;
        terminal = false;
        retainedScanPosition = 0;
        retainedQuote = 0;
        retainedEscape = false;
    }

    std::optional<Delta> parseChunk(const std::string& chunk,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;

    bool isStructuralEndTag(const std::string& chunk, size_t offset) const override;

    std::optional<Delta> drainPendingDelta() override {
        return parseChunk("", {}, ov::genai::GenerationFinishReason::NONE);
    }

private:
    std::string streamingContent;
    size_t streamingPosition{0};
    Lfm2ParseState currentState{Lfm2ParseState::Content};
    ToolCall toolCall;
    int toolCallIndex{-1};

    bool registryEnabled{false};
    bool parallelToolCalls{true};
    std::map<std::string, std::string> registeredSchemas;
    std::deque<Delta> pendingDeltas;
    std::unordered_set<std::string> publishedCallIds;
    bool retainedEnvelope{false};
    bool terminal{false};
    size_t retainedScanPosition{0};
    char retainedQuote{0};
    bool retainedEscape{false};

    std::optional<Delta> parseRetainedChunk(const std::string& chunk,
        ov::genai::GenerationFinishReason finishReason);
    void retainEnvelope(const std::string& body);

    bool parseNewContent();
};

}  // namespace ovms
