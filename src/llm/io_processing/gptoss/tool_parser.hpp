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

#include "src/port/rapidjson_document.hpp"

#include "../base_output_parser.hpp"

namespace ovms {
class GptOssToolParser : public BaseOutputParser {
    // This is the same as reasoning parser start tag, however since reasoning is always checked before tool parser, it is not a problem.
    static const std::string parsingStartTag;
    static const std::string parsingEndTag;

    enum class StreamState : int {
        READING_HEADER,
        READING_CONSTRAIN,
        READING_MESSAGE,
    };

    enum class StepAction : int {
        CONTINUE,
        NEED_MORE_INPUT,
        EMIT_DELTA,
    };

    struct StepResult {
        StepAction action;
        std::optional<Delta> delta;
    };

    static StepResult continueParsing() { return {StepAction::CONTINUE, std::nullopt}; }
    static StepResult needMoreInput() { return {StepAction::NEED_MORE_INPUT, std::nullopt}; }
    static StepResult emitDelta(Delta delta) { return {StepAction::EMIT_DELTA, std::move(delta)}; }

    // Streaming temp variables
    StreamState streamState = StreamState::READING_HEADER;
    std::string cache;
    bool isStreamingFunctionName = false;
    int toolCallIndex = -1;
    std::string functionNameCache;

    std::optional<Delta> wrapDeltaIntoDocument(const std::string& chunk);
    StepResult consumeHeader(std::string& chunk, std::optional<Delta>& pendingDelta);
    StepResult consumeConstrain(std::string& chunk, std::optional<Delta>& pendingDelta);
    StepResult consumeMessage(std::string& chunk, std::optional<Delta>& pendingDelta);
    StepResult consumePartialHeader(std::string chunk);
    bool consumeToolCallStartTag(std::string& chunk);
    bool consumeCompleteHeader(std::string& chunk, std::optional<Delta>& result);
    bool consumeHeaderMarker(std::string& chunk, std::optional<Delta>& result);
    bool closeMessage(std::string& chunk, std::optional<Delta>& result);

    void clearHeaderState();

public:
    GptOssToolParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<|channel|>commentary to=",
            "<|channel|>analysis to=",
            "<|start|>assistant to="};
        cfg.endTags = {"<|call|>", "<|end|>", "<|return|>"};
        cfg.needsSpecialTokens = true;
        cfg.defaultDecodingWithSpecialTokens = true;
        return cfg;
    }

    explicit GptOssToolParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override {
        streamState = StreamState::READING_HEADER;
        toolCallIndex = -1;
        clearHeaderState();
    }

    // Known limitation: arguments arriving with the function name in one chunk are dropped.
    // The one-delta parser contract requires a higher-layer change to deliver both separately.
    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
