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
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {

class OnyxReasoningParser : public BaseOutputParser {
    // Marks a private chain-of-thought turn (recipient="self").
    const std::string selfRecipientTag = "to=self";
    // Separates the routing prefix from the turn's body.
    const std::string messageTag = "<|message|>";
    // Terminator for continuation turns (reasoning and tool calls).
    const std::string continuationEndTag = "<|eom|>";
    // Terminator for turn-final turns (plain final answers).
    const std::string turnFinalEndTag = "<|eot|>";

public:
    OnyxReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"to=self"};
        cfg.endTag = "<|eom|>";
        cfg.needsSpecialTokens = true;
        cfg.defaultDecodingWithSpecialTokens = true;
        return cfg;
    }

    explicit OnyxReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override {
        headerConsumed = false;
        headerBuffer.clear();
    }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;

private:
    // Accumulates content until <|message|> is fully consumed at the start of each turn.
    bool headerConsumed = false;
    std::string headerBuffer;
};
}  // namespace ovms
