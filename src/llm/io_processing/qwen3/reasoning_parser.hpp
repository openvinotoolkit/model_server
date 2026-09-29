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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../base_output_parser.hpp"

namespace ovms {
class Qwen3ReasoningParser : public BaseOutputParser {
protected:
    // Tags used to identify the reasoning segment in the content
    const std::string parsingStartTag = "<think>";
    const std::string parsingEndTag = "</think>";

private:
    // Tracks whether the phase-entry start tag has already been consumed by parseChunk.
    // On the very first call the start tag is stripped (explicit start) or skipped
    // (implicit start — tag was already in the prompt). After that, any <think> in
    // the stream is treated as literal reasoning content and emitted as-is.
    bool phaseEntryTagConsumed_ = false;

public:
    Qwen3ReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<think>"};
        cfg.endTag = "</think>";
        return cfg;
    }

    explicit Qwen3ReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override { phaseEntryTagConsumed_ = false; }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
