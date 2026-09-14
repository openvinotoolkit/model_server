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
#include <utility>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../base_output_parser.hpp"

namespace ovms {

/*
    This parser handles only the analysis (reasoning) channel of the harmony format.
    Regular content (final/commentary channels) is handled separately by GptOssContentParser.
*/
class GptOssReasoningParser : public BaseOutputParser {
protected:
    const std::string parsingStartTag = "<|channel|>analysis<|message|>";
    const std::string parsingEndTag = "<|end|>";

    enum class StreamState : int {
        UNKNOWN = 0,
        READING_REASONING = 1,
    };
    StreamState state = StreamState::UNKNOWN;

public:
    GptOssReasoningParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<|channel|>analysis<|message|>"};
        cfg.endTag = "<|end|>";
        cfg.needsSpecialTokens = true;
        cfg.defaultDecodingWithSpecialTokens = true;
        return cfg;
    }

    explicit GptOssReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override { state = StreamState::UNKNOWN; }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
