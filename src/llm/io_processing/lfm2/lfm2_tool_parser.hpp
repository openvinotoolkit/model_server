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
#include <string>
#include <vector>
#include "src/llm/io_processing/base_output_parser.hpp"
#include "../../../logging.hpp"
#include "./lfm2_utils.hpp"

namespace ovms {
class Lfm2ToolParser : public BaseOutputParser {
protected:
    static const std::string TOOL_CALL_START_TAG;
    static const std::string TOOL_CALL_END_TAG;

    static const int64_t toolCallStartTokenId;
    static const int64_t toolCallEndTokenId;

public:
    Lfm2ToolParser() = delete;

    static ParsingConfig defaultParsingConfig() {
        ParsingConfig cfg;
        cfg.startTags                    = {"<|tool_call_start|>"};
        cfg.specialTokenStartTags        = {"<|tool_call_start|>"};
        cfg.endTag                        = "<|tool_call_end|>";
        cfg.toolCallPhaseNeedsSpecialTokens = true;
        return cfg;
    }

    explicit Lfm2ToolParser(ov::genai::Tokenizer& tokenizer,
                             std::optional<ParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
                         configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;

private:
    std::string streamingContent;
    size_t streamingPosition{0};
    State currentState{State::Content};
    ToolCall toolCall;
    TagIds tagIds{TOOL_CALL_START_TAG, TOOL_CALL_END_TAG, toolCallStartTokenId, toolCallEndTokenId};

    int toolCallIndex{TOOL_CALL_INDEX_START};

    bool parseNewContent();
};
}  // namespace ovms
