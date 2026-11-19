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
#include "src/llm/io_processing/qwen3/reasoning_parser.hpp"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ovms {
class DevstralReasoningParser : public Qwen3ReasoningParser {
public:
    DevstralReasoningParser() = delete;

    static OutputParsingConfig getDefaultReasoningParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"[THINK]"};
        cfg.endTags = {"[/THINK]"};
        return cfg;
    }

    explicit DevstralReasoningParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        Qwen3ReasoningParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : getDefaultReasoningParsingConfig()) {}
};
}  // namespace ovms
