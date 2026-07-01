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
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../base_output_parser.hpp"

namespace ovms {
class Qwen3ReasoningParser : public BaseOutputParser {
protected:
    // Tags used to identify the reasoning segment in the content
    const std::string parsingStartTag = "<think>";
    const std::string parsingEndTag = "</think>";

public:
    Qwen3ReasoningParser() = delete;

    static ParsingConfig defaultParsingConfig() {
        ParsingConfig cfg;
        cfg.startTags = {"<think>"};
        cfg.endTag    = "</think>";
        return cfg;
    }

    explicit Qwen3ReasoningParser(ov::genai::Tokenizer& tokenizer,
                                   std::optional<ParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
                         configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
