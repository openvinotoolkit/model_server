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
#include <vector>
#include <string>

#include "../qwen3/reasoning_parser.hpp"

namespace ovms {
class Gemma4ReasoningParser : public Qwen3ReasoningParser {
protected:
    const int64_t channelStartTokenId = 100;     // <|channel>
    const int64_t channelEndTokenId = 101;  // <channel|>
    
    const std::string reasoningStrIndicator = "thought\n";
    const std::string parsingStartTag = "<|channel>" + reasoningStrIndicator;
    const std::string parsingEndTag = "<channel|>";

    void skipToken(const std::vector<int64_t>& generatedTokens, size_t& pos, int64_t tokenId);

public:
    Gemma4ReasoningParser() = delete;
    explicit Gemma4ReasoningParser(ov::genai::Tokenizer& tokenizer) :
        Qwen3ReasoningParser(tokenizer) {}
    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }

    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags{this->parsingStartTag};
        return parsingStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
