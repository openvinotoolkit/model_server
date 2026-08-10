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
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {

class OnyxReasoningParser : public BaseOutputParser {
protected:
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
    explicit OnyxReasoningParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags{selfRecipientTag};
        return parsingStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        return continuationEndTag;
    }
    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms
