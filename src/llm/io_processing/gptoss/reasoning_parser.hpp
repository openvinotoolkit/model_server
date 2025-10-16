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
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../base_output_parser.hpp"

namespace ovms {

/*
    This parser handles reasoning, but is also responsible for parsing regular content.
    This model group requires use of reasoning to work even if reasoning is not needed.
    This is due to the fact that regular content is placed in harmony format in similar fashion as reasoning.
*/
class GptOssReasoningParser : public BaseOutputParser {
protected:
    const std::string parsingStartTag = "<|channel|>analysis<|message|>";
    const std::string parsingEndTag = "<|end|>";

    enum class StreamState : int {
        UNKNOWN = 0,
        READING_REASONING = 1,
        READING_CONTENT = 2,
    };
    StreamState state = StreamState::UNKNOWN;

public:
    GptOssReasoningParser() = delete;
    explicit GptOssReasoningParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    // Unary
    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    // Streaming
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;

    const std::vector<std::string>& getParsingStartTags() const override {
        // If you add another element you have to update implementation as well
        // as mostly it assumed just one element
        static const std::vector<std::string> parsingStartTags{parsingStartTag};
        return parsingStartTags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags = {
            "<|channel|>final<|message|>",
            "<|channel|>commentary<|message|>",               // Preable to reasoning, users usually sees that
            "<|start|>assistant<|channel|>final<|message|>",  // Final content users sees
        };
        return specialParsingStartTags;
    }

    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms
