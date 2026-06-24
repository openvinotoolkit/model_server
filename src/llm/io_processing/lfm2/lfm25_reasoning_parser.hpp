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

#include "../base_output_parser.hpp"

namespace ovms {
class Lfm25ReasoningParser : public BaseOutputParser {
protected:
    const std::string parsingStartTag = "<think>";
    const std::string parsingEndTag = "</think>";

    const int64_t reasoningStartTokenId = 124901;  // <think>
    const int64_t reasoningEndTokenId = 124902;    // </think>

public:
    Lfm25ReasoningParser() = delete;
    explicit Lfm25ReasoningParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
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

    // It may be removed after changing logic in Lfm2ToolParser to use tokens in streaming instead of chunk content, both tool parser and reasoning parser need to have the same value for this function  
    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}
