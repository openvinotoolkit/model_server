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

#include "./lfm2_utils.hpp"

namespace ovms {
class Lfm25ToolParser : public BaseOutputParser {
protected:
    static const std::string TOOL_CALL_START_TAG;
    static const std::string TOOL_CALL_END_TAG;

    static const int64_t toolCallStartTokenId;
    static const int64_t toolCallEndTokenId;
    static const int64_t reasoningEndTokenId;

public:
    Lfm25ToolParser() = delete;
    explicit Lfm25ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags = {TOOL_CALL_START_TAG};
        return parsingStartTags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }

    const std::vector<std::string>& getSpecialTagsToErase() const override {
        static const std::vector<std::string> tagsToErase = {EOS_TOKEN_STR};
        return tagsToErase;
    }

    const std::string& getParsingEndTag() const override {
        return TOOL_CALL_END_TAG;
    }

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }

private:
    std::string streamingContent;
    size_t streamingPosition{0};
    State currentState{State::Content};
    ToolCall toolCall;

    int toolCallIndex{TOOL_CALL_INDEX_START};

    bool parseNewContent();
};
}  // namespace ovms
