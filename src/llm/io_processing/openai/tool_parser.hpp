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
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../base_output_parser.hpp"

namespace ovms {
class GptToolParser : public BaseOutputParser {
    // This is the same as reasoning parser start tag, however since reasoning is always checked before tool parser, it is not a problem.
    const std::string parsingStartTag = "<|channel|>commentary";

    // TODO: There should be more end tags: <|end|> and <|return|>, implement on upper level, find some example prompt that uses them
    const std::string parsingEndTag = "<|call|>";

    enum class StreamState : int {
        READING_CHANNEL,
        READING_CONSTRAIN,
        READING_MESSAGE,
    };

    // Streaming temp variables
    StreamState streamState = StreamState::READING_CHANNEL;
    std::string cache;
    bool isStreamingFunctionName = false;
    int toolCallIndex = -1;
    std::string functionNameCache;


    std::optional<rapidjson::Document> wrapDeltaIntoDocument(const std::string& chunk);

public:
    GptToolParser() = delete;
    explicit GptToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    // Unary
    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    // Streaming
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;

    const std::string& getParsingStartTag() const override {
        return parsingStartTag;
    }

    const std::unordered_set<std::string>& getSpecialParsingStartTags() const override {
        static const std::unordered_set<std::string> specialParsingStartTags = {};
        return specialParsingStartTags;
    }

    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
