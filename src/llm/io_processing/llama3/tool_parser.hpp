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
class Llama3ToolParser : public BaseOutputParser {
protected:
    const std::string parsingStartTag = "<|python_tag|>";
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string parsingEndTag = "";

    // Id of the <|python_tag|> which is a special token used to indicate the start of a tool calls
    int64_t botTokenId = 128010;
    // ";" is used as a separator between tool calls in the response
    std::string separator = ";";

    // Streaming required members
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    int toolCallIndex = -1;  // Index to track the current tool call being processed, -1 means we are not processing any tool call yet
    // Storing last two chunks of arguments to return delta with delay.
    // We do this to properly close arguments when tool call end tag is received.
    // With support for more models this could be moved to the base class.
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};
    int escapeLevel = 0;

    void startNextToolCall();

public:
    Llama3ToolParser() = delete;
    explicit Llama3ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::string& getParsingStartTag() const override {
        return parsingStartTag;
    }
    const std::unordered_set<std::string>& getSpecialParsingStartTags() const override {
        static const std::unordered_set<std::string> specialParsingStartTags = {"{"};
        return specialParsingStartTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
