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
#include <optional>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../base_output_parser.hpp"

namespace ovms {
class Hermes3ToolParser : public BaseOutputParser {
protected:
    const std::string toolCallStartTag = "<tool_call>";
    const std::string toolCallEndTag = "</tool_call>";

    const std::string parsingStartTag = toolCallStartTag;
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string parsingEndTag = "";

    // Streaming required members
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    int toolCallIndex = -1;  // Index to track the current tool call being processed, -1 means we are not processing any tool call yet
    // Storing last two chunks of arguments to return delta with delay.
    // We do this to properly close arguments when tool call end tag is received.
    // With support for more models this could be moved to the base class.
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};

public:
    Hermes3ToolParser() = delete;
    explicit Hermes3ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason fr) override;
    const std::string& getParsingStartTag() const override {
        return parsingStartTag;
    }
    const std::unordered_set<std::string>& getBeginningOnlyParsingTags() const override {
        static const std::unordered_set<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
