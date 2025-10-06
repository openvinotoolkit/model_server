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

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {
class Phi4ToolParser : public BaseOutputParser {
protected:
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string parsingStartTag = "functools";
    const std::string parsingEndTag = "";

    // Streaming required members

    enum InternalState {
        AWAITING_START_TAG,
        AWAITING_TOOL_CALLS_OPENING_BRACKET,
        AWAITING_TOOL_CALL_OPENING_BRACE,
        PROCESSING_TOOL_CALL
    };

    InternalState internalState = AWAITING_START_TAG;
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    // Index to track the current tool call being processed (-1 means no tool call has been started yet)
    int toolCallIndex = -1;
    // Flag to indicate if double quote has been added at the beginning of arguments
    bool argumentsQuotesOpened = false;
    std::string unprocessedBuffer;

    // Stack of opened braces to track nested structures while in arguments collection phase
    // Starting with 1, since we count the tool call opening brace and expect it to be closed as arguments end
    size_t openBracesCount = 1;

    void movePostColonContentToUnprocessedBuffer(std::string& chunk);

public:
    Phi4ToolParser() = delete;
    explicit Phi4ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::string& getParsingStartTag() const override {
        return parsingStartTag;
    }
    const std::unordered_set<std::string>& getSpecialParsingStartTags() const override {
        static const std::unordered_set<std::string> specialParsingStartTags = {};
        return specialParsingStartTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
