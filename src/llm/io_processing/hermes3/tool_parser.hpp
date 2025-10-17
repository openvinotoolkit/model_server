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
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/partial_json_builder.hpp"

namespace ovms {
class Hermes3ToolParser : public BaseOutputParser {
protected:
    const std::string toolCallStartTag = "<tool_call>";
    const std::string toolCallEndTag = "</tool_call>";

    const std::string parsingStartTag = toolCallStartTag;
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string parsingEndTag = "";

    // Streaming required members and methods
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    // Index to track the current tool call being processed, -1 means we are not processing any tool call yet
    int toolCallIndex = -1;
    // Storing last two chunks of arguments to return delta with delay.
    // We do this to properly close arguments when tool call end tag is received.
    // With support for more models this could be moved to the base class.
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};
    // Buffer to store unprocessed part of the chunk that should be prepended to the next chunk
    std::string unprocessedBuffer;
    bool toolCallCompleted = false;

    // Split content in the chunk. Everything after the first colon is moved to unprocessedBuffer. Everything before and including the colon remains in chunk.
    // Example: chunk = '{"name": "tool_name", "arguments": {}}' -> chunk = '{"name":' , unprocessedBuffer = ' "tool_name", "arguments": {}}'
    void movePostColonContentToUnprocessedBuffer(std::string& chunk);
    // Split content in the chunk. Everything from the start tag is moved to unprocessedBuffer. Everything before the start tag remains in chunk.
    // Example: chunk = 'some text<tool_call>{"name":' -> chunk = 'some text' , unprocessedBuffer = '<tool_call>{"name":'
    void moveStartTagWithContentToUnprocessedBuffer(std::string& chunk);
    // Split content in the chunk. Everything after the end tag is moved to unprocessedBuffer. Everything before and including the end tag remains in chunk.
    // Example: chunk = '}}</tool_call><tool_call>' -> chunk = '}}</tool_call>' , unprocessedBuffer = '<tool_call>'
    void movePostEndTagContentToUnprocessedBuffer(std::string& chunk);
    // Look for the start tag. If found, clear state, increment toolCallIndex and remove the tag from the chunk.
    // Returns true if start tag was found and handled, false otherwise.
    bool findAndHandleStartTag(std::string& chunk);
    // Look for the end tag. If found, set flag to clear state in the next call and remove the tag from the chunk.
    // Returns true if end tag was found and handled, false otherwise.
    bool findAndHandleEndTag(std::string& chunk);
    // Add opening quote to the arguments value to force string type.
    void openArgumentsString(std::string& chunk);
    // Add closing quote to the arguments value to properly close the string.
    void closeArgumentsString(std::string& chunk);
    void clearState();

public:
    Hermes3ToolParser() = delete;
    explicit Hermes3ToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags = {parsingStartTag};
        return parsingStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
