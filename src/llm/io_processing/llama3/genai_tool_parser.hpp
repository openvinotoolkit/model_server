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

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/json_container.hpp>

#include "src/llm/io_processing/base_genai_parser.hpp"
#include "src/llm/io_processing/partial_json_builder.hpp"
#include "src/port/rapidjson_document.hpp"

namespace ovms {

/**
 * @brief Unary GenAI tool parser for Llama 3 Pythonic / JSON tool call format.
 *
 * Implements ov::genai::Parser::parse(JsonContainer&) — invoked once at the
 * end of generation. Reads message["content"], extracts tool calls delimited
 * by "<|python_tag|>" (or a bare "{" at position 0), splits them by ";",
 * parses each as JSON, and writes the result back into message["tool_calls"]
 * in OpenAI format.
 *
 * Because this class implements BaseGenAIParser (not BaseOutputParser), it
 * does not receive raw token IDs; all detection is text-based.
 */
class Llama3GenAIToolParser : public BaseGenAIParser {
    static constexpr const char* botTag = "<|python_tag|>";
    static constexpr const char* separator = ";";

public:
    Llama3GenAIToolParser() = default;
    ~Llama3GenAIToolParser() override = default;

    void parse(ov::genai::JsonContainer& message) override;

    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> tags = {botTag};
        return tags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> tags = {"{"};
        return tags;
    }

    const std::string& getParsingEndTag() const override {
        static const std::string empty;
        return empty;
    }
};

/**
 * @brief Incremental (streaming) GenAI tool parser for Llama 3 format.
 *
 * Implements ov::genai::IncrementalParser — invoked per streamed chunk.
 * Replicates the streaming logic of Llama3ToolParser using the same internal
 * state (PartialJsonBuilder, delay window, escape tracking) but outputs the
 * assembled tool call deltas into delta_message as a JsonContainer instead of
 * returning a rapidjson::Document directly.
 *
 * @note End-of-stream argument string closing is handled via flush(), which
 *       is called by GenAIParserAdapter::parseChunk when finishReason != NONE.
 *       This drains the last chunk from the delay window and properly closes
 *       the arguments JSON string.
 */
class Llama3GenAIIncrementalToolParser : public BaseGenAIIncrementalParser {
    static constexpr const char* botTag = "<|python_tag|>";
    static constexpr const char* separator = ";";

    // Internal streaming state — mirrors Llama3ToolParser
    rapidjson::Document lastJson;
    PartialJsonBuilder jsonBuilder;
    // Index of the tool call currently being assembled; -1 = not yet started
    int toolCallIndex = -1;
    // Two-slot delay window to defer argument chunks until we can close the string
    std::array<std::string, 2> argumentsDelayWindow{{"", ""}};
    int escapeLevel = 0;

    void startNextToolCall();

public:
    Llama3GenAIIncrementalToolParser() = default;
    ~Llama3GenAIIncrementalToolParser() override = default;

    /**
     * @brief Process one streamed chunk.
     *
     * @param delta_message Populated with the tool-call delta in OpenAI streaming
     *                      format when a delta is ready to emit; untouched otherwise.
     * @param delta_text    Incoming text chunk (modified in place to empty string
     *                      once tool call processing begins, so no raw tool-call
     *                      syntax leaks into the content stream).
     * @param delta_tokens  Unused.
     * @return Empty string — tool call content is not forwarded to the text stream.
     */
    std::string parse(
        ov::genai::JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) override;

    void reset() override;

    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> tags = {botTag};
        return tags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> tags = {"{"};
        return tags;
    }

    const std::string& getParsingEndTag() const override {
        static const std::string empty;
        return empty;
    }

    /**
     * @brief Flush the pending chunk from the delay window after end-of-generation.
     *
     * The streaming parser defers each incoming chunk by one slot to allow
     * look-ahead for the ";" separator. When generation ends without a ";"
     * (the normal case for the last tool call), the final chunk is still
     * sitting in window[1]. This method drains it, inserts the closing quote
     * before the last '}', and populates @p delta_message with the resulting
     * wrapDelta document.
     */
    void flush(ov::genai::JsonContainer& delta_message) override;
};

}  // namespace ovms
