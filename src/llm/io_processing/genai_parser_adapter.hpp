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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/json_container.hpp>
#include <openvino/genai/tokenizer.hpp>

#include "src/llm/io_processing/base_genai_parser.hpp"
#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/port/rapidjson_document.hpp"

namespace ovms {

/**
 * @brief Adapter that bridges BaseGenAIParser / BaseGenAIIncrementalParser
 *        with the OVMS BaseOutputParser interface.
 *
 * GenAIParserAdapter wraps a pair of OVMS-extended GenAI parser variants and
 * translates calls to/from the OVMS OutputParser flow:
 *  - parse()      (unary mode)   delegates to the wrapped BaseGenAIParser
 *  - parseChunk() (streaming)    delegates to the wrapped BaseGenAIIncrementalParser
 *
 * Start/end tags and the streaming-with-special-tokens flag are obtained
 * directly from the wrapped parsers via the BaseGenAIParser /
 * BaseGenAIIncrementalParser interface, keeping the adapter free of
 * parser-specific configuration.
 */
class GenAIParserAdapter : public BaseOutputParser {
    std::shared_ptr<BaseGenAIParser> parser;
    std::shared_ptr<BaseGenAIIncrementalParser> incrementalParser;

    /**
     * @brief Extract tool_calls from a GenAI JsonContainer into OVMS ToolCalls_t.
     *
     * Expects the standard OpenAI tool_calls format:
     * [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
     */
    static void extractToolCalls(const ov::genai::JsonContainer& message, ToolCalls_t& toolCalls);

    /**
     * @brief Convert a GenAI JsonContainer to a rapidjson::Document via JSON serialization.
     */
    static rapidjson::Document jsonContainerToDocument(const ov::genai::JsonContainer& container);

public:
    GenAIParserAdapter() = delete;

    /**
     * @brief Construct adapter wrapping a unary and an incremental GenAI parser.
     *
     * Tag configuration and the streaming-with-special-tokens flag are queried
     * from the parsers themselves via the BaseGenAIParser /
     * BaseGenAIIncrementalParser interface.
     *
     * @param tokenizer         Tokenizer forwarded to BaseOutputParser.
     * @param parser            OVMS-extended GenAI parser for unary (non-streaming) mode.
     * @param incrementalParser OVMS-extended GenAI parser for streaming mode.
     */
    explicit GenAIParserAdapter(
        ov::genai::Tokenizer& tokenizer,
        std::shared_ptr<BaseGenAIParser> parser,
        std::shared_ptr<BaseGenAIIncrementalParser> incrementalParser);

    /**
     * @brief Parse full (non-streaming) output via the wrapped ov::genai::Parser.
     *
     * Wraps parsedOutput.content in a GenAI JsonContainer, invokes the parser,
     * then maps results back:
     *   - content              -> parsedOutput.content
     *   - tool_calls           -> parsedOutput.toolCalls
     *   - reasoning_content    -> parsedOutput.reasoning
     *
     * @note generatedTokens is accepted for interface compatibility but not
     *       forwarded to the GenAI parser.
     */
    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;

    /**
     * @brief Parse a streaming chunk via the wrapped ov::genai::IncrementalParser.
     *
     * Builds an empty delta JsonContainer and invokes the incremental parser.
     * If the parser emits structured data (tool calls, reasoning), the delta
     * JsonContainer is converted to a rapidjson::Document and returned.
     * If the parser emits plain filtered content, it is wrapped in
     * {"content": <text>} and returned. Returns std::nullopt when the chunk
     * does not yet produce a meaningful response.
     *
     * @note When finishReason != NONE the adapter also calls flush() on the
     *       incremental parser to drain any content held in its delay window.
     *       The flush delta is combined with the parse delta (argument strings
     *       concatenated) when both are non-null.
     */
    std::optional<rapidjson::Document> parseChunk(
        const std::string& chunkResponse,
        ov::genai::GenerationFinishReason finishReason) override;

    const std::vector<std::string>& getParsingStartTags() const override;
    const std::vector<std::string>& getSpecialParsingStartTags() const override;
    const std::string& getParsingEndTag() const override;
    bool requiresStreamingWithSpecialTokens() const override;

private:
    /**
     * @brief Combine two wrapDelta documents by concatenating their
     *        delta.tool_calls[0].function.arguments strings.
     *
     * Used when both parse() and flush() produce argument deltas for the
     * final chunk so that only a single document needs to be returned.
     * If the documents cannot be combined (unexpected structure), @p primary
     * is returned unchanged.
     *
     * @param primary  Delta from the normal parse() call (moved).
     * @param flushed  Delta from flush() (moved, used only for its arguments value).
     * @return Combined document, or @p primary if combination is not applicable.
     */
    static rapidjson::Document combineArgumentDeltas(
        rapidjson::Document primary,
        rapidjson::Document flushed);
};

}  // namespace ovms
