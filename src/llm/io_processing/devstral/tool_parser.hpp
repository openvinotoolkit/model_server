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
#include "src/llm/apis/tool_schema_wrapper.hpp"

namespace ovms {
class DevstralToolParser : public BaseOutputParser {
    const int64_t argsTokenId;  // [ARGS]
    const int64_t botTokenId;   // [TOOL_CALLS]

    // in streaming mode we can rely on tags in string format as tokens are not available
    const std::string streamingParsingArgsStartTag = "[ARGS]";
    const std::string streamingParsingToolCallsStartTag = "[TOOL_CALLS]";
    const std::string streamingEndTag = "</s>";

    enum InternalState {
        AWAITING_START_TAG,
        AWAITING_ARGS_TAG,
        PROCESSING_ARGS
    };

    InternalState internalState = AWAITING_START_TAG;
    const ToolsSchemas_t& toolSchemas;
    // Index to track the current tool call being processed (-1 means no tool call has been started yet)
    int toolCallIndex = -1;
    std::string streamContent = "";  // content accumulated from stream chunks
    std::string toolName = "";
    std::optional<rapidjson::Document> sendFullDelta(ToolCall& toolCall);

public:
    DevstralToolParser() = delete;
    DevstralToolParser(ov::genai::Tokenizer& tokenizer, const ToolsSchemas_t& toolSchemas) :
        BaseOutputParser(tokenizer),
        argsTokenId([&tokenizer, this]() {
            // can not use streamingParsingArgsStartTag because object is not initialized yet
            auto encoded = tokenizer.encode("[ARGS]", {{"add_special_tokens", false}}).input_ids;
            if (encoded.get_shape()[0] != 1) {
                throw std::runtime_error("[ARGS] must be a single token in the tokenizer vocabulary.");
            }
            return encoded.data<int64_t>()[0];
        }()),
        botTokenId([&tokenizer, this]() {
            // can not use streamingParsingToolCallsStartTag because object is not initialized yet
            auto encoded = tokenizer.encode("[TOOL_CALLS]", {{"add_special_tokens", false}}).input_ids;
            if (encoded.get_shape()[0] != 1) {
                throw std::runtime_error("[TOOL_CALLS] must be a single token in the tokenizer vocabulary.");
            }
            return encoded.data<int64_t>()[0];
        }()),   
        toolSchemas(toolSchemas) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> toolCallStartTags{streamingParsingToolCallsStartTag};
        return toolCallStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        static const std::string toolCallEndTag = "</s>";
        return toolCallEndTag;
    }

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms
