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
#include <utility>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/partial_json_builder.hpp"

namespace ovms {
class Llama3ToolParser : public BaseOutputParser {
protected:
    // ";" separator between tool calls
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

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        // <|python_tag|> is a special token. Put it in both startTags (text-based,
        // used when the text is passed directly, e.g. in streaming tests) and
        // tokenIdStartTags (token-ID-based, used in production where the token
        // decodes to empty with skip_special_tokens=true).
        cfg.startTags = {"<|python_tag|>"};
        cfg.tokenIdStartTags = {"<|python_tag|>"};
        cfg.preambleStartTags = {"{"};
        return cfg;
    }

    explicit Llama3ToolParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    void resetState() override {
        lastJson.SetNull();
        jsonBuilder.clear();
        toolCallIndex = -1;
        argumentsDelayWindow = {{"    ", ""}};
        argumentsDelayWindow[0].clear();
        argumentsDelayWindow[1].clear();
        escapeLevel = 0;
    }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
};
}  // namespace ovms
