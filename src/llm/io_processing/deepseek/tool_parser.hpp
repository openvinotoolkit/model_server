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

namespace ovms {
class DeepSeekToolParser : public BaseOutputParser {
    // DeepSeek tool call tags are multi-token text sequences (NOT single special tokens).
    // They use fullwidth vertical bar U+FF5C and lower one-eighth block U+2581.
    // Template format:
    //   <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNC_NAME
    //   ```json
    //   {"param": "value"}
    //   ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    static const std::string toolCallsBeginTag;   // <｜tool▁calls▁begin｜>
    static const std::string toolCallBeginTag;     // <｜tool▁call▁begin｜>
    static const std::string toolSepTag;           // <｜tool▁sep｜>
    static const std::string toolCallEndTag;       // <｜tool▁call▁end｜>
    static const std::string toolCallsEndTag;      // <｜tool▁calls▁end｜>

    static const std::string parsingStartTag;      // <｜tool▁calls▁begin｜>
    static const std::string parsingEndTag;        // <｜tool▁calls▁end｜>

    enum InternalState {
        AWAITING_SEP,         // Looking for <｜tool▁sep｜> (stripping calls_begin, call_begin, function)
        AWAITING_FUNC_NAME,   // Found sep, accumulating until first newline gives us the function name
        PROCESSING_ARGS,      // Have function name, collecting arguments until call_end
    };

    InternalState internalState = AWAITING_SEP;
    int toolCallIndex = -1;
    std::string streamContent;
    std::string toolName;

    // Extract JSON arguments from the raw content between tool_sep and tool_call_end.
    // Handles both markdown code-block wrapped and raw JSON formats.
    static std::string extractArguments(const std::string& raw);

public:
    DeepSeekToolParser() = delete;
    explicit DeepSeekToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;

    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> tags{parsingStartTag};
        return tags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> empty{};
        return empty;
    }
    const std::string& getParsingEndTag() const override {
        return parsingEndTag;
    }
};
}  // namespace ovms
