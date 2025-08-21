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
class MistralToolParser : public BaseOutputParser {
    const int64_t botTokenId = 5;  // [TOOL_CALLS]

public:
    MistralToolParser() = delete;
    explicit MistralToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason fr) override;
    const std::string& getParsingStartTag() const override {
        static const std::string toolCallStartTag = "[TOOL_CALLS]";
        return toolCallStartTag;
    }
    const std::unordered_set<std::string>& getBeginningOnlyParsingTags() const override {
        static const std::unordered_set<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }
    // Tools calls are expected to be the last part of the content, so we do not specify an end tag.
    const std::string& getParsingEndTag() const override {
        static const std::string toolCallEndTag = "";
        return toolCallEndTag;
    }
};
}  // namespace ovms
