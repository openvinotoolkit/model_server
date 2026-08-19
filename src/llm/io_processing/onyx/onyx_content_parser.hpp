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
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {

// Handles plain-content turns in the Onyx harmony format.
//
// Strips the routing preamble ("<|eom|><|start|>assistant to=user<|message|>") that
// precedes user-visible content and the terminator ("<|eot|>") that follows it.
//
// Tag handling uses two tiers:
//   structural  — hold-eligible: "<|eom|>", "<|start|>assistant ", "<|message|>", "<|eot|>"
//                  A partial match at the END of the buffer causes a hold until the tag
//                  is fully assembled (or definitively not present).
//   routing     — immediate-erase: " to=user", "to=user"
//                  Erased whenever fully present; never trigger a hold, because their
//                  leading space would cause false FOUND_INCOMPLETE hits on any content
//                  token that ends with a space.
//
// startTags = {"to=user<|message|>"} — used by OutputParser::TOOL_CALLS_WAITING_FOR_TOOL
// to detect when the model switches from tool calls back to a plain content turn.
class OnyxContentParser : public BaseOutputParser {
public:
    OnyxContentParser() = delete;
    explicit OnyxContentParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt);

    static OutputParsingConfig defaultParsingConfig();

    // Returns a content delta document, or nullopt to hold (partial structural-tag match).
    // Returns a document with empty content string when the buffer contained only preamble
    // (caller suppresses the emit in that case).
    std::optional<rapidjson::Document> parseChunk(const std::string& buffer,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;
};

}  // namespace ovms
