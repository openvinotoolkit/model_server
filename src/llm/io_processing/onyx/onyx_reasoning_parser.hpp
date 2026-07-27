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

#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"

namespace ovms {

// Onyx (early preview model) framing:
// TODO @atobiszei simplify comment. tag naming convention. no need to define all tags here
//   <|start|>assistant[ to=<recipient>]<|message|>{content}{<|eom|>|<|eot|>}
// The chat template never emits a "<think>"-style dedicated reasoning tag: private
// chain-of-thought is just an assistant turn routed with recipient="self", ending in
// the continuation marker "<|eom|>" (never "<|eot|>", which is reserved for turns that
// end the whole assistant turn -- i.e. the final answer).
//
// Because generation stops at the first "<|eom|>"/"<|eot|>"/"<|end_of_text|>" (see
// generation_config.json's eos_token_id list in the Onyx HF conversion script), a single
// generate() call only ever produces ONE such framed segment. This parser is therefore
// also responsible for stripping the generic " to=<recipient>"+"<|message|>"+terminator
// envelope from plain final-answer turns (recipient="user" or absent) -- this class runs
// before the tool parser (see OutputParser::parse()), so it must NOT touch content when
// the envelope routes to a function call (recipient="functions.<name>"); it leaves that
// segment untouched so OnyxToolParser can find and parse it afterwards.
class OnyxReasoningParser : public BaseOutputParser {
protected:
    // Marks a private chain-of-thought turn (recipient="self").
    const std::string selfRecipientTag = "to=self";
    // Marks a tool-call turn (recipient="functions.<name>") -- left untouched here.
    const std::string functionsRecipientTag = "to=functions.";
    // Separates the routing prefix from the turn's body.
    const std::string messageTag = "<|message|>";
    // Terminator for continuation turns (reasoning and tool calls).
    const std::string continuationEndTag = "<|eom|>";
    // Terminator for turn-final turns (plain final answers).
    const std::string turnFinalEndTag = "<|eot|>";

public:
    OnyxReasoningParser() = delete;
    explicit OnyxReasoningParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags{selfRecipientTag};
        return parsingStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        return continuationEndTag;
    }
    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms
