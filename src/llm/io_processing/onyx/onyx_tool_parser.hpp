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
#include <set>
#include <stack>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/status.hpp"

namespace ovms {

// Onyx (early preview model) tool-call framing:
// TODO @atobiszei is functions namespace always "functions"?
//   <|start|>assistant to=functions.<name><|message|>{raw JSON args}<|eom|>
// Unlike qwen3coder/hermes3, Onyx never wraps arguments in a schema-validated,
// per-parameter structure -- the segment between "<|message|>" and "<|eom|>" is
// already the complete, raw JSON arguments blob the caller is expected to forward
// as-is (per the model card: "the SFT tokenizer tokenizes message content ...
// (raw body)"). So no tool-schema-driven type coercion is needed here, unlike
// Qwen3CoderToolParser.

// Pure state machine that accumulates raw generated text and hands back fully
// assembled tool calls -- mirrors Qwen3CoderToolParserImpl's split between "parse the
// framing" and "turn it into OpenAI delta JSON" (done by the owning OnyxToolParser).
// Because Onyx's arguments are already a complete raw JSON blob (no per-parameter
// schema coercion needed), a tool call is fully known as soon as its end tag is seen --
// unlike Qwen3Coder there is no incremental per-parameter streaming to do.
struct OnyxToolParserImpl {
    enum class State {
        Content,           // looking for the next "to=functions." recipient tag
        InsideName,         // accumulating the function name, looking for messageTag
        InsideArguments     // accumulating the raw JSON arguments blob, looking for endTag
    };

    // Marks the start of a tool-call turn; the function name follows immediately.
    static const std::string FUNCTIONS_RECIPIENT_TAG;
    // Separates the function name from the raw JSON arguments blob.
    static const std::string MESSAGE_TAG;
    // Tool calls always end the turn as a continuation (never a full turn end).
    static const std::string END_TAG;

    // Return all tool calls fully closed (end tag seen) in the aggregated content so far
    // that were not returned before -- nullopt if none completed yet.
    std::optional<ToolCalls_t> parseChunk(const std::string& chunk);
    std::optional<std::string> getCurrentFunctionName() const;
    Status removeToolCallsFromContentIfNeeded(std::string& outContent);

private:
    State currentState = State::Content;
    std::string streamContent;  // content accumulated from stream chunks
    size_t lastProcessedPosition{0};
    std::string currentFunctionName;
    struct ToolCallPositions {
        std::stack<size_t> begin;
        std::stack<size_t> end;
    };
    ToolCallPositions toolCallPositions;

    // Process streamContent from lastProcessedPosition until a state change happens;
    // return true if the state changed (caller should keep looping), false once no more
    // progress is possible with the currently available content.
    bool parseUntilStateChange(ToolCalls_t& toolCalls);
};

class OnyxToolParser : public BaseOutputParser {
private:
    // for streaming parsing we need to keep the parser as a member
    OnyxToolParserImpl streamParser;
    int toolCallIndex{-1};
    std::set<int> returnedFirstDeltas;
    std::set<int> returnedCompleteDeltas;

    std::optional<rapidjson::Document> sendFirstDeltaIfNeeded(const std::string& functionName);
    std::optional<rapidjson::Document> sendFullDelta(const ToolCalls_t& toolCalls);

public:
    OnyxToolParser() = delete;
    explicit OnyxToolParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<rapidjson::Document> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags{OnyxToolParserImpl::FUNCTIONS_RECIPIENT_TAG};
        return parsingStartTags;
    }
    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> specialParsingStartTags{};
        return specialParsingStartTags;
    }
    const std::string& getParsingEndTag() const override {
        return OnyxToolParserImpl::END_TAG;
    }
    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }

};
}  // namespace ovms
