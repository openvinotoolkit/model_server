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

#include "./lfm2_tool_parser.hpp"

namespace ovms {
class Lfm25ToolParser : public Lfm2ToolParser {
protected:
    static const int64_t toolCallStartTokenId = 124905; // <|tool_call_start|>
    static const int64_t toolCallEndTokenId = 124906;   // <|tool_call_end|>
    static const int64_t reasoningEndTokenId = 124902;   // </think>
public:
    Lfm25ToolParser() = delete;
    explicit Lfm25ToolParser(ov::genai::Tokenizer& tokenizer) :
        Lfm2ToolParser(tokenizer, toolCallStartTokenId, toolCallEndTokenId) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override {
        Lfm2ToolParser::parse(parsedOutput, generatedTokens);

        auto contentTokens = std::vector<int64_t>(generatedTokens.begin(), generatedTokens.end());
        auto reasoningEnd = std::find(contentTokens.begin(), contentTokens.end(), reasoningEndTokenId);
        if (reasoningEnd != contentTokens.end()) {
            contentTokens.erase(contentTokens.begin(), reasoningEnd + 1);
        }
        auto toolCallStart = std::find(contentTokens.begin(), contentTokens.end(), toolCallStartTokenId);
        auto toolCallEnd = std::find(contentTokens.begin(), contentTokens.end(), toolCallEndTokenId);
        if (toolCallStart != contentTokens.end() && toolCallEnd != contentTokens.end() && toolCallStart < toolCallEnd) {
            contentTokens.erase(toolCallStart, toolCallEnd + 1);
        }
        parsedOutput.content = tokenizer.decode(contentTokens, ov::AnyMap{ov::genai::skip_special_tokens(true)});
    }
};
}
