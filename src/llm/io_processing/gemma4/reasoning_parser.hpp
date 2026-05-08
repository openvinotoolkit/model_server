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

#include "../qwen3/reasoning_parser.hpp"

namespace ovms {
class Gemma4ReasoningParser : public Qwen3ReasoningParser {
protected:
    const int64_t reasoningTokenId = 100;
    const int64_t reasoningEndTokenId = 101;

public:
    Gemma4ReasoningParser() = delete;
    explicit Gemma4ReasoningParser(ov::genai::Tokenizer& tokenizer) :
        Qwen3ReasoningParser(tokenizer) {}
    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }
};
}  // namespace ovms
