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

#include "../base_output_parser.hpp"

namespace ovms {

// Implements the streaming content normalization in IBM's
// granite_thinking_parser.py: strip '\n' only from the first non-empty final
// content delta. An all-newline first segment drains as an empty delta while
// retaining contentStarted_ == false, matching the vLLM streaming wrapper.
class Granite42ContentParser final : public BaseOutputParser {
    bool contentStarted_ = false;

public:
    Granite42ContentParser() = delete;
    explicit Granite42ContentParser(ov::genai::Tokenizer& tokenizer) :
        BaseOutputParser(tokenizer) {}

    void resetState() override { contentStarted_ = false; }

    std::optional<Delta> parseChunk(const std::string& chunk,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;
};

}  // namespace ovms
