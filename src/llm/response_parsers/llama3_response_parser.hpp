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
#include <vector>
#include "base_response_parser.hpp"

namespace ovms {
class Llama3ResponseParser : public BaseResponseParser {
protected:
    // Id of the <|python_tag|> which is a special token used to indicate the start of a tool calls
    int64_t botTokenId = 128010;
    // ";" is used as a separator between tool calls in the response
    std::string separator = ";";

public:
    Llama3ResponseParser() = delete;
    explicit Llama3ResponseParser(ov::genai::Tokenizer& tokenizer) :
        BaseResponseParser(tokenizer) {}

    ParsedResponse parse(const std::vector<int64_t>& generatedTokens) override;
};
}  // namespace ovms
