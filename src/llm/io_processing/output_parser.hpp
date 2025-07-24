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

#include <memory>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "base_output_parser.hpp"
#include "llama3/output_parser.hpp"
#include "qwen3/output_parser.hpp"
#include "hermes3/output_parser.hpp"
#include "phi4/output_parser.hpp"

namespace ovms {
class OutputParser {
    std::unique_ptr<BaseOutputParser> parser_impl;

public:
    OutputParser() = delete;
    explicit OutputParser(ov::genai::Tokenizer& tokenizer, std::string parserName) {
        if (parserName == "llama3") {
            parser_impl = std::make_unique<Llama3OutputParser>(tokenizer);
        } else if (parserName == "qwen3") {
            parser_impl = std::make_unique<Qwen3OutputParser>(tokenizer);
        } else if (parserName == "hermes3") {
            parser_impl = std::make_unique<Hermes3OutputParser>(tokenizer);
        } else if (parserName == "phi4") {
            parser_impl = std::make_unique<Phi4OutputParser>(tokenizer);
        } else {
            throw std::invalid_argument("Unsupported response parser: " + parserName);
        }
    }
    ParsedOutput parse(const std::vector<int64_t>& generatedTokens) {
        return parser_impl->parse(generatedTokens);
    }
    std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse) {
        return parser_impl->parseChunk(chunkResponse);
    }
};
}  // namespace ovms
