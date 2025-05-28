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

#include "base_response_parser.hpp"
#include "llama3_response_parser.hpp"
#include "qwen3_response_parser.hpp"

class ResponseParser {
    std::unique_ptr<BaseResponseParser> parser_impl;

public:
    ResponseParser() = delete;
    explicit ResponseParser(ov::genai::Tokenizer& tokenizer, std::string parserName) {
        // Parser name is read from tokenizer_config.json, "response_parser_name" field.
        if (parserName == "llama3") {
            parser_impl = std::make_unique<Llama3ResponseParser>(tokenizer);
        } else if (parserName == "qwen3") {
            parser_impl = std::make_unique<Qwen3ResponseParser>(tokenizer);
        } else {
            throw std::invalid_argument("Unsupported response parser: " + parserName);
        }
    }
    ParsedResponse parse(const std::vector<int64_t>& generatedTokens) {
        return parser_impl->parse(generatedTokens);
    }
};
