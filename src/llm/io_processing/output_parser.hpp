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

namespace ovms {

class OutputParser {
    enum ProcessingPhase {
        UNKNOWN,
        CONTENT,
        REASONING,
        TOOL_CALLS
    };

    ov::genai::Tokenizer tokenizer;
    std::unique_ptr<BaseOutputParser> toolParser = nullptr;       // Tool parser for extracting tool calls
    std::unique_ptr<BaseOutputParser> reasoningParser = nullptr;  // Reasoning parser for extracting reasoning content

    // Streaming related members
    ProcessingPhase processingPhase = UNKNOWN;

    // Common method for parsing content chunk in the streaming mode.
    rapidjson::Document parseContentChunk(const std::string& chunk);

public:
    OutputParser() = delete;
    explicit OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName);

    ParsedOutput parse(const std::vector<int64_t>& generatedTokens);

    // Parse model output chunk in the steaming mode. Returns a JSON object containing the delta that conforms to OpenAI API
    // or nullopt if no response can be produced.
    std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse);
};
}  // namespace ovms
