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

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>
#include <regex>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "harmony.hpp"
#include "../utils.hpp"

namespace ovms {

void GptToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    openai::Harmony harmony(tokenizer, generatedTokens);
    if (!harmony.parse()) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Harmony parsing failed");
    } else {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed with harmony");
    }

    parsedOutput.content = harmony.getContent();  // what if someone has only reasoning parsers and no tool parser?
    parsedOutput.toolCalls = harmony.getToolCalls();
}

std::optional<rapidjson::Document> GptToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    // Not implemented
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "GptToolParser::parseChunk is not implemented");
    return std::nullopt;
}
}  // namespace ovms
