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
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "lfm25_reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {
namespace {
constexpr int64_t REASONING_START_TOKEN_ID = 124901;  // <think>
constexpr int64_t REASONING_END_TOKEN_ID = 124902;    // </think>
}  // namespace

std::optional<rapidjson::Document> Lfm25ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) {
    if (tokens.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty tokens for Lfm25ReasoningParser");
        return std::nullopt;
    }

    if (std::find(tokens.begin(), tokens.end(), REASONING_START_TOKEN_ID) != tokens.end() ||
        std::find(tokens.begin(), tokens.end(), REASONING_END_TOKEN_ID) != tokens.end()) {
        return std::nullopt;
    } else {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("delta");
        writer.StartObject();
        writer.String("reasoning_content");
        writer.String(chunk.c_str());
        writer.EndObject();
        writer.EndObject();
        rapidjson::Document doc;
        doc.Parse(buffer.GetString());
        return doc;
    }
}
}  // namespace ovms
