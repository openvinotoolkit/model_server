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

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "reasoning_parser.hpp"
#include "../utils.hpp"

namespace ovms {

std::optional<rapidjson::Document> Qwen3ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Qwen3ReasoningParser");
        return std::nullopt;
    }

    if (chunk.find(getParsingStartTags()[0]) != std::string::npos || chunk.find(getParsingEndTag()) != std::string::npos) {
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
    return std::nullopt;
}
}  // namespace ovms
