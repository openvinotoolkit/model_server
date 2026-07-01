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

#include "src/logging.hpp"
#include "minicpm5_reasoning_parser.hpp"
#include "src/llm/io_processing/utils.hpp"

namespace ovms {
std::optional<rapidjson::Document> Minicpm5ReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) {
    if (tokens.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty tokens for Minicpm5ReasoningParser");
        return std::nullopt;
    }

    // Never emit the start tag token itself.
    if (std::find(tokens.begin(), tokens.end(), reasoningStartTokenId) != tokens.end()) {
        return std::nullopt;
    }

    // When the end tag token arrives it may be batched with the last reasoning
    // text by the streamer's delay buffer. Strip the </think> tag and emit any
    // text that preceded it; if nothing preceded it, emit nothing.
    std::string text = chunk;
    if (std::find(tokens.begin(), tokens.end(), reasoningEndTokenId) != tokens.end()) {
        const std::string& endTag = getParsingEndTag();
        const size_t pos = text.rfind(endTag);
        if (pos != std::string::npos) {
            text = text.substr(0, pos);
        } else {
            return std::nullopt;
        }
    }

    if (text.empty()) {
        return std::nullopt;
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("delta");
    writer.StartObject();
    writer.String("reasoning_content");
    writer.String(text.c_str());
    writer.EndObject();
    writer.EndObject();
    rapidjson::Document doc;
    doc.Parse(buffer.GetString());
    return doc;
}
}  // namespace ovms
