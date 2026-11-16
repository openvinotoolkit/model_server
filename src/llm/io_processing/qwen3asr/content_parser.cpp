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

#include "content_parser.hpp"

#include "src/stringutils.hpp"

namespace ovms {

namespace {

const std::string asrTextMarker = "<asr_text>";

}  // namespace

Qwen3AsrContentParser::Qwen3AsrContentParser(ov::genai::Tokenizer& tokenizer) :
    BaseOutputParser(tokenizer) {}

void Qwen3AsrContentParser::resetState() {
    asrTextStarted = false;
}

std::optional<Delta> Qwen3AsrContentParser::parseChunk(const std::string& chunkResponse,
    const std::vector<int64_t>&,
    ov::genai::GenerationFinishReason) {
    if (asrTextStarted) {
        return ContentDelta{chunkResponse};
    }

    const size_t markerPosition = chunkResponse.find(asrTextMarker);
    if (markerPosition != std::string::npos) {
        asrTextStarted = true;
        return ContentDelta{chunkResponse.substr(markerPosition + asrTextMarker.size())};
    }
    if (stringsOverlap(chunkResponse, asrTextMarker)) {
        return std::nullopt;
    }
    return ContentDelta{};
}

}  // namespace ovms