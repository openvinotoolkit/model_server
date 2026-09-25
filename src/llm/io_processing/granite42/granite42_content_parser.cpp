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

#include <string>
#include <utility>
#include <vector>

#include "granite42_content_parser.hpp"

namespace ovms {

std::optional<Delta> Granite42ContentParser::parseChunk(
    const std::string& chunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {
    // No generated content segment exists. Returning nullopt keeps this distinct
    // from a non-empty segment which becomes empty after leading-newline removal.
    if (chunk.empty()) {
        return std::nullopt;
    }

    if (contentStarted_) {
        return ContentDelta{chunk};
    }

    const size_t firstNonNewline = chunk.find_first_not_of('\n');
    if (firstNonNewline == std::string::npos) {
        // vLLM turns this into content=None and leaves its first-content state
        // unset. OutputParser suppresses this empty delta after draining it.
        return ContentDelta{""};
    }

    contentStarted_ = true;
    return ContentDelta{chunk.substr(firstNonNewline)};
}

}  // namespace ovms
