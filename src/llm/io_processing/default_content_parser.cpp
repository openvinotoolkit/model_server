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

#include <string>
#include <utility>
#include <vector>

#include "src/stringutils.hpp"

#include "default_content_parser.hpp"

namespace ovms {

DefaultContentParser::DefaultContentParser(ov::genai::Tokenizer& tokenizer,
    std::vector<std::string> stringsToErase) :
    BaseOutputParser(tokenizer, [&] {
        OutputParsingConfig cfg;
        cfg.stringsToErase = std::move(stringsToErase);
        return cfg;
    }()) {}

std::optional<Delta> DefaultContentParser::parseChunk(
    const std::string& buf,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {

    bool anyComplete = false;
    for (const auto& tag : parsingConfig.stringsToErase) {
        if (buf.find(tag) != std::string::npos)
            anyComplete = true;
    }
    if (!anyComplete) {
        for (const auto& tag : parsingConfig.stringsToErase) {
            if (stringsOverlap(buf, tag))
                return std::nullopt;  // partial match — hold
        }
    }

    std::string content = buf;
    if (anyComplete) {
        for (const auto& tag : parsingConfig.stringsToErase) {
            size_t pos = 0;
            while ((pos = content.find(tag, pos)) != std::string::npos)
                content.erase(pos, tag.size());
        }
    }

    return ContentDelta{std::move(content)};
}

}  // namespace ovms
