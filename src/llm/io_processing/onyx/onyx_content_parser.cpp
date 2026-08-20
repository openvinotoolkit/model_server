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

#include "onyx_content_parser.hpp"

namespace ovms {

namespace {

void eraseTags(std::string& content, const std::vector<std::string>& tags) {
    for (const auto& tag : tags) {
        size_t pos = 0;
        while ((pos = content.find(tag, pos)) != std::string::npos)
            content.erase(pos, tag.size());
    }
}

static const std::vector<std::string> HOLD_TAGS = {
    "<|eom|>", "<|start|>assistant ", "<|message|>", "<|eot|>"};

static const std::vector<std::string> ROUTING_TAGS = {" to=user", "to=user"};

}  // namespace

OutputParsingConfig OnyxContentParser::defaultParsingConfig() {
    OutputParsingConfig cfg;
    // Resume signal: detected in TOOL_CALLS_WAITING_FOR_TOOL to switch back to content.
    cfg.startTags = {"to=user<|message|>"};
    cfg.needsSpecialTokens = true;
    return cfg;
}

OnyxContentParser::OnyxContentParser(ov::genai::Tokenizer& tokenizer,
    std::optional<OutputParsingConfig> configOverride) :
    BaseOutputParser(tokenizer,
        configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

std::optional<Delta> OnyxContentParser::parseChunk(
    const std::string& buffer,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {

    // Check structural (hold-eligible) tags.
    bool anyComplete = false;
    for (const auto& tag : HOLD_TAGS) {
        if (buffer.find(tag) != std::string::npos) {
            anyComplete = true;
        } else if (stringsOverlap(buffer, tag)) {
            return std::nullopt;  // partial match — hold until tag is fully assembled
        }
    }

    std::string content = buffer;
    if (anyComplete)
        eraseTags(content, HOLD_TAGS);
    eraseTags(content, ROUTING_TAGS);

    return ContentDelta{std::move(content)};
}

}  // namespace ovms
