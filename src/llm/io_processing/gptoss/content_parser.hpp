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
#pragma once

#include <cstddef>
#include <optional>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "../base_output_parser.hpp"

namespace ovms {

class GptOssContentParser final : public BaseOutputParser {
    enum class StreamState : int {
        READING_HEADER,
        READING_BODY,
    };

    enum class ParseProgress : int {
        ADVANCED,
        NEED_MORE_INPUT,
    };

    StreamState streamState = StreamState::READING_HEADER;
    std::string pendingInput;
    std::size_t processedChunkSize = 0;
    bool visibleMessage = false;

    bool shouldEmitBody(const std::string& header) const;
    void appendNewChunk(const std::string& chunk);
    ParseProgress consumeHeader(std::string& content);
    ParseProgress consumeBody(std::string& content, bool& consumedMessage);

public:
    GptOssContentParser() = delete;

    static OutputParsingConfig defaultParsingConfig();

    explicit GptOssContentParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt);

    void resetState() override;

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;
};

}  // namespace ovms
