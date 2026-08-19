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
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"

#include "base_output_parser.hpp"

namespace ovms {

// Generic content parser: strips model-specific structural tokens or any other strings that should not be included in the final content output.
// (e.g. BOS/EOS for minicpm5, chat-template turn markers for gemma4/lfm2).
// Parsers that need richer hold logic (e.g. Onyx) provide their own content parser subclass.
class DefaultContentParser final : public BaseOutputParser {
public:
    DefaultContentParser() = delete;
    explicit DefaultContentParser(ov::genai::Tokenizer& tokenizer,
        std::vector<std::string> stringsToErase = {});

    std::optional<rapidjson::Document> parseChunk(const std::string& buffer,
        const std::vector<int64_t>& tokens,
        ov::genai::GenerationFinishReason finishReason) override;
};

}  // namespace ovms
