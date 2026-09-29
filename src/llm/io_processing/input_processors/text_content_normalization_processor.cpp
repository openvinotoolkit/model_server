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

#include "text_content_normalization_processor.hpp"

#include <string>
#include <variant>

namespace ovms {

absl::Status TextContentNormalizationProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "TextContentNormalizationProcessor received input that is not a ChatHistory");
    }
    ov::genai::ChatHistory& chatHistory = std::get<ov::genai::ChatHistory>(req.input);
    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        if (!content.is_array()) {
            continue;
        }
        // Only flatten arrays that contain exclusively text parts. Arrays with
        // images (or other modalities) are left untouched for ImageDecodingProcessor.
        // Single pass: build the combined string while scanning, and bail out on the
        // first non-text part without touching the message.
        std::string combined;
        bool allText = true;
        for (size_t j = 0; j < content.size(); j++) {
            const auto part = content[j];
            if (part["type"].as_string().value_or("") != "text") {
                allText = false;
                break;
            }
            if (!combined.empty()) {
                combined += "\n";
            }
            combined += part["text"].as_string().value_or("");
        }
        if (!allText) {
            continue;
        }
        chatHistory[i]["content"] = combined;
    }
    return absl::OkStatus();
}

}  // namespace ovms
