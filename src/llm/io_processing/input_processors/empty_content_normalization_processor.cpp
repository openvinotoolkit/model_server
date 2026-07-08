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

#include "empty_content_normalization_processor.hpp"

#include <variant>

namespace ovms {

absl::Status EmptyContentNormalizationProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "EmptyContentNormalizationProcessor received input that is not a ChatHistory");
    }
    ov::genai::ChatHistory& chatHistory = std::get<ov::genai::ChatHistory>(req.input);
    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        if (content.is_array() && content.size() == 0) {
            chatHistory[i]["content"] = ov::genai::JsonContainer(nullptr);
        }
    }
    return absl::OkStatus();
}

}  // namespace ovms
