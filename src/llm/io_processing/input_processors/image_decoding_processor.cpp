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

#include "image_decoding_processor.hpp"

#include <string>
#include <utility>
#include <variant>

#include "../../io_processing/image_utils.hpp"
#include "../../../logging.hpp"

namespace ovms {

ImageDecodingProcessor::ImageDecodingProcessor(
    std::optional<std::string> allowedLocalMediaPath,
    std::optional<std::vector<std::string>> allowedMediaDomains) :
    allowedLocalMediaPath(std::move(allowedLocalMediaPath)),
    allowedMediaDomains(std::move(allowedMediaDomains)) {}

absl::Status ImageDecodingProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "ImageDecodingProcessor received input that is not a ChatHistory");
    }
    auto& chatHistory = std::get<ov::genai::ChatHistory>(req.input);

    // Injection guard: reject requests that already contain image tags to
    // prevent prompt injection via pre-baked tags.
    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        // Check plain string content.
        if (content.as_string().value_or("").find("<ov_genai_image_") != std::string::npos) {
            return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
        }
        // Check text parts within array content (multimodal messages).
        if (content.is_array()) {
            for (size_t j = 0; j < content.size(); j++) {
                const auto part = content[j];
                if (part["type"].as_string().value_or("") == "text") {
                    if (part["text"].as_string().value_or("").find("<ov_genai_image_") != std::string::npos) {
                        return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
                    }
                }
            }
        }
    }

    size_t imageIndex = 0;
    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        if (!content.is_array()) {
            continue;
        }

        // Flatten multipart content in its original order, replacing each image
        // with the placeholder consumed by the VLM pipeline.
        std::string flattenedContent;
        bool previousPartWasText = false;

        for (size_t j = 0; j < content.size(); j++) {
            const auto part = content[j];
            const auto type = part["type"].as_string().value_or("");

            if (type == "image_url") {
                const auto url = part["image_url"]["url"].as_string().value_or("");
                auto imageResult = loadImage(url, allowedLocalMediaPath, allowedMediaDomains);
                if (!imageResult.ok()) {
                    return imageResult.status();
                }
                req.inputImages.push_back(std::move(imageResult).value());
                flattenedContent += "<ov_genai_image_" + std::to_string(imageIndex++) + ">\n";
                previousPartWasText = false;
            } else if (type == "text") {
                if (previousPartWasText) {
                    flattenedContent += "\n";
                }
                flattenedContent += part["text"].as_string().value_or("");
                previousPartWasText = true;
            }
        }

        if (!flattenedContent.empty()) {
            chatHistory[i]["content"] = flattenedContent;
        }
    }

    return absl::OkStatus();
}

}  // namespace ovms
