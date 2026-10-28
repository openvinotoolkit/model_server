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

#include "video_frames_processor.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <openvino/genai/json_container.hpp>

#include "../video_utils.hpp"
#include "../../../logging.hpp"

namespace ovms {

namespace {
constexpr std::string_view VIDEO_TAG_PREFIX = "<ov_genai_video_";
}  // namespace

VideoFramesProcessor::VideoFramesProcessor(
    std::optional<std::string> allowedLocalMediaPath,
    std::optional<std::vector<std::string>> allowedMediaDomains) :
    allowedLocalMediaPath(std::move(allowedLocalMediaPath)),
    allowedMediaDomains(std::move(allowedMediaDomains)) {}

absl::Status VideoFramesProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "VideoFramesProcessor received input that is not a ChatHistory");
    }
    auto& chatHistory = std::get<ov::genai::ChatHistory>(req.input);

    // Injection guard: reject reserved video tags in any client-controlled
    // container the chat template renders (messages, tools, extra context).
    const auto& messages = chatHistory.get_messages();
    const auto& tools = chatHistory.get_tools();
    const auto& extraContext = chatHistory.get_extra_context();
    if (messages.to_json_string().find(VIDEO_TAG_PREFIX) != std::string::npos ||
        (!tools.empty() && tools.to_json_string().find(VIDEO_TAG_PREFIX) != std::string::npos) ||
        (!extraContext.empty() && extraContext.to_json_string().find(VIDEO_TAG_PREFIX) != std::string::npos)) {
        return absl::InvalidArgumentError("Message contains restricted <ov_genai_video> tag");
    }

    size_t videoIndex = 0;
    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        if (!content.is_array()) {
            continue;
        }

        for (size_t j = 0; j < content.size(); j++) {
            const auto part = content[j];
            if (part["type"].as_string().value_or("") == "video_url") {
                const auto urls = part["video_url"]["url"];
                // Reject oversized frame arrays before allocating from an untrusted length.
                if (static_cast<int64_t>(urls.size()) > MAX_VIDEO_FRAMES) {
                    return absl::InvalidArgumentError("Number of video frames exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_FRAMES));
                }
                std::vector<std::string> frameSources;
                frameSources.reserve(urls.size());
                for (size_t k = 0; k < urls.size(); k++) {
                    frameSources.push_back(urls[k].as_string().value_or(""));
                }
                auto videoResult = loadVideoFrames(frameSources, allowedLocalMediaPath, allowedMediaDomains, req.totalDecodedPixels);
                if (!videoResult.ok()) {
                    return videoResult.status();
                }
                req.inputVideos.push_back(std::move(videoResult).value());
                std::string tag = std::string(VIDEO_TAG_PREFIX) + std::to_string(videoIndex++) + ">";
                ov::genai::JsonContainer textEntry({{"type", "text"}, {"text", tag}});
                content[j] = textEntry;
            }
        }
    }

    return absl::OkStatus();
}

}  // namespace ovms
