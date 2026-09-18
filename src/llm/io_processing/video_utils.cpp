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

#include "video_utils.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "image_utils.hpp"
#include "../../logging.hpp"

namespace ovms {

absl::StatusOr<ov::Tensor> loadVideoFrames(const std::vector<std::string>& frameSources,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains) {
    if (frameSources.empty()) {
        return absl::InvalidArgumentError("Video must contain at least one frame");
    }
    if (static_cast<int64_t>(frameSources.size()) > MAX_VIDEO_FRAMES) {
        return absl::InvalidArgumentError("Number of video frames exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_FRAMES));
    }

    std::vector<ov::Tensor> frames;
    frames.reserve(frameSources.size());
    ov::Shape frameShape;
    for (size_t i = 0; i < frameSources.size(); i++) {
        auto frameResult = loadImage(frameSources[i], allowedLocalMediaPath, allowedMediaDomains);
        if (!frameResult.ok()) {
            return frameResult.status();
        }
        const ov::Tensor& frame = frameResult.value();
        // loadImage returns [1, H, W, C]; all frames must share the same H, W, C.
        if (i == 0) {
            frameShape = frame.get_shape();
        } else if (frame.get_shape() != frameShape) {
            return absl::InvalidArgumentError("All video frames must have the same height, width and channel count");
        }
        frames.push_back(frame);
    }

    const size_t numFrames = frames.size();
    const size_t height = frameShape[1];
    const size_t width = frameShape[2];
    const size_t channels = frameShape[3];
    ov::Tensor video(frames[0].get_element_type(), ov::Shape{numFrames, height, width, channels});

    const size_t frameBytes = height * width * channels * video.get_element_type().size();
    auto* dst = static_cast<uint8_t*>(video.data());
    for (size_t i = 0; i < numFrames; i++) {
        std::memcpy(dst + i * frameBytes, frames[i].data(), frameBytes);
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Loaded video with {} frames of shape [{}, {}, {}]", numFrames, height, width, channels);
    return video;
}

}  // namespace ovms
