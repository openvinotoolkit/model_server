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
    const size_t numFrames = frameSources.size();

    // Decode the first frame to determine the shared frame shape. All frames
    // must have the same [1, H, W, C], so this shape (times numFrames) gives the
    // exact size of the stacked video tensor up front.
    auto firstResult = loadImage(frameSources[0], allowedLocalMediaPath, allowedMediaDomains);
    if (!firstResult.ok()) {
        return firstResult.status();
    }
    ov::Tensor firstFrame = firstResult.value();
    const ov::Shape frameShape = firstFrame.get_shape();
    const size_t height = frameShape[1];
    const size_t width = frameShape[2];
    const size_t channels = frameShape[3];
    const size_t frameBytes = height * width * channels * firstFrame.get_element_type().size();

    // Predictive byte-budget check: reject before allocating the stacked tensor
    // if the total decoded size would exceed the budget. This bounds memory even
    // when the frame count is within MAX_VIDEO_FRAMES but the resolution is large.
    const int64_t totalBytes = static_cast<int64_t>(frameBytes) * static_cast<int64_t>(numFrames);
    if (totalBytes > MAX_VIDEO_DECODED_BYTES) {
        return absl::InvalidArgumentError("Total decoded video size exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_DECODED_BYTES) + " bytes");
    }

    // Allocate the stacked tensor once and copy each frame into it incrementally,
    // releasing the per-frame tensor right after. This keeps the peak memory at
    // roughly the stacked tensor plus a single frame, instead of holding all
    // decoded frames plus the stacked copy simultaneously.
    ov::Tensor video(firstFrame.get_element_type(), ov::Shape{numFrames, height, width, channels});
    auto* dst = static_cast<uint8_t*>(video.data());
    std::memcpy(dst, firstFrame.data(), frameBytes);
    firstFrame = ov::Tensor();  // release the first frame

    for (size_t i = 1; i < numFrames; i++) {
        auto frameResult = loadImage(frameSources[i], allowedLocalMediaPath, allowedMediaDomains);
        if (!frameResult.ok()) {
            return frameResult.status();
        }
        const ov::Tensor& frame = frameResult.value();
        // Validate the shape before the copy: memcpy below relies on every frame
        // being exactly frameBytes; a mismatching frame would otherwise over-read.
        if (frame.get_shape() != frameShape) {
            return absl::InvalidArgumentError("All video frames must have the same height, width and channel count");
        }
        std::memcpy(dst + i * frameBytes, frame.data(), frameBytes);
        // frame is released as it goes out of scope on the next iteration.
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Loaded video with {} frames of shape [{}, {}, {}]", numFrames, height, width, channels);
    return video;
}

}  // namespace ovms
