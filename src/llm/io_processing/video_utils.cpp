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
#include "../../predict_request_validation_utils_impl.hpp"

namespace ovms {

absl::StatusOr<ov::Tensor> loadVideoFrames(const std::vector<std::string>& frameSources,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    size_t& totalAllocatedPixels) {
    if (frameSources.empty()) {
        return absl::InvalidArgumentError("Video must contain at least one frame");
    }
    if (static_cast<int64_t>(frameSources.size()) > MAX_VIDEO_FRAMES) {
        return absl::InvalidArgumentError("Number of video frames exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_FRAMES));
    }
    const size_t numFrames = frameSources.size();

    // Per-request decoded-pixel budget shared across all frames and across all
    // image_url/video_url parts of the request (totalAllocatedPixels is owned by
    // the caller): fetchAndDecodeImage rejects a frame before allocating its pixel
    // buffer once the running total would exceed the configured limit, bounding the
    // whole request's decoded size rather than giving each video a fresh budget.
    const size_t maxAllowedImagePixels = request_validation_utils::getMaxImageDecodePixels();

    // Decode the first frame to determine the shared frame shape. All frames
    // must have the same [1, H, W, C], so this shape (times numFrames) gives the
    // exact size of the stacked video tensor up front.
    auto firstResult = fetchAndDecodeImage(frameSources[0], allowedLocalMediaPath, allowedMediaDomains, totalAllocatedPixels, maxAllowedImagePixels);
    if (!firstResult.ok()) {
        return firstResult.status();
    }
    ov::Tensor firstFrame = firstResult.value();
    const ov::Shape frameShape = firstFrame.get_shape();
    const size_t height = frameShape[1];
    const size_t width = frameShape[2];
    const size_t channels = frameShape[3];
    const size_t frameBytes = height * width * channels * firstFrame.get_element_type().size();

    // Validate the aggregate decoded-pixel budget before allocating the stacked
    // tensor. All frames share frameShape (enforced during the copy loop below),
    // so framePixels is exact; the first frame is already counted in
    // totalAllocatedPixels, leaving numFrames - 1 frames to account for. Without
    // this up-front check the full {numFrames, H, W, C} tensor would be allocated
    // before the per-frame budget checks run, letting a request with many frame
    // references allocate a multi-GB tensor only to fail on a later frame. The
    // comparison uses division to stay overflow-safe.
    const size_t framePixels = height * width;
    const size_t remainingPixels = maxAllowedImagePixels - totalAllocatedPixels;
    if (numFrames > 1 && framePixels > remainingPixels / (numFrames - 1)) {
        return absl::InvalidArgumentError("Image exceeds maximum decoded size");
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
        auto frameResult = fetchAndDecodeImage(frameSources[i], allowedLocalMediaPath, allowedMediaDomains, totalAllocatedPixels, maxAllowedImagePixels);
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
