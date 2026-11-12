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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/statusor.h"
#pragma warning(pop)

#include "openvino/runtime/tensor.hpp"

namespace ovms {

constexpr int64_t MAX_VIDEO_FRAMES = 1024;

// Calculates how many decoded frames can be retained after applying the
// administrator's FPS and maximum-frame settings to a client video source.
absl::StatusOr<size_t> calculateSelectedVideoFrameCount(uint64_t durationUnits,
    uint32_t timescale, uint32_t sampleCount, std::optional<float> requestedFps, uint32_t maxDecodedFrames);

// Returns whether reserving frameCount decoded frames fits the remaining pixel budget.
bool videoFrameAllocationFitsBudget(size_t framePixels, size_t frameCount,
    size_t totalAllocatedPixels, size_t maxAllowedPixels);

class VideoFrameSelector {
public:
    VideoFrameSelector(std::optional<long double> frameInterval, size_t maxSelectedFrames);
    bool shouldSelect(uint64_t timestamp);

private:
    const std::optional<long double> frameInterval;
    const size_t maxSelectedFrames{0};
    size_t selectedFrameCount{0};
    long double nextSelectedTimestamp{0.0L};
};

// Loads a sequence of video frames from a list of frame references (base64 data
// URIs, HTTP/HTTPS URLs, or local file paths) and stacks them into a single
// video tensor with [N, H, W, C] layout (RGB, u8), as expected by the GenAI VLM
// pipeline video input.
absl::StatusOr<ov::Tensor> loadVideoFrames(const std::vector<std::string>& frameSources,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    size_t& totalAllocatedPixels);

// Decodes an H.264/AVC MP4 source and samples output frames at requestedFps.
// The request is rejected when the estimated sampled frame count exceeds
// maxDecodedFrames
absl::StatusOr<ov::Tensor> decodeVideoSource(const std::string& videoSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    std::optional<float> requestedFps, uint32_t maxDecodedFrames, size_t& totalAllocatedPixels);

}  // namespace ovms
