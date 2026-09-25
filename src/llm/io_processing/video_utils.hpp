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

// Total decoded-byte budget for a single video (1 GiB). Frames are stacked into
// one {N, H, W, C} tensor in loadVideoFrames() at their original resolution
// before any downstream sampling happens: the GenAI VLM pipeline performs fps
// sampling and temporal frame merging only afterwards, so the server must 
// first hold every frame the client sends. This cap bounds that allocation. 
// It is derived from the tightest supported deployment (a 32 GB host running 
// a ~20 GB int4 35B model, leaving ~12 GB). The process-level peak is roughly
// 2x this buffer: downstream, GenAI's sample_video_if_needed() copies the
// selected frames into a newly allocated tensor while the original stacked
// buffer is still alive as the copy source, so both briefly coexist (this is in
// GenAI, not in loadVideoFrames(), which itself copies incrementally). A 1 GiB
// budget therefore keeps that peak near 2 GB, while still covering realistic
// inputs (~692 frames at 540p, ~388 at 720p, ~172 at 1080p). Adjust for the
// target deployment memory.
constexpr int64_t MAX_VIDEO_DECODED_BYTES = 1LL * 1024 * 1024 * 1024;

// Loads a sequence of video frames from a list of frame references (base64 data
// URIs, HTTP/HTTPS URLs, or local file paths) and stacks them into a single
// video tensor with [N, H, W, C] layout (RGB, u8), as expected by the GenAI VLM
// pipeline video input. All frames must share the same height, width and channel
// count. The mp4 decoding is intentionally not performed here; frames are
// expected to be already decoded on the client side.
absl::StatusOr<ov::Tensor> loadVideoFrames(const std::vector<std::string>& frameSources,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains);

}  // namespace ovms
