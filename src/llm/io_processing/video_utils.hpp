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
