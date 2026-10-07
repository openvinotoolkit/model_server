// Copyright (c) 2026 Intel Corporation
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

// P1 packet contract (see current_task/P1_PacketContract.md). Video-source
// specific by design. Lives alongside the calculators for now; promote at P4.

#pragma once

#include <cstdint>
#include <memory>
#include <variant>

namespace mediapipe {

struct VaSurfaceFrame {
    uint32_t surface_id = 0;
};

struct D3D11Frame {
    uintptr_t texture_handle = 0;
    uint32_t subresource = 0;
};

struct CpuNv12Frame {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    int y_stride = 0;
    int uv_stride = 0;
};

using GstVideoFrameResource = std::variant<VaSurfaceFrame, D3D11Frame, CpuNv12Frame>;

// The native frame resource stays alive through owner until all packet consumers finish.
struct GstVideoFramePacket {
    GstVideoFrameResource resource = VaSurfaceFrame{};
    std::shared_ptr<void> owner;
    int width = 0;
    int height = 0;
};

}  // namespace mediapipe
