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

// Native frame handle owner. Opaque here; fully defined in src/mpi/intel_mpi.h.
// Forward-declared so consumers that only retain ownership (e.g. the inference
// calculator) don't pull in the whole C API. The source constructs the
// shared_ptr with imp_tensor_release as its deleter, so an incomplete type is
// sufficient everywhere the packet is used.
extern "C" {
typedef struct imp_tensor_s imp_tensor_t;
}

namespace mediapipe {

// resource: non-owning native handle snapshot. The inference calculators read
// it to bind VA/D3D11 surfaces without depending on //src/mpi.
struct VaSurfaceFrame {
    uint32_t surface_id = 0;
    void* display = nullptr;
};

struct D3D11Frame {
    uintptr_t texture_handle = 0;
    uint32_t subresource = 0;
};

// Host-memory discriminant only: sink/overlay read planes from owner via
// imp_tensor_get_nv12_planes(), so these fields are currently unused.
// TODO @atobiszei: confirm these fields are used with CPU frames
struct CpuNv12Frame {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    int y_stride = 0;
    int uv_stride = 0;
};

using GstVideoFrameResource = std::variant<VaSurfaceFrame, D3D11Frame, CpuNv12Frame>;

// owner owns the decoded frame (shared_ptr with imp_tensor_release) and is the
// canonical data accessor for overlay/sink/encode. resource is a non-owning
// handle/tag for the inference calculators and must not outlive owner.
struct GstVideoFramePacket {
    GstVideoFrameResource resource = VaSurfaceFrame{};
    std::shared_ptr<imp_tensor_t> owner;
    int width = 0;
    int height = 0;
};

}  // namespace mediapipe
