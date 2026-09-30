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
#include <vector>

#include <openvino/openvino.hpp>

namespace mediapipe {

// One decoded video frame flowing on a MediaPipe stream.
//
// The surface backing `planes` is kept alive by `owner` — a type-erased handle
// (typically shared_ptr<imp_tensor_t> with imp_tensor_release as its deleter).
// Because MediaPipe packets are shared, the surface survives until inference and
// every consumer have released the packet. Each frame owns its own surface, so
// distinct in-flight packets reference distinct, simultaneously valid surfaces.
struct GstVideoFramePacket {
    std::vector<ov::RemoteTensor> planes;  // NV12 GPU: [Y, UV]
    std::shared_ptr<void> owner;           // keeps the backing surface alive
    uint32_t va_surface_id = 0;            // VA surface id (verification/debug)
};

}  // namespace mediapipe
