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

// Shared detection geometry passed between the inference and overlay
// calculators of the P6a.1 decode+infer -> overlay -> encode pipeline.
// Box coordinates are in full-res (overlay) pixels, matching the reference POC.

#pragma once

#include <vector>

namespace mediapipe {

struct GstDetectionBox {
    float score = 0.0f;
    float xmin = 0.0f;
    float ymin = 0.0f;
    float xmax = 0.0f;
    float ymax = 0.0f;
};

using GstDetectionBoxes = std::vector<GstDetectionBox>;

}  // namespace mediapipe
