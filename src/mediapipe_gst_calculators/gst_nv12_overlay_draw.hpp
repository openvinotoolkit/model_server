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

// Shared NV12 bounding-box draw ported from the reference POC
// (draw_detections_on_nv12). Used by both the host-NV12 overlay and the
// VA-surface overlay, so the Y/UV plane writes live in one place. The plane
// pointers and strides let the same code target packed host memory (stride ==
// width) or a mapped VA image (driver-provided pitches).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "src/mediapipe_gst_calculators/gst_detection_box.hpp"

namespace mediapipe {
namespace nv12overlay {

struct Color {
    uint8_t y;
    uint8_t u;
    uint8_t v;
};

// NV12 colours ported from the reference POC.
constexpr Color kGreen{150, 44, 21};
constexpr uint8_t kWhiteY = 235, kWhiteU = 128, kWhiteV = 128;
constexpr int kBorderThickness = 3;

// Fill an inclusive rectangle in an NV12 frame. y_stride/uv_stride are the
// per-plane row pitches (== width for packed host NV12).
inline void fillRect(uint8_t* y, int y_stride, uint8_t* uv, int uv_stride,
                     int w, int h, int x1, int y1, int x2, int y2,
                     uint8_t fy, uint8_t fu, uint8_t fv) {
    x1 = std::clamp(x1, 0, w - 1);
    y1 = std::clamp(y1, 0, h - 1);
    x2 = std::clamp(x2, 0, w - 1);
    y2 = std::clamp(y2, 0, h - 1);
    if (x2 < x1 || y2 < y1)
        return;
    for (int row = y1; row <= y2; ++row)
        std::memset(&y[row * y_stride + x1], fy, static_cast<size_t>(x2 - x1 + 1));
    const int ux1 = x1 / 2, ux2 = x2 / 2;
    const int uy1 = y1 / 2, uy2 = y2 / 2;
    for (int row = uy1; row <= uy2; ++row) {
        for (int col = ux1; col <= ux2; ++col) {
            uv[row * uv_stride + col * 2] = fu;
            uv[row * uv_stride + col * 2 + 1] = fv;
        }
    }
}

// Draw green box borders (thickness 3) for every detection plus a confidence
// bar for the top box, matching the reference POC.
inline void drawDetections(uint8_t* y, int y_stride, uint8_t* uv, int uv_stride,
                           int w, int h, const GstDetectionBoxes& boxes,
                           const Color& color = kGreen) {
    if (boxes.empty())
        return;
    for (const auto& d : boxes) {
        const int x1 = std::clamp(static_cast<int>(d.xmin), 0, w - 1);
        const int y1 = std::clamp(static_cast<int>(d.ymin), 0, h - 1);
        const int x2 = std::clamp(static_cast<int>(d.xmax), 0, w - 1);
        const int y2 = std::clamp(static_cast<int>(d.ymax), 0, h - 1);
        const int t = kBorderThickness;
        fillRect(y, y_stride, uv, uv_stride, w, h, x1, y1, x2, std::min(y1 + t - 1, h - 1), color.y, color.u, color.v);
        fillRect(y, y_stride, uv, uv_stride, w, h, x1, std::max(y2 - t + 1, 0), x2, y2, color.y, color.u, color.v);
        fillRect(y, y_stride, uv, uv_stride, w, h, x1, y1, std::min(x1 + t - 1, w - 1), y2, color.y, color.u, color.v);
        fillRect(y, y_stride, uv, uv_stride, w, h, std::max(x2 - t + 1, 0), y1, x2, y2, color.y, color.u, color.v);
    }

    const float best = boxes.front().score;
    if (best > 0.1f) {
        const int bar_w = static_cast<int>(best * 200.0f);
        fillRect(y, y_stride, uv, uv_stride, w, h, 4, 4, 208, 16, 16, 128, 128);
        fillRect(y, y_stride, uv, uv_stride, w, h, 4, 4, 4 + bar_w, 16, color.y, color.u, color.v);
        fillRect(y, y_stride, uv, uv_stride, w, h, 3, 3, 209, 3, kWhiteY, kWhiteU, kWhiteV);
        fillRect(y, y_stride, uv, uv_stride, w, h, 3, 17, 209, 17, kWhiteY, kWhiteU, kWhiteV);
        fillRect(y, y_stride, uv, uv_stride, w, h, 3, 3, 3, 17, kWhiteY, kWhiteU, kWhiteV);
        fillRect(y, y_stride, uv, uv_stride, w, h, 209, 3, 209, 17, kWhiteY, kWhiteU, kWhiteV);
    }
}

}  // namespace nv12overlay
}  // namespace mediapipe
