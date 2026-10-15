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

/**
 * GstDetectionOverlayCalculator — detection-box overlay (host NV12 or VA surface).
 *
 * Joins a full-res frame (FRAME_FULL) with the detections produced for the same
 * timestamp (DETECTION_BOXES) and draws green bounding boxes plus a confidence
 * bar, exactly like the reference POC's draw_detections_on_nv12. One calculator
 * serves both overlay paths, selected at runtime from the frame's memory type:
 *
 *   IMP_MEM_SYSTEM     — host NV12 frame: draw directly into the Y/UV planes
 *                        (the explicit GPU->CPU overlay path).
 *   IMP_MEM_VA_SURFACE — GPU VA surface: map the surface for CPU write with
 *                        vaDeriveImage (which aliases surface memory on Intel
 *                        GPUs), or fall back to vaCreateImage + vaGetImage/
 *                        vaPutImage, and draw in place — no host NV12 branch and
 *                        no full-frame download (P6a.2 zero-copy). Compiled only
 *                        when OVMS_ENABLE_VA_OVERLAY is defined (see BUILD).
 *
 * FRAME_FULL has a single consumer (this calculator), so the in-place write is
 * not observed by any other node. The frame is forwarded as FRAME_OVERLAID so
 * the sink encodes it (host NV12 or on the GPU via vah264lpenc).
 *
 * Inputs:
 *   FRAME_FULL      : GstVideoFramePacket — full-res frame (host NV12 or VA surface).
 *   DETECTION_BOXES : GstDetectionBoxes   — boxes in full-res pixels.
 * Output:
 *   FRAME_OVERLAID  : GstVideoFramePacket — annotated frame (same packet type).
 */

#include <cstdint>

#ifdef OVMS_ENABLE_VA_OVERLAY
#include <va/va.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/mediapipe_gst_calculators/gst_detection_box.hpp"
#include "src/mediapipe_gst_calculators/gst_nv12_overlay_draw.hpp"
#include "src/mediapipe_gst_calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kFrameFullTag[] = "FRAME_FULL";
constexpr char kDetectionBoxesTag[] = "DETECTION_BOXES";
constexpr char kFrameOverlaidTag[] = "FRAME_OVERLAID";
constexpr char kColorRTag[] = "COLOR_R";
constexpr char kColorGTag[] = "COLOR_G";
constexpr char kColorBTag[] = "COLOR_B";

uint8_t clampColorComponent(int value) {
    return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

nv12overlay::Color rgbToNv12(int red, int green, int blue) {
    const float r = static_cast<float>(clampColorComponent(red));
    const float g = static_cast<float>(clampColorComponent(green));
    const float b = static_cast<float>(clampColorComponent(blue));
    return {
        static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b),
        static_cast<uint8_t>(-0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f),
        static_cast<uint8_t>(0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f)};
}
}  // namespace

class GstDetectionOverlayCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kFrameFullTag).Set<GstVideoFramePacket>();
        cc->Inputs().Tag(kDetectionBoxesTag).Set<GstDetectionBoxes>();
        cc->Outputs().Tag(kFrameOverlaidTag).Set<GstVideoFramePacket>();
        if (cc->InputSidePackets().HasTag(kColorRTag))
            cc->InputSidePackets().Tag(kColorRTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kColorGTag))
            cc->InputSidePackets().Tag(kColorGTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kColorBTag))
            cc->InputSidePackets().Tag(kColorBTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        const bool hasRed = cc->InputSidePackets().HasTag(kColorRTag);
        const bool hasGreen = cc->InputSidePackets().HasTag(kColorGTag);
        const bool hasBlue = cc->InputSidePackets().HasTag(kColorBTag);
        RET_CHECK(hasRed == hasGreen && hasGreen == hasBlue)
            << "COLOR_R, COLOR_G, and COLOR_B must be provided together";
        if (hasRed) {
            color_ = rgbToNv12(cc->InputSidePackets().Tag(kColorRTag).Get<int>(),
                               cc->InputSidePackets().Tag(kColorGTag).Get<int>(),
                               cc->InputSidePackets().Tag(kColorBTag).Get<int>());
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        const auto& frame = cc->Inputs().Tag(kFrameFullTag).Get<GstVideoFramePacket>();
        const auto& boxes = cc->Inputs().Tag(kDetectionBoxesTag).Get<GstDetectionBoxes>();

        imp_tensor_t* tensor = frame.owner.get();
        RET_CHECK(tensor != nullptr) << "FRAME_FULL has no native tensor owner";

        const imp_tensor_memory_type_t mem = imp_tensor_get_memory_type(tensor);
        if (mem == IMP_MEM_SYSTEM) {
            absl::Status st = drawOnHost(tensor, boxes, color_);
            if (!st.ok())
                return st;
        } else if (mem == IMP_MEM_VA_SURFACE) {
#ifdef OVMS_ENABLE_VA_OVERLAY
            if (!boxes.empty()) {
                absl::Status st = drawOnSurface(tensor, boxes, color_);
                if (!st.ok())
                    return st;
            }
#else
            RET_CHECK(false) << "VA-surface overlay requires OVMS_ENABLE_VA_OVERLAY";
#endif
        } else {
            RET_CHECK(false) << "overlay supports host NV12 or VA-surface frames only";
        }

        cc->Outputs()
            .Tag(kFrameOverlaidTag)
            .AddPacket(cc->Inputs().Tag(kFrameFullTag).Value().At(cc->InputTimestamp()));
        return absl::OkStatus();
    }

 private:
    static absl::Status drawOnHost(imp_tensor_t* tensor, const GstDetectionBoxes& boxes,
                                   const nv12overlay::Color& color) {
        const uint8_t* y_const = nullptr;
        const uint8_t* uv_const = nullptr;
        int w = 0, h = 0;
        RET_CHECK_EQ(imp_tensor_get_nv12_planes(tensor, &y_const, &uv_const, &w, &h), IMP_OK);
        RET_CHECK_GT(w, 0);
        RET_CHECK_GT(h, 0);

        // Single-consumer FRAME_FULL: draw in place on the host planes.
        auto* y = const_cast<uint8_t*>(y_const);
        auto* uv = const_cast<uint8_t*>(uv_const);

        // Packed host NV12: both plane strides equal the width.
        nv12overlay::drawDetections(y, w, uv, w, w, h, boxes, color);
        return absl::OkStatus();
    }

#ifdef OVMS_ENABLE_VA_OVERLAY
    static absl::Status drawOnSurface(imp_tensor_t* tensor, const GstDetectionBoxes& boxes,
                                      const nv12overlay::Color& color) {
        uint32_t surface_id = 0;
        void* disp = nullptr;
        int w = 0, h = 0;
        RET_CHECK_EQ(imp_tensor_get_va_surface(tensor, &surface_id, &disp, &w, &h), IMP_OK);
        RET_CHECK(disp != nullptr) << "VA surface has no VADisplay";
        RET_CHECK_GT(w, 0);
        RET_CHECK_GT(h, 0);

        VADisplay va = reinterpret_cast<VADisplay>(disp);
        VASurfaceID surface = static_cast<VASurfaceID>(surface_id);
        vaSyncSurface(va, surface);

        // Map the surface memory in place; fall back to a copied image if the
        // driver will not derive a writable image for this surface.
        VAImage image{};
        bool derived = vaDeriveImage(va, surface, &image) == VA_STATUS_SUCCESS;
        if (!derived) {
            VAImageFormat fmt{};
            fmt.fourcc = VA_FOURCC_NV12;
            fmt.byte_order = VA_LSB_FIRST;
            fmt.bits_per_pixel = 12;
            RET_CHECK_EQ(vaCreateImage(va, &fmt, w, h, &image), VA_STATUS_SUCCESS)
                << "vaCreateImage(NV12) failed";
            if (vaGetImage(va, surface, 0, 0, w, h, image.image_id) != VA_STATUS_SUCCESS) {
                vaDestroyImage(va, image.image_id);
                return absl::InternalError("vaGetImage failed");
            }
        }

        void* buf = nullptr;
        if (vaMapBuffer(va, image.buf, &buf) != VA_STATUS_SUCCESS) {
            vaDestroyImage(va, image.image_id);
            return absl::InternalError("vaMapBuffer failed");
        }

        auto* base = static_cast<uint8_t*>(buf);
        uint8_t* y = base + image.offsets[0];
        uint8_t* uv = base + image.offsets[1];
        nv12overlay::drawDetections(y, image.pitches[0], uv, image.pitches[1], w, h, boxes, color);

        vaUnmapBuffer(va, image.buf);
        if (!derived)
            vaPutImage(va, surface, image.image_id, 0, 0, w, h, 0, 0, w, h);
        vaDestroyImage(va, image.image_id);
        return absl::OkStatus();
    }
#endif
    nv12overlay::Color color_ = nv12overlay::kGreen;
};

REGISTER_CALCULATOR(GstDetectionOverlayCalculator);

}  // namespace mediapipe
