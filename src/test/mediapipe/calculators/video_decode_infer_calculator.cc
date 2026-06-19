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
 * VideoDecodeInferCalculator — GStreamer integration smoke test.
 *
 * Reads a video file using Intel MPI (imp_* API), decodes frames to NV12,
 * converts to BGR float32, and runs inference with face-detection-adas-0001
 * via OpenVINO directly (not through an OVMS model instance).
 *
 * face-detection-adas-0001 expects: [1, 3, 384, 672]  BGR NCHW FP32
 *
 * Side-packet inputs:
 *   VIDEO_PATH  : std::string — path to input video file
 *   MODEL_PATH  : std::string — path to face-detection-adas-0001.xml
 *
 * Output stream:
 *   DETECTIONS  : int — total detections above threshold across all frames
 */

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"

namespace mediapipe {

// ============================================================================
// NV12 → BGR float32 conversion (plain C, no opencv dependency)
//
// Output: contiguous BGR row-major buffer [H×W×3] float32, values [0,1]
// ============================================================================
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void nv12_to_bgr_float(const uint8_t* y_plane, const uint8_t* uv_plane,
                               int src_w, int src_h,
                               int dst_w, int dst_h,
                               std::vector<float>& out_bgr) {
    // Simple nearest-neighbour resize + NV12→BGR
    out_bgr.resize(static_cast<size_t>(dst_h) * dst_w * 3);

    const float scale_x = static_cast<float>(src_w) / dst_w;
    const float scale_y = static_cast<float>(src_h) / dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        int sy = static_cast<int>(dy * scale_y);
        if (sy >= src_h) sy = src_h - 1;

        int uv_row = (sy / 2) * src_w;  // UV plane row offset

        for (int dx = 0; dx < dst_w; dx++) {
            int sx = static_cast<int>(dx * scale_x);
            if (sx >= src_w) sx = src_w - 1;

            // Y sample
            float Y = static_cast<float>(y_plane[sy * src_w + sx]) - 16.0f;

            // UV sample (interleaved: U at even index, V at odd)
            int uv_col = (sx / 2) * 2;
            float U = static_cast<float>(uv_plane[uv_row + uv_col])     - 128.0f;
            float V = static_cast<float>(uv_plane[uv_row + uv_col + 1]) - 128.0f;

            // BT.601 limited range
            float r = 1.164f * Y + 1.596f * V;
            float g = 1.164f * Y - 0.392f * U - 0.813f * V;
            float b = 1.164f * Y + 2.017f * U;

            // Clamp and normalize to [0,1]
            auto clamp01 = [](float v) { return v < 0.0f ? 0.0f : (v > 255.0f ? 1.0f : v / 255.0f); };

            size_t idx = static_cast<size_t>((dy * dst_w + dx) * 3);
            out_bgr[idx + 0] = clamp01(b);
            out_bgr[idx + 1] = clamp01(g);
            out_bgr[idx + 2] = clamp01(r);
        }
    }
}
#pragma GCC diagnostic pop

// ============================================================================
// Calculator
// ============================================================================

class VideoDecodeInferCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag("VIDEO_PATH").Set<std::string>();
        cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
        cc->Outputs().Tag("DETECTIONS").Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        video_path_ = cc->InputSidePackets().Tag("VIDEO_PATH").Get<std::string>();
        model_path_ = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        int total_detections = run_decode_infer();

        cc->Outputs()
            .Tag("DETECTIONS")
            .AddPacket(MakePacket<int>(total_detections)
                           .At(cc->InputTimestamp()));
        return absl::OkStatus();
    }

 private:
    std::string video_path_;
    std::string model_path_;

    // face-detection-adas-0001 input geometry
    static constexpr int kModelH = 384;
    static constexpr int kModelW = 672;

    int run_decode_infer() {
        // ----------------------------------------------------------------
        // 1. Open standalone context (CPU, no GPU model needed)
        // ----------------------------------------------------------------
        imp_context_t* ctx = nullptr;
        imp_status_t st = imp_context_create(&ctx, nullptr);
        if (st != IMP_OK) {
            LOG(ERROR) << "[VideoDecodeInferCalculator] imp_context_create failed: " << st;
            return -1;
        }

        // ----------------------------------------------------------------
        // 2. Open video — source dimensions come from the file
        // ----------------------------------------------------------------
        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, IMP_SOURCE_FILE);
        imp_video_source_set(src, "path", video_path_.c_str());

        imp_video_stream_t* stream = nullptr;
        st = imp_video_open(&stream, src, ctx, nullptr);
        imp_video_source_destroy(src);

        if (st != IMP_OK) {
            LOG(ERROR) << "[VideoDecodeInferCalculator] imp_video_open failed: " << st
                       << "  ctx_error: " << imp_context_get_error(ctx);
            imp_context_destroy(ctx);
            return -1;
        }

        uint32_t frame_w = 0, frame_h = 0;
        imp_video_get_info(stream, &frame_w, &frame_h, nullptr, nullptr);
        LOG(INFO) << "[VideoDecodeInferCalculator] Video: " << frame_w << "x" << frame_h;

        // ----------------------------------------------------------------
        // 3. Load face-detection-adas-0001 — expects [1,3,384,672] BGR FP32
        // ----------------------------------------------------------------
        ov::Core core;
        ov::CompiledModel model;
        try {
            model = core.compile_model(model_path_, "CPU");
        } catch (const std::exception& e) {
            LOG(ERROR) << "[VideoDecodeInferCalculator] model load failed: " << e.what();
            imp_video_close(stream);
            imp_context_destroy(ctx);
            return -1;
        }

        ov::InferRequest req = model.create_infer_request();
        // Get actual model input shape
        auto in_shape = model.input(0).get_shape();  // [1,3,H,W]
        int model_h = static_cast<int>(in_shape[2]);
        int model_w = static_cast<int>(in_shape[3]);
        LOG(INFO) << "[VideoDecodeInferCalculator] Model input: "
                  << model_w << "x" << model_h;

        // ----------------------------------------------------------------
        // 4. Decode loop
        // ----------------------------------------------------------------
        int frame_count      = 0;
        int total_detections = 0;
        const float detection_threshold = 0.5f;

        std::vector<float> bgr_buf;

        for (;;) {
            imp_tensor_t* tensor = nullptr;
            st = imp_video_read_frame(&tensor, stream, 0);
            if (st == IMP_ERROR_STREAM_END) break;
            if (st != IMP_OK || !tensor) break;

            frame_count++;

            // Get NV12 plane pointers via public API (tensor struct is opaque)
            const uint8_t* y_ptr  = nullptr;
            const uint8_t* uv_ptr = nullptr;
            int fw = 0, fh = 0;
            if (imp_tensor_get_nv12_planes(tensor, &y_ptr, &uv_ptr, &fw, &fh) != IMP_OK)
                continue;

            // NV12 → BGR float, resize to model input
            nv12_to_bgr_float(y_ptr, uv_ptr,
                               fw, fh,
                               model_w, model_h,
                               bgr_buf);

            // NHWC → NCHW transpose
            std::vector<float> nchw(bgr_buf.size());
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < model_h; y++)
                    for (int x = 0; x < model_w; x++)
                        nchw[c * model_h * model_w + y * model_w + x] =
                            bgr_buf[(y * model_w + x) * 3 + c];

            ov::Shape shape = {1, 3,
                               static_cast<size_t>(model_h),
                               static_cast<size_t>(model_w)};
            ov::Tensor input_tensor(ov::element::f32, shape, nchw.data());
            req.set_input_tensor(input_tensor);
            req.infer();

            // Output: [1,1,N,7]  columns: [img_id, label, conf, x1,y1,x2,y2]
            auto out = req.get_output_tensor(0);
            const float* det = out.data<const float>();
            size_t num_dets = out.get_shape()[2];

            int frame_dets = 0;
            for (size_t i = 0; i < num_dets; i++) {
                float conf = det[i * 7 + 2];
                if (conf > detection_threshold) frame_dets++;
            }
            total_detections += frame_dets;

            if (frame_count <= 5 || frame_count % 50 == 0)
                LOG(INFO) << "[VideoDecodeInferCalculator] frame " << frame_count
                          << " detections=" << frame_dets;
        }

        LOG(INFO) << "[VideoDecodeInferCalculator] Done: " << frame_count
                  << " frames, total detections=" << total_detections;

        imp_video_close(stream);
        imp_context_destroy(ctx);
        return total_detections;
    }
};

REGISTER_CALCULATOR(VideoDecodeInferCalculator);

}  // namespace mediapipe
