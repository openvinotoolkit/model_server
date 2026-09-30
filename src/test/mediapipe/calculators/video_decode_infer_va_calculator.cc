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
 * VideoDecodeInferVACalculator — P0 monolithic decode+infer calculator.
 *
 * Proves the GStreamer VA zero-copy path works inside the MediaPipe framework.
 * The whole decode -> VA surface -> create_tensor_nv12 -> infer loop runs inside
 * a single Process() call, so no cross-packet surface ownership is needed yet
 * (that is deferred to P2). This lifts the GPU branch of the standalone proof
 * (video_decode_infer_test.cpp) almost verbatim.
 *
 * Trigger:
 *   TICK        : input stream, one packet drives the whole run.
 *
 * Side packets:
 *   VIDEO_PATH  : std::string — path to input video file.
 *   MODEL_PATH  : std::string — path to face-detection-adas-0001.xml.
 *   DEVICE      : std::string — "GPU" (default, VA zero-copy) or "CPU" (fallback).
 *
 * Outputs (each emitted only if wired in the graph):
 *   DETECTIONS   : int         — total detections above threshold across frames.
 *   INFER_OUTPUT : ov::Tensor  — host copy of the last frame's raw model output.
 */

#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"

namespace mediapipe {

namespace {
constexpr char kTickTag[] = "TICK";
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kModelPathTag[] = "MODEL_PATH";
constexpr char kDeviceTag[] = "DEVICE";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kInferOutputTag[] = "INFER_OUTPUT";

// NV12 -> BGR float32, NCHW [3 x H x W], BT.601 limited range, nearest resize.
// CPU fallback path only.
void nv12_to_bgr_float_nchw(const uint8_t* y_plane, const uint8_t* uv_plane,
                            int src_w, int src_h, int dst_w, int dst_h,
                            std::vector<float>& out_nchw) {
    out_nchw.resize(static_cast<size_t>(3) * dst_h * dst_w);
    const float scale_x = static_cast<float>(src_w) / dst_w;
    const float scale_y = static_cast<float>(src_h) / dst_h;
    float* B = out_nchw.data();
    float* G = B + static_cast<size_t>(dst_h) * dst_w;
    float* R = G + static_cast<size_t>(dst_h) * dst_w;

    for (int dy = 0; dy < dst_h; dy++) {
        int sy = static_cast<int>(dy * scale_y);
        if (sy >= src_h)
            sy = src_h - 1;
        int uv_row = (sy / 2) * src_w;
        for (int dx = 0; dx < dst_w; dx++) {
            int sx = static_cast<int>(dx * scale_x);
            if (sx >= src_w)
                sx = src_w - 1;
            float Y = static_cast<float>(y_plane[sy * src_w + sx]) - 16.0f;
            int uv_col = (sx / 2) * 2;
            float U = static_cast<float>(uv_plane[uv_row + uv_col]) - 128.0f;
            float V = static_cast<float>(uv_plane[uv_row + uv_col + 1]) - 128.0f;
            float r = 1.164f * Y + 1.596f * V;
            float g = 1.164f * Y - 0.392f * U - 0.813f * V;
            float b = 1.164f * Y + 2.017f * U;
            auto clamp255 = [](float v) { return v < 0.f ? 0.f : (v > 255.f ? 255.f : v); };
            size_t pix = static_cast<size_t>(dy) * dst_w + dx;
            B[pix] = clamp255(b);
            G[pix] = clamp255(g);
            R[pix] = clamp255(r);
        }
    }
}
}  // namespace

class VideoDecodeInferVACalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kTickTag).SetAny();
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kModelPathTag).Set<std::string>();
        if (cc->InputSidePackets().HasTag(kDeviceTag)) {
            cc->InputSidePackets().Tag(kDeviceTag).Set<std::string>();
        }
        if (cc->Outputs().HasTag(kDetectionsTag)) {
            cc->Outputs().Tag(kDetectionsTag).Set<int>();
        }
        if (cc->Outputs().HasTag(kInferOutputTag)) {
            cc->Outputs().Tag(kInferOutputTag).Set<ov::Tensor>();
        }
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        video_path_ = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        model_path_ = cc->InputSidePackets().Tag(kModelPathTag).Get<std::string>();
        if (cc->InputSidePackets().HasTag(kDeviceTag)) {
            device_ = cc->InputSidePackets().Tag(kDeviceTag).Get<std::string>();
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        ov::Tensor last_output;
        const bool want_output = cc->Outputs().HasTag(kInferOutputTag);
        int total_detections = run_decode_infer(want_output, last_output);
        if (total_detections < 0) {
            return absl::InternalError("VideoDecodeInferVACalculator: decode/infer failed");
        }
        if (cc->Outputs().HasTag(kDetectionsTag)) {
            cc->Outputs()
                .Tag(kDetectionsTag)
                .AddPacket(MakePacket<int>(total_detections).At(cc->InputTimestamp()));
        }
        if (cc->Outputs().HasTag(kInferOutputTag) && last_output) {
            cc->Outputs()
                .Tag(kInferOutputTag)
                .AddPacket(MakePacket<ov::Tensor>(last_output).At(cc->InputTimestamp()));
        }
        return absl::OkStatus();
    }

 private:
    std::string video_path_;
    std::string model_path_;
    std::string device_ = "GPU";
    static constexpr float kDetectionThreshold = 0.5f;

    // Returns total detections, or -1 on failure. When want_output is set, fills
    // last_output with a host copy of the last frame's raw model output.
    int run_decode_infer(bool want_output, ov::Tensor& last_output) {
        const bool want_va = (device_ == "GPU") && imp_video_va_available();

        ov::Core core;
        std::shared_ptr<ov::Model> raw_model;
        try {
            raw_model = core.read_model(model_path_);
        } catch (const std::exception& e) {
            LOG(ERROR) << "[VideoDecodeInferVACalculator] read_model failed: " << e.what();
            return -1;
        }
        auto in_shape = raw_model->input(0).get_shape();  // [1,3,H,W]
        int model_h = static_cast<int>(in_shape[2]);
        int model_w = static_cast<int>(in_shape[3]);

        imp_context_t* ctx = nullptr;
        if (imp_context_create(&ctx, nullptr) != IMP_OK) {
            LOG(ERROR) << "[VideoDecodeInferVACalculator] imp_context_create failed";
            return -1;
        }

        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, IMP_SOURCE_FILE);
        imp_video_source_set(src, "path", video_path_.c_str());
        if (want_va) {
            // Size decode output to model input so vapostproc resizes on GPU.
            imp_video_source_set(src, "width", std::to_string(model_w).c_str());
            imp_video_source_set(src, "height", std::to_string(model_h).c_str());
        }

        imp_video_decode_opts_t vopts{};
        vopts.use_va_surface_memory = want_va;

        imp_video_stream_t* stream = nullptr;
        imp_status_t st = imp_video_open(&stream, src, ctx, &vopts);
        imp_video_source_destroy(src);
        if (st != IMP_OK) {
            LOG(ERROR) << "[VideoDecodeInferVACalculator] imp_video_open failed: " << st
                       << " (" << imp_context_get_error(ctx) << ")";
            imp_context_destroy(ctx);
            return -1;
        }

        // For the VA path we need GStreamer's native VADisplay, which only exists
        // after the first frame is decoded — read it early and reuse it below.
        imp_tensor_t* first_tensor = nullptr;
        ov::CompiledModel model;
        if (want_va) {
            st = imp_video_read_frame(&first_tensor, stream, 0);
            if (st != IMP_OK || !first_tensor) {
                LOG(ERROR) << "[VideoDecodeInferVACalculator] first VA frame read failed: " << st;
                imp_video_close(stream);
                imp_context_destroy(ctx);
                return -1;
            }
            uint32_t surf_id = 0;
            void* gst_va_disp = nullptr;
            int fw = 0, fh = 0;
            imp_tensor_get_va_surface(first_tensor, &surf_id, &gst_va_disp, &fw, &fh);
            if (!gst_va_disp) {
                LOG(ERROR) << "[VideoDecodeInferVACalculator] no VADisplay from first frame";
                imp_video_close(stream);
                imp_context_destroy(ctx);
                return -1;
            }
            try {
                auto va_ctx = ov::intel_gpu::ocl::VAContext(core, gst_va_disp);
                ov::preprocess::PrePostProcessor ppp(raw_model);
                ppp.input()
                    .tensor()
                    .set_element_type(ov::element::u8)
                    .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                    .set_memory_type(ov::intel_gpu::memory_type::surface);
                ppp.input()
                    .preprocess()
                    .convert_color(ov::preprocess::ColorFormat::BGR)
                    .convert_element_type(ov::element::f32);
                ppp.input().model().set_layout("NCHW");
                model = core.compile_model(ppp.build(), va_ctx);
            } catch (const std::exception& e) {
                LOG(ERROR) << "[VideoDecodeInferVACalculator] VA compile failed: " << e.what();
                imp_video_close(stream);
                imp_context_destroy(ctx);
                return -1;
            }
        } else {
            try {
                model = core.compile_model(model_path_, device_);
            } catch (const std::exception& e) {
                LOG(ERROR) << "[VideoDecodeInferVACalculator] compile failed: " << e.what();
                imp_video_close(stream);
                imp_context_destroy(ctx);
                return -1;
            }
        }

        ov::InferRequest req = model.create_infer_request();
        std::optional<ov::intel_gpu::ocl::VAContext> va_ctx_infer;
        if (want_va) {
            va_ctx_infer = model.get_context().as<ov::intel_gpu::ocl::VAContext>();
        }

        int frame_count = 0;
        int total_detections = 0;
        std::vector<float> nchw_buf;

        using Clock = std::chrono::high_resolution_clock;
        using Ms = std::chrono::duration<double, std::milli>;
        double t_decode_ms = 0, t_preproc_ms = 0, t_infer_ms = 0;

        for (;;) {
            imp_tensor_t* tensor = nullptr;
            if (first_tensor) {
                tensor = first_tensor;
                first_tensor = nullptr;
            } else {
                auto t0 = Clock::now();
                st = imp_video_read_frame(&tensor, stream, 0);
                auto t1 = Clock::now();
                if (st == IMP_ERROR_STREAM_END)
                    break;
                if (st != IMP_OK || !tensor)
                    break;
                t_decode_ms += Ms(t1 - t0).count();
            }
            frame_count++;

            try {
                auto tp0 = Clock::now();
                if (imp_tensor_get_memory_type(tensor) == IMP_MEM_VA_SURFACE) {
                    uint32_t surface_id = 0;
                    void* va_disp = nullptr;
                    int fw = 0, fh = 0;
                    imp_tensor_get_va_surface(tensor, &surface_id, &va_disp, &fw, &fh);
                    auto nv12 = va_ctx_infer->create_tensor_nv12(
                        static_cast<size_t>(fh), static_cast<size_t>(fw), surface_id);
                    req.set_input_tensor(0, nv12.first);
                    req.set_input_tensor(1, nv12.second);
                    auto tp1 = Clock::now();
                    t_preproc_ms += Ms(tp1 - tp0).count();
                    req.infer();
                    t_infer_ms += Ms(Clock::now() - tp1).count();
                } else {
                    const uint8_t* y_ptr = nullptr;
                    const uint8_t* uv_ptr = nullptr;
                    int fw = 0, fh = 0;
                    imp_tensor_get_nv12_planes(tensor, &y_ptr, &uv_ptr, &fw, &fh);
                    nv12_to_bgr_float_nchw(y_ptr, uv_ptr, fw, fh, model_w, model_h, nchw_buf);
                    ov::Tensor input_tensor(ov::element::f32,
                                            {1, 3, static_cast<size_t>(model_h),
                                             static_cast<size_t>(model_w)},
                                            nchw_buf.data());
                    req.set_input_tensor(input_tensor);
                    auto tp1 = Clock::now();
                    t_preproc_ms += Ms(tp1 - tp0).count();
                    req.infer();
                    t_infer_ms += Ms(Clock::now() - tp1).count();
                }
            } catch (const std::exception& e) {
                LOG(ERROR) << "[VideoDecodeInferVACalculator] infer failed (frame "
                           << frame_count << "): " << e.what();
                imp_tensor_release(tensor);
                imp_video_close(stream);
                imp_context_destroy(ctx);
                return -1;
            }

            auto out = req.get_output_tensor(0);  // [1,1,N,7]
            const float* det = out.data<const float>();
            size_t num_dets = out.get_shape()[2];
            for (size_t i = 0; i < num_dets; i++) {
                if (det[i * 7 + 2] > kDetectionThreshold)
                    total_detections++;
            }
            if (want_output) {
                // Host copy of the last output for the optional INFER_OUTPUT stream.
                last_output = ov::Tensor(out.get_element_type(), out.get_shape());
                std::memcpy(last_output.data(), out.data(), out.get_byte_size());
            }
            imp_tensor_release(tensor);
        }

        double t_total_ms = t_decode_ms + t_preproc_ms + t_infer_ms;
        LOG(INFO) << "[VideoDecodeInferVACalculator] device=" << device_
                  << " va=" << want_va << " frames=" << frame_count
                  << " total_detections=" << total_detections;
        if (frame_count > 0) {
            LOG(INFO) << "[VideoDecodeInferVACalculator] per-frame ms:"
                      << " decode=" << t_decode_ms / frame_count
                      << " preprocess=" << t_preproc_ms / frame_count
                      << " infer=" << t_infer_ms / frame_count
                      << " total=" << t_total_ms / frame_count
                      << " FPS=" << 1000.0 * frame_count / t_total_ms;
        }

        imp_video_close(stream);
        imp_context_destroy(ctx);
        return total_detections;
    }
};

REGISTER_CALCULATOR(VideoDecodeInferVACalculator);

}  // namespace mediapipe
