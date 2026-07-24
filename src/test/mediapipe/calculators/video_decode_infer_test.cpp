// Copyright (c) 2025 Intel Corporation
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
 * video_decode_infer_test — standalone end-to-end smoke test.
 *
 * Usage:
 *   ./video_decode_infer_test <video_file> <model_xml>
 *
 * Returns exit code 0 if at least one detection found across all frames,
 * non-zero otherwise.
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#include "src/mpi/intel_mpi.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
// ============================================================================
// NV12 → BGR float32 (nearest-neighbour resize, BT.601 limited range)
// Output: NCHW [3 × H × W] float, values [0,1]
// ============================================================================
static void nv12_to_bgr_float_nchw(const uint8_t* y_plane,
                                    const uint8_t* uv_plane,
                                    int src_w, int src_h,
                                    int dst_w, int dst_h,
                                    std::vector<float>& out_nchw) {
    out_nchw.resize(static_cast<size_t>(3) * dst_h * dst_w);

    const float scale_x = static_cast<float>(src_w) / dst_w;
    const float scale_y = static_cast<float>(src_h) / dst_h;

    // Pointers to B/G/R planes in NCHW output
    float* B = out_nchw.data();
    float* G = B + static_cast<size_t>(dst_h) * dst_w;
    float* R = G + static_cast<size_t>(dst_h) * dst_w;

    for (int dy = 0; dy < dst_h; dy++) {
        int sy = static_cast<int>(dy * scale_y);
        if (sy >= src_h) sy = src_h - 1;
        int uv_row = (sy / 2) * src_w;

        for (int dx = 0; dx < dst_w; dx++) {
            int sx = static_cast<int>(dx * scale_x);
            if (sx >= src_w) sx = src_w - 1;

            float Y = static_cast<float>(y_plane[sy * src_w + sx]) - 16.0f;

            int uv_col = (sx / 2) * 2;
            float U = static_cast<float>(uv_plane[uv_row + uv_col])     - 128.0f;
            float V = static_cast<float>(uv_plane[uv_row + uv_col + 1]) - 128.0f;

            float r = (1.164f * Y + 1.596f * V);
            float g = (1.164f * Y - 0.392f * U - 0.813f * V);
            float b = (1.164f * Y + 2.017f * U);

            auto clamp255 = [](float v) { return v < 0.f ? 0.f : (v > 255.f ? 255.f : v); };

            size_t pix = static_cast<size_t>(dy) * dst_w + dx;
            B[pix] = clamp255(b);
            G[pix] = clamp255(g);
            R[pix] = clamp255(r);
        }
    }
}
#pragma GCC diagnostic pop

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file> <model_xml> [threshold] [infer_device]\n";
        return 2;
    }
    const char* video_path   = argv[1];
    const char* model_path   = argv[2];
    const float threshold    = (argc > 3) ? std::stof(argv[3]) : 0.5f;
    const std::string device = (argc > 4) ? argv[4] : "CPU";

    // ---- Open standalone context -----------------------------------------
    imp_context_t* ctx = nullptr;
    imp_status_t st = imp_context_create(&ctx, nullptr);
    if (st != IMP_OK) {
        std::cerr << "imp_context_create failed: " << st << "\n";
        return 1;
    }

    // ---- Decide whether to request VA surface memory ----------------------
    // VA surface mode is used when GPU decode is available AND inference runs
    // on GPU: the frame never leaves GPU memory — zero host copies.
    const bool want_va = (device == "GPU") && imp_video_va_available();

    // ---- Load + compile model -----------------------------------------------
    // Model must be compiled BEFORE video is opened so that when want_va=true
    // we can extract OV's internal VADisplay and inject it into gst_loader.
    // This ensures GStreamer VA surfaces are valid in OV's GPU context.
    ov::Core core;
    ov::CompiledModel model;

    std::cout << "Inference device: " << device << "\n";

    if (want_va) {
        // VA surface path: let OV create the GPU context first, then compile
        // with NV12 surface preprocessing. We extract OV's internal VADisplay
        // and inject it into gst_loader so both GStreamer and OV share one VA
        // context — required for valid VASurfaceID cross-context access.
        auto raw_model = core.read_model(model_path);

        ov::preprocess::PrePostProcessor ppp(raw_model);
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES,
                              {"y", "uv"})
            .set_memory_type(ov::intel_gpu::memory_type::surface);
        ppp.input()
            .preprocess()
            .convert_color(ov::preprocess::ColorFormat::BGR)
            .convert_element_type(ov::element::f32);
        ppp.input().model().set_layout("NCHW");
        auto processed = ppp.build();

        // Compile once to get OV's own VADisplay.
        auto tmp_compiled = core.compile_model(processed, "GPU");
        auto ov_va_ctx = tmp_compiled.get_context().as<ov::intel_gpu::ocl::VAContext>();
        VADisplay ov_va_disp = ov_va_ctx;

        // Override gst_loader's VADisplay BEFORE opening the video stream so
        // GStreamer VA surfaces are allocated in the same VA context as OV.
        imp_video_set_va_display(ov_va_disp);

        // Re-compile explicitly with OV's VAContext so subsequent
        // create_tensor_nv12 calls share the same device.
        auto va_ctx_build = ov::intel_gpu::ocl::VAContext(core, ov_va_disp);
        try {
            model = core.compile_model(processed, va_ctx_build);
        } catch (const std::exception& e) {
            std::cerr << "model compile (VA) failed: " << e.what() << "\n";
            imp_context_destroy(ctx);
            return 1;
        }
        std::cout << "Model compiled with VA surface input (NV12 two-plane). "
                     "Shared VADisplay=" << (void*)ov_va_disp << "\n";
    } else {
        // System-memory path: standard compile.
        try {
            model = core.compile_model(model_path, device);
        } catch (const std::exception& e) {
            std::cerr << "model compile failed: " << e.what() << "\n";
            imp_context_destroy(ctx);
            return 1;
        }
    }

    // ---- Open video ---------------------------------------------------------
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", video_path);

    imp_video_decode_opts_t vopts{};
    vopts.use_va_surface_memory = want_va;

    imp_video_stream_t* stream = nullptr;
    st = imp_video_open(&stream, src, ctx, &vopts);
    imp_video_source_destroy(src);

    if (st != IMP_OK) {
        std::cerr << "imp_video_open failed: " << st
                  << "  (" << imp_context_get_error(ctx) << ")\n";
        imp_context_destroy(ctx);
        return 1;
    }

    uint32_t frame_w = 0, frame_h = 0;
    imp_video_get_info(stream, &frame_w, &frame_h, nullptr, nullptr);
    std::cout << "Video opened: " << frame_w << "x" << frame_h << "\n";

    auto in_shape = model.input(0).get_shape();  // [1,3,H,W] or [1,1,H,W] per plane for NV12
    // For NV12 VA surface input the model has two inputs (y, uv) — use output shape for sizing.
    int model_h = 0, model_w = 0;
    if (want_va) {
        // Input 0 is the Y plane [1,1,H,W]; resize happens inside OV.
        // The model's first input is the full-res source — use the original model input size.
        auto out_shape = model.output(0).get_shape();
        // fall back: read from original model to get H,W
        auto orig = core.read_model(model_path);
        auto orig_in = orig->input(0).get_shape();
        model_h = static_cast<int>(orig_in[2]);
        model_w = static_cast<int>(orig_in[3]);
    } else {
        model_h = static_cast<int>(in_shape[2]);
        model_w = static_cast<int>(in_shape[3]);
    }
    std::cout << "Model input: " << model_w << "x" << model_h << "\n";

    ov::InferRequest req = model.create_infer_request();

    // VA context for surface import (only used in VA path)
    std::optional<ov::intel_gpu::ocl::VAContext> va_ctx_infer;
    if (want_va) {
        va_ctx_infer = model.get_context().as<ov::intel_gpu::ocl::VAContext>();
    }

    // ---- Decode + infer loop ------------------------------------------------
    int frame_count      = 0;
    int total_detections = 0;
    std::vector<float> nchw_buf;
    double t_decode_ms = 0, t_preproc_ms = 0, t_infer_ms = 0;

    for (;;) {
        imp_tensor_t* tensor = nullptr;
        auto t0 = Clock::now();
        st = imp_video_read_frame(&tensor, stream, 0);
        auto t1 = Clock::now();
        if (st == IMP_ERROR_STREAM_END) break;
        if (st != IMP_OK || !tensor) {
            std::cerr << "read_frame failed: " << st << "\n";
            break;
        }

        frame_count++;
        t_decode_ms += Ms(t1 - t0).count();

        auto t2 = Clock::now();

        if (imp_tensor_get_memory_type(tensor) == IMP_MEM_VA_SURFACE) {
            // VA surface path: import surface directly into OV — zero copies.
            uint32_t surface_id = 0;
            void*    va_disp    = nullptr;
            int      fw = 0, fh = 0;
            imp_tensor_get_va_surface(tensor, &surface_id, &va_disp, &fw, &fh);

            auto nv12 = va_ctx_infer->create_tensor_nv12(
                static_cast<size_t>(fh),
                static_cast<size_t>(fw),
                static_cast<uint32_t>(surface_id));

            // NV12_TWO_PLANES: set y input (index 0) and uv input (index 1).
            req.set_input_tensor(0, nv12.first);
            req.set_input_tensor(1, nv12.second);

            auto t3 = Clock::now();
            t_preproc_ms += Ms(t3 - t2).count();  // essentially zero

            req.infer();
            auto t4 = Clock::now();
            t_infer_ms += Ms(t4 - t3).count();

        } else {
            // System-memory path: CPU NV12→BGR conversion then infer.
            const uint8_t* y_ptr  = nullptr;
            const uint8_t* uv_ptr = nullptr;
            int fw = 0, fh = 0;
            imp_tensor_get_nv12_planes(tensor, &y_ptr, &uv_ptr, &fw, &fh);

            nv12_to_bgr_float_nchw(y_ptr, uv_ptr, fw, fh, model_w, model_h, nchw_buf);
            auto t3 = Clock::now();
            t_preproc_ms += Ms(t3 - t2).count();

            ov::Tensor input_tensor(ov::element::f32,
                                    {1, 3,
                                     static_cast<size_t>(model_h),
                                     static_cast<size_t>(model_w)},
                                    nchw_buf.data());
            req.set_input_tensor(input_tensor);
            req.infer();
            auto t4 = Clock::now();
            t_infer_ms += Ms(t4 - t3).count();
        }

        auto out = req.get_output_tensor(0);
        const float* det  = out.data<const float>();
        size_t num_dets   = out.get_shape()[2];

        int frame_dets = 0;
        for (size_t i = 0; i < num_dets; i++) {
            float conf = det[i * 7 + 2];
            if (conf > threshold) frame_dets++;
        }
        total_detections += frame_dets;

        if (frame_count <= 5 || frame_count % 50 == 0)
            std::cout << "  frame " << frame_count
                      << "  detections=" << frame_dets << "\n";
    }

    double t_total_ms = t_decode_ms + t_preproc_ms + t_infer_ms;
    std::cout << "Done: " << frame_count << " frames, "
              << "total detections=" << total_detections << "\n";
    std::cout << "\nPer-frame averages (ms):\n"
              << "  decode      : " << t_decode_ms  / frame_count << "\n"
              << "  preprocess  : " << t_preproc_ms / frame_count << "\n"
              << "  infer       : " << t_infer_ms   / frame_count << "\n"
              << "  total timed : " << t_total_ms   / frame_count << "\n"
              << "  FPS (timed) : " << 1000.0 * frame_count / t_total_ms << "\n";

    imp_video_close(stream);
    imp_context_destroy(ctx);

    if (total_detections < 1) {
        std::cerr << "FAIL: expected at least 1 detection\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
