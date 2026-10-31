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
    const bool want_va = (device == "GPU") && imp_video_va_available();

    // ---- Read model to get input dimensions ---------------------------------
    ov::Core core;
    auto raw_model = core.read_model(model_path);
    auto orig_in = raw_model->input(0).get_shape();  // [1,3,H,W]
    int model_h = static_cast<int>(orig_in[2]);
    int model_w = static_cast<int>(orig_in[3]);
    std::cout << "Model input: " << model_w << "x" << model_h << "\n";

    // ---- Open video ---------------------------------------------------------
    // For VA surface path, output at model's resolution so vapostproc does the
    // resize on GPU — surfaces arrive at exactly the size OV expects.
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", video_path);
    if (want_va) {
        // Output at model's resolution so vapostproc resizes on GPU.
        imp_video_source_set(src, "width",  std::to_string(model_w).c_str());
        imp_video_source_set(src, "height", std::to_string(model_h).c_str());
    }

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

    // ---- Compile model ------------------------------------------------------
    ov::CompiledModel model;
    std::cout << "Inference device: " << device << "\n";

    // Read first frame early if VA path — we need GStreamer's native VADisplay.
    imp_tensor_t* first_tensor = nullptr;
    if (want_va) {
        st = imp_video_read_frame(&first_tensor, stream, 0);
        if (st != IMP_OK || !first_tensor) {
            std::cerr << "Failed to read first frame for VA display discovery\n";
            imp_video_close(stream);
            imp_context_destroy(ctx);
            return 1;
        }

        uint32_t surf_id = 0;
        void*    gst_va_disp = nullptr;
        int      fw = 0, fh = 0;
        imp_tensor_get_va_surface(first_tensor, &surf_id, &gst_va_disp, &fw, &fh);
        std::cerr << "First frame: surface=" << surf_id
                  << " va_display=" << gst_va_disp
                  << " " << fw << "x" << fh << "\n";

        if (!gst_va_disp) {
            std::cerr << "Could not get GStreamer's VADisplay from first frame\n";
            imp_video_close(stream);
            imp_context_destroy(ctx);
            return 1;
        }

        try {
            auto va_ctx = ov::intel_gpu::ocl::VAContext(core, gst_va_disp);

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

            model = core.compile_model(processed, va_ctx);
        } catch (const std::exception& e) {
            std::cerr << "VA model compile failed: " << e.what() << "\n";
            imp_video_close(stream);
            imp_context_destroy(ctx);
            return 1;
        }
        std::cout << "Model compiled with VA surface input (NV12 two-plane).\n";
    } else {
        try {
            model = core.compile_model(model_path, device);
        } catch (const std::exception& e) {
            std::cerr << "model compile failed: " << e.what() << "\n";
            imp_video_close(stream);
            imp_context_destroy(ctx);
            return 1;
        }
    }

    ov::InferRequest req = model.create_infer_request();

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

        // First frame was already read for VA display discovery.
        if (first_tensor) {
            tensor = first_tensor;
            first_tensor = nullptr;
        } else {
            auto t0 = Clock::now();
            st = imp_video_read_frame(&tensor, stream, 0);
            auto t1 = Clock::now();
            if (st == IMP_ERROR_STREAM_END) break;
            if (st != IMP_OK || !tensor) {
                std::cerr << "read_frame failed: " << st << "\n";
                break;
            }
            t_decode_ms += Ms(t1 - t0).count();
        }

        frame_count++;

        auto t2 = Clock::now();

        if (imp_tensor_get_memory_type(tensor) == IMP_MEM_VA_SURFACE) {
            // VA surface path: import surface directly into OV — zero copies.
            uint32_t surface_id = 0;
            void*    va_disp    = nullptr;
            int      fw = 0, fh = 0;
            imp_tensor_get_va_surface(tensor, &surface_id, &va_disp, &fw, &fh);

            if (frame_count <= 3)
                std::cerr << "  VA surface: id=" << surface_id
                          << " disp=" << va_disp
                          << " " << fw << "x" << fh << "\n";

            try {
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
            } catch (const std::exception& e) {
                std::cerr << "VA infer failed (frame " << frame_count
                          << ", surface=" << surface_id << "): " << e.what() << "\n";
                break;
            }

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

        imp_tensor_release(tensor);
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
