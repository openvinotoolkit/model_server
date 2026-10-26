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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <gtest/gtest.h>
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
#include <cstring>
#include <va/va.h>
#include "src/mediapipe_gst_calculators/gst_vaapi_inference.hpp"
#endif
#include "mpi_inference_benchmark_utils.hpp"

namespace {
namespace bench = inference_benchmark;

TEST(MpiOpenVinoBenchmark, Direct) {
    std::vector<double> samples;
    const std::string modelPath = bench::modelPath();
    const std::string inputName = benchmark_config::kInputName;
    const std::string performanceHint = benchmark_config::kPerformanceHint;
    const uint32_t streams = std::stoi(benchmark_config::kNumStreams);
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
        bench::DecodeStream decode;
        decode.open();
#ifdef _WIN32
        ov::Core core;
        std::unique_ptr<ov::intel_gpu::ocl::D3DContext> context;
        ov::CompiledModel compiled;
        ov::InferRequest request;
#else
        mediapipe::GstVaapiInference inference;
#endif
        int frames = 0;
        bench::Clock::time_point start;
        bench::Clock::time_point stop;
        for (;;) {
            imp_tensor_t* raw = nullptr;
            const imp_status_t status = imp_video_read_frame(&raw, decode.stream, 0);
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw, imp_tensor_release);
            if (status == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(status, IMP_OK);
            ASSERT_NE(tensor, nullptr);
            int width = 0;
            int height = 0;
#ifdef _WIN32
            void* textureHandle = nullptr;
            void* device = nullptr;
            uint32_t subresource = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor.get(), &textureHandle, &device,
                          &subresource, &width, &height),
                IMP_OK);
            ASSERT_EQ(subresource, 0u);
            if (!context) {
                context = std::make_unique<ov::intel_gpu::ocl::D3DContext>(core, static_cast<ID3D11Device*>(device));
                compiled = core.compile_model(bench::model(core, true), *context, bench::properties());
                request = compiled.create_infer_request();
            }
            auto planes = context->create_tensor_nv12(height, width, static_cast<ID3D11Texture2D*>(textureHandle));
            request.set_tensor(bench::kYInputName, planes.first);
            request.set_tensor(bench::kUvInputName, planes.second);
            request.infer();
            ASSERT_GT(request.get_output_tensor().get_size(), 0);
#else
            uint32_t surface = 0;
            void* display = nullptr;
            ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
            ASSERT_GE(inference.infer(display, surface, width, height, modelPath,
                          inputName, performanceHint, streams),
                0);
#endif
            tensor.reset();
            ++frames;
            if (frames == benchmark_config::kWarmupFrames)
                start = bench::Clock::now();
            if (frames == bench::kTotalFrames)
                stop = bench::Clock::now();
        }
        ASSERT_EQ(frames, bench::kTotalFrames);
        samples.push_back(bench::milliseconds(start, stop) / benchmark_config::kMeasuredFrames);
    }
    bench::report("mpi_direct_ov_native", samples);
}

TEST(MpiOpenVinoBenchmark, OneCalculator) { bench::measureGraph("mpi_ov_one_calculator", false, false); }
TEST(MpiOpenVinoBenchmark, TwoCalculators) { bench::measureGraph("mpi_ov_two_calculators", false, true); }

TEST(MpiOpenVinoIntegration, DecodeEosCleanup) {
    for (int iteration = 0; iteration < 2; ++iteration) {
        bench::DecodeStream decode;
        decode.open();
        int frames = 0;
        for (;;) {
            imp_tensor_t* raw = nullptr;
            const imp_status_t status = imp_video_read_frame(&raw, decode.stream, 0);
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw, imp_tensor_release);
            if (status == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(status, IMP_OK);
            ASSERT_NE(tensor, nullptr);
#ifdef _WIN32
            ASSERT_EQ(imp_tensor_get_memory_type(tensor.get()), IMP_MEM_D3D11_SURFACE);
#else
            ASSERT_EQ(imp_tensor_get_memory_type(tensor.get()), IMP_MEM_VA_SURFACE);
#endif
            ++frames;
        }
        ASSERT_EQ(frames, bench::kTotalFrames);
    }
}

TEST(MpiOpenVinoIntegration, CalculatorConstruction) {
    for (bool split : {false, true}) {
        mediapipe::CalculatorGraph graph;
        const auto status = graph.Initialize(bench::graphConfig(false, split));
        ASSERT_TRUE(status.ok()) << status;
    }
}

#ifndef _WIN32
TEST(MpiOpenVinoIntegration, NativePixelsMatchPureFixture) {
    const auto expected = bench::frames();
    bench::DecodeStream decode;
    decode.open();
    for (int frame = 0; frame < bench::kTotalFrames; ++frame) {
        SCOPED_TRACE(frame);
        imp_tensor_t* raw = nullptr;
        ASSERT_EQ(imp_video_read_frame(&raw, decode.stream, 0), IMP_OK);
        std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw, imp_tensor_release);
        uint32_t surface = 0;
        void* display = nullptr;
        int width = 0;
        int height = 0;
        ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
        ASSERT_EQ(width, benchmark_config::kModelWidth);
        ASSERT_EQ(height, benchmark_config::kModelHeight);
        ASSERT_EQ(vaSyncSurface(display, surface), VA_STATUS_SUCCESS);
        struct Image {
            VADisplay display;
            VAImage image{};
            void* data = nullptr;
            bool derived = false;
            ~Image() {
                if (data)
                    vaUnmapBuffer(display, image.buf);
                if (derived)
                    vaDestroyImage(display, image.image_id);
            }
        } image{display};
        ASSERT_EQ(vaDeriveImage(display, surface, &image.image), VA_STATUS_SUCCESS);
        image.derived = true;
        ASSERT_EQ(image.image.format.fourcc, VA_FOURCC_NV12);
        ASSERT_EQ(vaMapBuffer(display, image.image.buf, &image.data), VA_STATUS_SUCCESS);
        for (int plane = 0; plane < 2; ++plane) {
            const int rows = plane == 0 ? height : height / 2;
            const size_t offset = plane == 0 ? 0 : static_cast<size_t>(width) * height;
            for (int row = 0; row < rows; ++row) {
                ASSERT_EQ(std::memcmp(static_cast<const uint8_t*>(image.data) + image.image.offsets[plane] +
                                          row * image.image.pitches[plane],
                              expected[frame].data() + offset + row * width, width),
                    0)
                    << "plane=" << plane << " row=" << row;
            }
        }
    }
    imp_tensor_t* extra = nullptr;
    EXPECT_EQ(imp_video_read_frame(&extra, decode.stream, 0), IMP_ERROR_STREAM_END);
    if (extra)
        imp_tensor_release(extra);
}
#endif
}  // namespace