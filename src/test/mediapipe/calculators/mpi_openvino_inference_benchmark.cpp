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
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <openvino/runtime/properties.hpp>
#include "src/mediapipe_gst_calculators/gst_vaapi_inference.hpp"
#endif
#include "mpi_inference_benchmark_utils.hpp"

namespace {
namespace bench = inference_benchmark;

bool e2DiagnosticsEnabled() {
    return bench::environment("OVMS_MPI_E2_DIAGNOSTICS", "off") == "stages";
}

double milliseconds(std::chrono::nanoseconds duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

#ifndef _WIN32
struct NativeVaDisplay {
    int descriptor = -1;
    VADisplay display = nullptr;
    bool initialized = false;
    bool registered = false;

    bool open() {
        const std::string renderNode = bench::environment("OVMS_PERF_RENDER_NODE", "/dev/dri/renderD128");
        descriptor = ::open(renderNode.c_str(), O_RDWR);
        if (descriptor < 0)
            return false;
        display = vaGetDisplayDRM(descriptor);
        if (!display)
            return false;
        int major = 0;
        int minor = 0;
        if (vaInitialize(display, &major, &minor) != VA_STATUS_SUCCESS) {
            display = nullptr;
            return false;
        }
        initialized = true;
        imp_video_set_va_display(display);
        registered = true;
        return true;
    }

    ~NativeVaDisplay() {
        if (registered)
            imp_video_set_va_display(nullptr);
        if (initialized)
            vaTerminate(display);
        if (descriptor >= 0)
            close(descriptor);
    }
};
#endif

TEST(MpiOpenVinoBenchmark, Direct) {
    std::vector<double> samples;
    const std::string modelPath = bench::modelPath();
    const std::string inputName = benchmark_config::kInputName;
    const std::string performanceHint = benchmark_config::kPerformanceHint;
    const uint32_t streams = std::stoi(benchmark_config::kNumStreams);
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
#ifdef _WIN32
        bench::DecodeStream decode;
        decode.open();
        ov::Core core;
        std::unique_ptr<ov::intel_gpu::ocl::D3DContext> context;
        ov::CompiledModel compiled;
        ov::InferRequest request;
#else
        const bool preinitializeModel = bench::environment("OVMS_MPI_E2_PREINITIALIZE_OV", "off") == "on";
        std::unique_ptr<NativeVaDisplay> preinitializedDisplay;
        mediapipe::GstVaapiInference inference;
        if (preinitializeModel) {
            preinitializedDisplay = std::make_unique<NativeVaDisplay>();
            ASSERT_TRUE(preinitializedDisplay->open()) << "VA display initialization failed before decode";
            try {
                inference.initialize(preinitializedDisplay->display, benchmark_config::kModelWidth,
                    benchmark_config::kModelHeight, modelPath, inputName, performanceHint, streams);
            } catch (const std::exception& error) {
                FAIL() << "OpenVINO pre-initialization failed: " << error.what();
            }
        }
        bench::DecodeStream decode;
        decode.open();
#endif
        int frames = 0;
#ifndef _WIN32
        const bool collectE2 = e2DiagnosticsEnabled();
    const bool synchronizeSurface = collectE2 &&
        bench::environment("OVMS_MPI_E2_SURFACE_SYNC", "off") == "on";
        std::chrono::nanoseconds readTime{};
        std::chrono::nanoseconds surfaceExtractionTime{};
    std::chrono::nanoseconds surfaceSyncTime{};
        std::chrono::nanoseconds tensorImportTime{};
        std::chrono::nanoseconds tensorBindingTime{};
        std::chrono::nanoseconds inferCallTime{};
        std::chrono::nanoseconds outputHandlingTime{};
        std::chrono::nanoseconds releaseTime{};
#endif
        bench::Clock::time_point start;
        bench::Clock::time_point stop;
        for (;;) {
            imp_tensor_t* raw = nullptr;
#ifndef _WIN32
            const auto readStart = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            const imp_status_t status = imp_video_read_frame(&raw, decode.stream, 0);
#ifndef _WIN32
            const auto readStop = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw, imp_tensor_release);
            if (status == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(status, IMP_OK);
            ASSERT_NE(tensor, nullptr);
#ifndef _WIN32
            const bool measuredFrame = frames >= benchmark_config::kWarmupFrames;
            if (collectE2 && measuredFrame)
                readTime += readStop - readStart;
#endif
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
            const auto surfaceStart = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
            ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
            const auto surfaceStop = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
            if (collectE2 && measuredFrame)
                surfaceExtractionTime += surfaceStop - surfaceStart;
            if (synchronizeSurface) {
                const auto syncStart = bench::Clock::now();
                ASSERT_EQ(vaSyncSurface(display, surface), VA_STATUS_SUCCESS);
                const auto syncStop = bench::Clock::now();
                if (measuredFrame)
                    surfaceSyncTime += syncStop - syncStart;
            }
            mediapipe::GstVaapiInferenceTimings inferenceTimings;
            const int detections = inference.infer(display, surface, width, height, modelPath,
                inputName, performanceHint, streams, collectE2 ? &inferenceTimings : nullptr);
            ASSERT_GE(detections,
                0);
            if (collectE2 && measuredFrame) {
                tensorImportTime += inferenceTimings.tensorImport;
                tensorBindingTime += inferenceTimings.tensorBinding;
                inferCallTime += inferenceTimings.inferCall;
                outputHandlingTime += inferenceTimings.outputHandling;
            }
#endif
#ifndef _WIN32
            const auto releaseStart = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            tensor.reset();
#ifndef _WIN32
            const auto releaseStop = collectE2 ? bench::Clock::now() : bench::Clock::time_point{};
            if (collectE2 && measuredFrame)
                releaseTime += releaseStop - releaseStart;
#endif
            ++frames;
            if (frames == benchmark_config::kWarmupFrames) {
                start = bench::Clock::now();
#ifndef _WIN32
                if (collectE2) {
                    std::string executionDevice = "unavailable";
                    std::string inferencePrecision = "unavailable";
                    try {
                        const auto devices = inference.compiledModel().get_property(ov::execution_devices);
                        if (!devices.empty())
                            executionDevice = devices.front();
                    } catch (const std::exception&) {
                    }
                    try {
                        inferencePrecision = inference.compiledModel().get_property(ov::hint::inference_precision).to_string();
                    } catch (const std::exception&) {
                    }
                    std::cout << "E2_CONFIG case=mpi_direct_ov_native device=" << executionDevice
                              << " inference_precision=" << inferencePrecision
                              << " performance_hint=" << performanceHint
                              << " num_streams=" << streams
                              << " preinitialize_openvino=" << (preinitializeModel ? "on" : "off")
                              << " sync_appsink=" << bench::environment("OVMS_MPI_E2_SYNC_APPSINK", "off")
                              << " explicit_va_sync=" << (synchronizeSurface ? "on" : "off")
                              << " request_count=1 preprocessing=NV12_to_BGR_NCHW"
                              << " output=detection_count warmup=" << benchmark_config::kWarmupFrames
                              << " measured=" << benchmark_config::kMeasuredFrames << std::endl;
                }
#endif
            }
            if (frames == bench::kTotalFrames)
                stop = bench::Clock::now();
        }
        ASSERT_EQ(frames, bench::kTotalFrames);
        samples.push_back(bench::milliseconds(start, stop) / benchmark_config::kMeasuredFrames);
#ifndef _WIN32
        if (collectE2) {
            const double scale = 1.0 / benchmark_config::kMeasuredFrames;
            const double wall = samples.back();
            const double stageSum = milliseconds(readTime + surfaceExtractionTime + surfaceSyncTime + tensorImportTime +
                tensorBindingTime + inferCallTime + outputHandlingTime + releaseTime) * scale;
            std::cout << std::fixed << std::setprecision(6)
                      << "E2_DIRECT sample=" << sample + 1
                      << " measured=" << benchmark_config::kMeasuredFrames
                      << " read_pull_ms_per_frame=" << milliseconds(readTime) * scale
                      << " surface_extract_ms_per_frame=" << milliseconds(surfaceExtractionTime) * scale
                      << " explicit_va_sync_ms_per_frame=" << milliseconds(surfaceSyncTime) * scale
                      << " va_tensor_import_ms_per_frame=" << milliseconds(tensorImportTime) * scale
                      << " tensor_binding_ms_per_frame=" << milliseconds(tensorBindingTime) * scale
                      << " infer_call_ms_per_frame=" << milliseconds(inferCallTime) * scale
                      << " output_handling_ms_per_frame=" << milliseconds(outputHandlingTime) * scale
                      << " tensor_release_ms_per_frame=" << milliseconds(releaseTime) * scale
                      << " completion_wall_ms_per_frame=" << wall
                      << " residual_ms_per_frame=" << wall - stageSum << std::endl;
        }
#endif
    }
    bench::report("mpi_direct_ov_native", samples);
}

#ifndef _WIN32
TEST(MpiOpenVinoDiagnostics, DecodeSupply) {
    if (!e2DiagnosticsEnabled())
        GTEST_SKIP() << "Set OVMS_MPI_E2_DIAGNOSTICS=stages for E2 decode supply timings";

    std::vector<double> samples;
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
        bench::DecodeStream decode;
        decode.open();
        int frames = 0;
        std::chrono::nanoseconds readTime{};
        std::chrono::nanoseconds releaseTime{};
        bench::Clock::time_point start;
        bench::Clock::time_point stop;
        for (;;) {
            imp_tensor_t* raw = nullptr;
            const auto readStart = bench::Clock::now();
            const imp_status_t status = imp_video_read_frame(&raw, decode.stream, 0);
            const auto readStop = bench::Clock::now();
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw, imp_tensor_release);
            if (status == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(status, IMP_OK);
            ASSERT_NE(tensor, nullptr);
            ASSERT_EQ(imp_tensor_get_memory_type(tensor.get()), IMP_MEM_VA_SURFACE);
            if (frames >= benchmark_config::kWarmupFrames)
                readTime += readStop - readStart;
            const auto releaseStart = bench::Clock::now();
            tensor.reset();
            const auto releaseStop = bench::Clock::now();
            if (frames >= benchmark_config::kWarmupFrames)
                releaseTime += releaseStop - releaseStart;
            ++frames;
            if (frames == benchmark_config::kWarmupFrames)
                start = bench::Clock::now();
            if (frames == bench::kTotalFrames)
                stop = bench::Clock::now();
        }
        ASSERT_EQ(frames, bench::kTotalFrames);
        const double wall = bench::milliseconds(start, stop) / benchmark_config::kMeasuredFrames;
        samples.push_back(wall);
        const double scale = 1.0 / benchmark_config::kMeasuredFrames;
        std::cout << std::fixed << std::setprecision(6)
                  << "E2_DECODE_SUPPLY sample=" << sample + 1
                  << " warmup=" << benchmark_config::kWarmupFrames
                  << " measured=" << benchmark_config::kMeasuredFrames
                  << " consumed=" << frames << " eos=1"
                  << " read_pull_ms_per_frame=" << milliseconds(readTime) * scale
                  << " tensor_release_ms_per_frame=" << milliseconds(releaseTime) * scale
                  << " completion_wall_ms_per_frame=" << wall << std::endl;
    }
    bench::report("mpi_va_decode_supply_only", samples);
}
#endif

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