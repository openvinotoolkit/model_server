
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

#include <cstring>
#include <memory>
#include <gtest/gtest.h>
#include <CL/cl2.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>
#include <openvino/runtime/profiling_info.hpp>
#include <openvino/runtime/properties.hpp>
#endif
#include "inference_benchmark_utils.hpp"

namespace {
namespace bench = inference_benchmark;
constexpr size_t kWidth = benchmark_config::kModelWidth;
constexpr size_t kHeight = benchmark_config::kModelHeight;
const ov::Shape kYShape{1, kHeight, kWidth, 1};
const ov::Shape kUvShape{1, kHeight / 2, kWidth / 2, 2};

#ifndef _WIN32
std::string diagnosticMode() {
    return bench::environment("OVMS_PERF_DIAGNOSTICS", "off");
}

bool collectStageTimings() {
    const std::string mode = diagnosticMode();
    return mode == "stages" || mode == "profile";
}

bool collectNodeProfiles() {
    return diagnosticMode() == "profile";
}

ov::AnyMap compileProperties() {
    ov::AnyMap result = bench::properties();
    if (collectNodeProfiles())
        result[ov::enable_profiling.name()] = true;
    return result;
}

void reportCompiledProperties(const std::string& name, const ov::CompiledModel& compiled,
    const char* inputMode) {
    std::string devices = "unavailable";
    std::string precision = "unavailable";
    try {
        const auto executionDevices = compiled.get_property(ov::execution_devices);
        devices.clear();
        for (const auto& device : executionDevices) {
            if (!devices.empty())
                devices += ",";
            devices += device;
        }
    } catch (const std::exception&) {
    }
    try {
        precision = compiled.get_property(ov::hint::inference_precision).to_string();
    } catch (const std::exception&) {
    }
    std::cout << "E1_CONFIG case=" << name << " input=" << inputMode
              << " device=" << devices << " inference_precision=" << precision
              << " performance_hint=" << benchmark_config::kPerformanceHint
              << " num_streams=" << benchmark_config::kNumStreams
              << " profiling=" << (collectNodeProfiles() ? "on" : "off")
              << " warmup=" << benchmark_config::kWarmupFrames
              << " measured=" << benchmark_config::kMeasuredFrames
              << " samples=" << benchmark_config::kSampleCount << std::endl;
}

struct NodeProfileTotals {
    uint64_t realTimeMicroseconds = 0;
    uint64_t cpuTimeMicroseconds = 0;
    uint64_t observations = 0;
};
#else
ov::AnyMap compileProperties() {
    return bench::properties();
}

void reportCompiledProperties(const std::string&, const ov::CompiledModel&, const char*) {}
#endif

template <typename Bind>
void measure(const std::string& name, ov::CompiledModel& compiled, Bind bind) {
    std::vector<double> samples;
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
        ov::InferRequest request = compiled.create_infer_request();
        double fullPathElapsed = 0;
        double inferenceElapsed = 0;
#ifndef _WIN32
        double bindElapsed = 0;
        double outputElapsed = 0;
        std::map<std::tuple<std::string, std::string, std::string>, NodeProfileTotals> nodeProfiles;
#endif
        for (int frame = 0; frame < bench::totalFrames(); ++frame) {
        const auto frameStart = bench::Clock::now();
#ifndef _WIN32
            const auto bindStart = collectStageTimings() ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            bind(request, frame);
#ifndef _WIN32
            const auto bindStop = collectStageTimings() ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            if (testing::Test::HasFatalFailure())
                return;
            const auto inferenceStart = bench::Clock::now();
            request.infer();
            const auto inferenceStop = bench::Clock::now();
#ifdef _WIN32
            const size_t outputSize = request.get_output_tensor().get_size();
#else
            const auto outputStart = collectStageTimings() ? bench::Clock::now() : bench::Clock::time_point{};
            const size_t outputSize = request.get_output_tensor().get_size();
            const auto outputStop = collectStageTimings() ? bench::Clock::now() : bench::Clock::time_point{};
#endif
            const auto frameStop = bench::Clock::now();
            ASSERT_GT(outputSize, 0);
            if (frame >= bench::warmupFrames()) {
                fullPathElapsed += bench::milliseconds(frameStart, frameStop);
                inferenceElapsed += bench::milliseconds(inferenceStart, inferenceStop);
#ifndef _WIN32
                if (collectStageTimings()) {
                    bindElapsed += bench::milliseconds(bindStart, bindStop);
                    outputElapsed += bench::milliseconds(outputStart, outputStop);
                }
                if (collectNodeProfiles()) {
                    for (const auto& info : request.get_profiling_info()) {
                        auto& totals = nodeProfiles[{info.node_name, info.node_type, info.exec_type}];
                        totals.realTimeMicroseconds += info.real_time.count();
                        totals.cpuTimeMicroseconds += info.cpu_time.count();
                        ++totals.observations;
                    }
                }
#endif
            }
        }
        samples.push_back(fullPathElapsed / bench::measuredFrames());
#ifndef _WIN32
        if (collectStageTimings()) {
            std::cout << std::fixed << std::setprecision(6)
                      << "E1_STAGE case=" << name << " sample=" << sample + 1
                      << " measured=" << bench::measuredFrames()
                      << " bind_upload_ms_per_frame=" << bindElapsed / bench::measuredFrames()
                      << " infer_call_ms_per_frame=" << inferenceElapsed / bench::measuredFrames()
                      << " output_access_ms_per_frame=" << outputElapsed / bench::measuredFrames()
                      << " full_path_ms_per_frame=" << fullPathElapsed / bench::measuredFrames()
                      << std::endl;
        }
        for (const auto& [key, totals] : nodeProfiles) {
            const auto& [nodeName, nodeType, execType] = key;
            std::cout << std::fixed << std::setprecision(3)
                      << "E1_NODE_PROFILE case=" << name << " sample=" << sample + 1
                      << " node=" << nodeName << " type=" << nodeType << " exec=" << execType
                      << " mean_real_us=" << static_cast<double>(totals.realTimeMicroseconds) / totals.observations
                      << " mean_cpu_us=" << static_cast<double>(totals.cpuTimeMicroseconds) / totals.observations
                      << " observations=" << totals.observations << std::endl;
        }
#endif
    }
    bench::report(name, samples);
}

void host(const std::string& device) {
    auto frames = bench::frames();
    ov::Core core;
    auto compiled = core.compile_model(bench::model(core, false), device, compileProperties());
    reportCompiledProperties("pure_ov_" + device + "_host", compiled, "host");
    measure("pure_ov_" + device + "_host", compiled, [&](ov::InferRequest& request, int index) {
        request.set_tensor(bench::kYInputName, ov::Tensor(ov::element::u8, kYShape, frames[index].data()));
        request.set_tensor(bench::kUvInputName, ov::Tensor(ov::element::u8, kUvShape, frames[index].data() + kWidth * kHeight));
    });
}

TEST(PureOpenVinoBenchmark, CpuHost) { host("CPU"); }
TEST(PureOpenVinoBenchmark, GpuHost) { host("GPU"); }

TEST(PureOpenVinoBenchmark, OpenClRemote) {
    auto frames = bench::frames();
    ov::Core core;
    auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    auto compiled = core.compile_model(bench::model(core, false), context, compileProperties());
    reportCompiledProperties("pure_ov_GPU_opencl", compiled, "opencl_buffers");
    cl::Context clContext(context.get(), true);
    cl::CommandQueue queue(clContext);
    cl::Buffer yBuffer(clContext, CL_MEM_READ_WRITE, kWidth * kHeight);
    cl::Buffer uvBuffer(clContext, CL_MEM_READ_WRITE, kWidth * kHeight / 2);
    auto yTensor = context.create_tensor(ov::element::u8, kYShape, yBuffer);
    auto uvTensor = context.create_tensor(ov::element::u8, kUvShape, uvBuffer);
    measure("pure_ov_GPU_opencl", compiled, [&](ov::InferRequest& request, int index) {
        ASSERT_EQ(queue.enqueueWriteBuffer(yBuffer, CL_TRUE, 0, kWidth * kHeight, frames[index].data()), CL_SUCCESS);
        ASSERT_EQ(queue.enqueueWriteBuffer(uvBuffer, CL_TRUE, 0, kWidth * kHeight / 2,
                      frames[index].data() + kWidth * kHeight),
            CL_SUCCESS);
        request.set_tensor(bench::kYInputName, yTensor);
        request.set_tensor(bench::kUvInputName, uvTensor);
    });
}

#ifndef _WIN32
struct NativeSurface {
    int descriptor = -1;
    VADisplay display = nullptr;
    VASurfaceID surface = VA_INVALID_SURFACE;
    ~NativeSurface() {
        if (surface != VA_INVALID_SURFACE)
            vaDestroySurfaces(display, &surface, 1);
        if (display)
            vaTerminate(display);
        if (descriptor >= 0)
            close(descriptor);
    }
};
#endif

TEST(PureOpenVinoBenchmark, NativeGpuMemory) {
    auto frames = bench::frames();
    ov::Core core;
#ifdef _WIN32
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    D3D_FEATURE_LEVEL level;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &device, &level, nullptr)));
    ov::intel_gpu::ocl::D3DContext context(core, device.Get());
    auto compiled = core.compile_model(bench::model(core, true), context, bench::properties());
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> deviceContext;
    device->GetImmediateContext(&deviceContext);
    D3D11_TEXTURE2D_DESC description{};
    description.Width = kWidth;
    description.Height = kHeight;
    description.MipLevels = 1;
    description.ArraySize = 1;
    description.Format = DXGI_FORMAT_NV12;
    description.SampleDesc.Count = 1;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&description, nullptr, &texture)));
    auto planes = context.create_tensor_nv12(kHeight, kWidth, texture.Get());
    measure("pure_ov_GPU_d3d11", compiled, [&](ov::InferRequest& request, int index) {
        deviceContext->UpdateSubresource(texture.Get(), 0, nullptr, frames[index].data(), kWidth, 0);
        request.set_tensor(bench::kYInputName, planes.first);
        request.set_tensor(bench::kUvInputName, planes.second);
    });
#else
    NativeSurface native;
    native.descriptor = open(bench::environment("OVMS_PERF_RENDER_NODE", "/dev/dri/renderD128").c_str(), O_RDWR);
    if (native.descriptor < 0)
        GTEST_SKIP() << "VA-API render node unavailable";
    native.display = vaGetDisplayDRM(native.descriptor);
    int major = 0;
    int minor = 0;
    if (!native.display || vaInitialize(native.display, &major, &minor) != VA_STATUS_SUCCESS) {
        native.display = nullptr;
        GTEST_SKIP() << "VA-API display initialization unsupported";
    }
    ASSERT_EQ(vaCreateSurfaces(native.display, VA_RT_FORMAT_YUV420, kWidth, kHeight,
                  &native.surface, 1, nullptr, 0),
        VA_STATUS_SUCCESS);
    std::unique_ptr<ov::intel_gpu::ocl::VAContext> context;
    ov::CompiledModel compiled;
    try {
        context = std::make_unique<ov::intel_gpu::ocl::VAContext>(core, native.display);
        compiled = core.compile_model(bench::model(core, true), *context, compileProperties());
    } catch (const std::exception& error) {
        GTEST_SKIP() << "VA-API remote inference unavailable: " << error.what();
    }
    reportCompiledProperties("pure_ov_GPU_vaapi", compiled, "va_surface");
    auto planes = context->create_tensor_nv12(kHeight, kWidth, native.surface);
    measure("pure_ov_GPU_vaapi", compiled, [&](ov::InferRequest& request, int index) {
        VAImage image{};
        ASSERT_EQ(vaDeriveImage(native.display, native.surface, &image), VA_STATUS_SUCCESS);
        void* mapped = nullptr;
        ASSERT_EQ(vaMapBuffer(native.display, image.buf, &mapped), VA_STATUS_SUCCESS);
        ASSERT_EQ(image.format.fourcc, VA_FOURCC_NV12);
        for (size_t row = 0; row < kHeight; ++row)
            std::memcpy(static_cast<uint8_t*>(mapped) + image.offsets[0] + row * image.pitches[0],
                frames[index].data() + row * kWidth, kWidth);
        for (size_t row = 0; row < kHeight / 2; ++row)
            std::memcpy(static_cast<uint8_t*>(mapped) + image.offsets[1] + row * image.pitches[1],
                frames[index].data() + kWidth * kHeight + row * kWidth, kWidth);
        ASSERT_EQ(vaUnmapBuffer(native.display, image.buf), VA_STATUS_SUCCESS);
        ASSERT_EQ(vaDestroyImage(native.display, image.image_id), VA_STATUS_SUCCESS);
        request.set_tensor(bench::kYInputName, planes.first);
        request.set_tensor(bench::kUvInputName, planes.second);
    });
#endif
}
}  // namespace