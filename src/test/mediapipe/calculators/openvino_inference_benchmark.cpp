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
#endif
#include "inference_benchmark_utils.hpp"

namespace {
namespace bench = inference_benchmark;
constexpr size_t kWidth = benchmark_config::kModelWidth;
constexpr size_t kHeight = benchmark_config::kModelHeight;
const ov::Shape kYShape{1, kHeight, kWidth, 1};
const ov::Shape kUvShape{1, kHeight / 2, kWidth / 2, 2};

template <typename Bind>
void measure(const std::string& name, ov::CompiledModel& compiled, Bind bind) {
    std::vector<double> samples;
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
        ov::InferRequest request = compiled.create_infer_request();
        double elapsed = 0;
        for (int frame = 0; frame < bench::kTotalFrames; ++frame) {
            bind(request, frame);
            if (testing::Test::HasFatalFailure())
                return;
            const auto start = bench::Clock::now();
            request.infer();
            const auto stop = bench::Clock::now();
            ASSERT_GT(request.get_output_tensor().get_size(), 0);
            if (frame >= benchmark_config::kWarmupFrames)
                elapsed += bench::milliseconds(start, stop);
        }
        samples.push_back(elapsed / benchmark_config::kMeasuredFrames);
    }
    bench::report(name, samples);
}

void host(const std::string& device) {
    auto frames = bench::frames();
    ov::Core core;
    auto compiled = core.compile_model(bench::model(core, false), device, bench::properties());
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
    auto compiled = core.compile_model(bench::model(core, false), context, bench::properties());
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
        compiled = core.compile_model(bench::model(core, true), *context, bench::properties());
    } catch (const std::exception& error) {
        GTEST_SKIP() << "VA-API remote inference unavailable: " << error.what();
    }
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