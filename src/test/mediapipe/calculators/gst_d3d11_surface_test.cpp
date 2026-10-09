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
 * P5 Windows test A — device identity + texture-array probe.
 *
 * Proves the server-owned ID3D11Device is the one GStreamer decodes onto (the
 * D3D11 counterpart of GStreamerDecodesOntoServerOwnedVADisplay), and records
 * the decoded texture's ArraySize / subresource so we know whether OpenVINO can
 * import it directly (ArraySize==1, subresource==0) or whether a slice copy is
 * required (OpenVINO's D3D11 import has no subresource parameter).
 *
 * GPU + D3D11 + a test video are required, so it skips otherwise.
 *   VIDEO_PATH (default C:\git\model_server\recording_3_raw.avi)
 *   WIDTH/HEIGHT (default 672x384 — face-detection-adas-0001 input)
 */

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "src/mpi/intel_mpi.h"

namespace {

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int envInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Create a hardware ID3D11Device, standing in for the server-owned device.
Microsoft::WRL::ComPtr<ID3D11Device> createDevice() {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        nullptr, 0, D3D11_SDK_VERSION,
        device.GetAddressOf(), &got, nullptr);
    if (FAILED(hr))
        return nullptr;
    return device;
}

TEST(GstD3D11Surface, DecodesOntoServerOwnedDevice) {
    const std::string video_path =
        envOr("VIDEO_PATH", "C:\\git\\model_server\\recording_3_raw.avi");
    const int width = envInt("WIDTH", 672);
    const int height = envInt("HEIGHT", 384);

    if (!imp_video_d3d11_available())
        GTEST_SKIP() << "D3D11 surface sharing not available on this host";
    if (!fileExists(video_path))
        GTEST_SKIP() << "test video not found: " << video_path;

    auto device = createDevice();
    if (!device)
        GTEST_SKIP() << "could not create a hardware ID3D11Device";

    // Share the server-owned device with GStreamer BEFORE opening the stream.
    imp_video_set_d3d11_device(device.Get());

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);

    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", video_path.c_str());
    imp_video_source_set(src, "width", std::to_string(width).c_str());
    imp_video_source_set(src, "height", std::to_string(height).c_str());

    imp_video_decode_opts_t vopts{};
    vopts.use_d3d11_surface_memory = true;

    imp_video_stream_t* stream = nullptr;
    imp_status_t st = imp_video_open(&stream, src, ctx, &vopts);
    imp_video_source_destroy(src);
    ASSERT_EQ(st, IMP_OK) << "imp_video_open failed: " << imp_context_get_error(ctx);

    imp_tensor_t* tensor = nullptr;
    st = imp_video_read_frame(&tensor, stream, 0);
    ASSERT_EQ(st, IMP_OK);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE)
        << "expected a GPU D3D11 texture frame";

    void* texture = nullptr;
    void* decode_device = nullptr;
    uint32_t subresource = 0xffffffff;
    int fw = 0, fh = 0;
    ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decode_device,
                                           &subresource, &fw, &fh),
              IMP_OK);
    ASSERT_NE(texture, nullptr);

    // The proof: GStreamer decoded onto the server-owned device.
    EXPECT_EQ(decode_device, static_cast<void*>(device.Get()))
        << "GStreamer used its own device instead of the server-owned one";

    // Probe the texture layout to decide the OpenVINO import strategy.
    D3D11_TEXTURE2D_DESC desc{};
    reinterpret_cast<ID3D11Texture2D*>(texture)->GetDesc(&desc);
    std::cout << "[D3D11 probe] format=" << desc.Format
              << " ArraySize=" << desc.ArraySize
              << " subresource=" << subresource
              << " w=" << fw << " h=" << fh << std::endl;
    EXPECT_EQ(desc.Format, DXGI_FORMAT_NV12) << "expected NV12 texture";

    imp_tensor_release(tensor);
    imp_video_close(stream);
    imp_context_destroy(ctx);
    imp_video_set_d3d11_device(nullptr);
}

}  // namespace
