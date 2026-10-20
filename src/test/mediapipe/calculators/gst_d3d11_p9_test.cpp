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
 * P9 Windows live-streaming tests. Self-contained (no MediaPipe graph): each
 * test drives the intel_mpi GStreamer/D3D11 pipeline directly and, where it
 * needs inference, the OVMS C-API against a ModelManager-loaded GPU model.
 *
 * Covered:
 *   - Device ownership: GStreamer decodes onto the server-owned ID3D11Device
 *     and the ModelManager GPU model runs on that same device (zero host copy).
 *   - Source EOS lifecycle: a finite file source reaches EOS and releases
 *     cleanly (Task 4).
 *   - Direct camera capture -> D3D11 inference + H.264 MP4 output (Task 2, gated
 *     on OVMS_TEST_CAMERA).
 *   - RTSP ingest onto the owned device (Task 3, gated on OVMS_TEST_RTSP_URL).
 *   - RTSP publication of encoded H.264 (Task 3, gated on OVMS_TEST_RTSP_OUT).
 *
 * Needs Windows + Intel GPU + GStreamer D3D11 + the face_detection_adas fixture
 * and a test video; skips otherwise. Env overrides:
 *   VIDEO_PATH            (default C:\git\model_server\recording_3_raw.avi)
 *   OVMS_REPO_ROOT        (default C:\git\model_server)
 *   OVMS_TEST_CAMERA      (set to 1 to run the live camera test)
 *   OVMS_TEST_CAMERA_DEVICE (default "HP 5MP Camera")
 *   OVMS_TEST_RTSP_URL    (rtsp:// source URL for ingest/publish tests)
 *   OVMS_TEST_RTSP_OUT    (rtsp:// destination URL for the publish test)
 */

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "src/mpi/intel_mpi.h"
#include "src/ovms.h"

#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

namespace {

using Microsoft::WRL::ComPtr;

constexpr char kModelName[] = "face_detection_adas";
constexpr char kInputName[] = "data";
constexpr int kModelW = 672;
constexpr int kModelH = 384;

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string repoRoot() {
    return envOr("OVMS_REPO_ROOT", "C:\\git\\model_server");
}

ComPtr<ID3D11Device> createDevice() {
    ComPtr<ID3D11Device> device;
    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0,
                                   D3D11_SDK_VERSION, device.GetAddressOf(), &got, nullptr);
    return SUCCEEDED(hr) ? device : nullptr;
}

std::string writeTempConfig() {
    const std::string modelDir = (std::filesystem::path(repoRoot()) / "src/test/face_detection_adas").generic_string();
    const std::string cfgPath = repoRoot() + "\\src\\test\\d3d11_p9_tmp.json";
    std::ofstream f(cfgPath);
    f << "{\"model_config_list\":[{\"config\":{\"name\":\"" << kModelName
      << "\",\"base_path\":\"" << modelDir << "\",\"target_device\":\"GPU\"}}]}";
    f.close();
    return cfgPath;
}

bool capiOk(OVMS_Status* status) {
    if (!status)
        return true;
    uint32_t code = 0;
    const char* details = nullptr;
    OVMS_Status* c = OVMS_StatusCode(status, &code);
    OVMS_Status* d = OVMS_StatusDetails(status, &details);
    std::cerr << "[CAPI ERROR] code=" << code
              << " details=" << (details ? details : "(null)") << std::endl;
    if (c)
        OVMS_StatusDelete(c);
    if (d)
        OVMS_StatusDelete(d);
    OVMS_StatusDelete(status);
    return false;
}

struct P9ServerFixture : public ::testing::Test {
    inline static ComPtr<ID3D11Device> suiteDevice;
    ComPtr<ID3D11Device> device;
    OVMS_Server* server = nullptr;
    std::string videoPath;
    bool ready = false;

    void SetUp() override {
        videoPath = envOr("VIDEO_PATH", repoRoot() + "\\recording_3_raw.avi");
        if (!imp_video_d3d11_available())
            GTEST_SKIP() << "D3D11 surface sharing not available";
        if (!fileExists(repoRoot() + "\\src\\test\\face_detection_adas\\1\\face-detection-adas-0001.xml"))
            GTEST_SKIP() << "face_detection_adas fixture not found";
        if (!suiteDevice)
            suiteDevice = createDevice();
        device = suiteDevice;
        if (!device)
            GTEST_SKIP() << "could not create a hardware ID3D11Device";

        const std::string cfg = writeTempConfig();
        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_TRUE(capiOk(OVMS_ServerSettingsNew(&serverSettings)));
        ASSERT_TRUE(capiOk(OVMS_ModelsSettingsNew(&modelsSettings)));
        ASSERT_TRUE(capiOk(OVMS_ServerSettingsSetGrpcPort(serverSettings, 9719)));
        ASSERT_TRUE(capiOk(OVMS_ModelsSettingsSetConfigPath(modelsSettings, cfg.c_str())));
        ASSERT_TRUE(capiOk(OVMS_ServerNew(&server)));

        // Register the server-owned device and share it with GStreamer BEFORE
        // the server loads GPU models (so they compile on the matching device).
        ASSERT_TRUE(capiOk(OVMS_ServerSetGlobalD3D11Device(server, device.Get())));
        imp_video_set_d3d11_device(device.Get());

        ASSERT_TRUE(capiOk(OVMS_ServerStartFromConfigurationFile(server, serverSettings, modelsSettings)));
        ready = true;
    }

    void TearDown() override {
        if (server) {
            OVMS_ServerSetGlobalD3D11Device(server, nullptr);
            imp_video_set_d3d11_device(nullptr);
            OVMS_ServerDelete(server);
        }
        std::filesystem::remove(repoRoot() + "\\src\\test\\d3d11_p9_tmp.json");
    }

    static void TearDownTestSuite() {
        suiteDevice.Reset();
    }

    int runCapiInference(void* texture, int w, int h) {
        OVMS_InferenceRequest* request = nullptr;
        if (!capiOk(OVMS_InferenceRequestNew(&request, server, kModelName, 1)))
            return -1;
        const std::string yName = std::string(kInputName) + "/y";
        const std::string uvName = std::string(kInputName) + "/uv";
        const int64_t shapeY[] = {1, h, w, 1};
        const int64_t shapeUV[] = {1, h / 2, w / 2, 2};
        const size_t bytesY = static_cast<size_t>(w) * h;
        const size_t bytesUV = bytesY / 2;
        capiOk(OVMS_InferenceRequestAddInput(request, yName.c_str(), OVMS_DATATYPE_U8, shapeY, 4));
        capiOk(OVMS_InferenceRequestInputSetData(request, yName.c_str(), texture, bytesY, OVMS_BUFFERTYPE_D3D11_TEXTURE_Y, 1));
        capiOk(OVMS_InferenceRequestAddInput(request, uvName.c_str(), OVMS_DATATYPE_U8, shapeUV, 4));
        capiOk(OVMS_InferenceRequestInputSetData(request, uvName.c_str(), texture, bytesUV, OVMS_BUFFERTYPE_D3D11_TEXTURE_UV, 1));

        OVMS_InferenceResponse* response = nullptr;
        if (!capiOk(OVMS_Inference(server, request, &response))) {
            OVMS_InferenceRequestDelete(request);
            return -1;
        }
        const void* out = nullptr;
        size_t bytes = 0;
        uint32_t id = 0;
        OVMS_DataType dt = static_cast<OVMS_DataType>(199);
        const int64_t* shape = nullptr;
        size_t dims = 0;
        OVMS_BufferType bt = OVMS_BUFFERTYPE_CPU;
        uint32_t did = 0;
        const char* oname = nullptr;
        int detections = -1;
        if (capiOk(OVMS_InferenceResponseOutput(response, id, &oname, &dt, &shape, &dims, &out, &bytes, &bt, &did)) && out) {
            detections = 0;
            const float* v = static_cast<const float*>(out);
            for (size_t i = 0; i + 7 <= bytes / sizeof(float); i += 7) {
                if (v[i + 2] >= 0.5f)
                    ++detections;
            }
        }
        OVMS_InferenceResponseDelete(response);
        OVMS_InferenceRequestDelete(request);
        return detections;
    }
};

// Device ownership: the decoded D3D11 texture lives on the server-owned device
// and the ModelManager GPU model runs inference on that same device.
TEST_F(P9ServerFixture, DecodeAndInferShareOwnedDevice) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (!fileExists(videoPath))
        GTEST_SKIP() << "test video not found: " << videoPath;

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", videoPath.c_str());
    imp_video_source_set(src, "width", std::to_string(kModelW).c_str());
    imp_video_source_set(src, "height", std::to_string(kModelH).c_str());
    imp_video_decode_opts_t vopts{};
    vopts.use_d3d11_surface_memory = true;
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);

    for (int frameIndex = 0; frameIndex < 30; ++frameIndex) {
        SCOPED_TRACE(frameIndex);
        imp_tensor_t* tensor = nullptr;
        ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK);
        ASSERT_NE(tensor, nullptr);
        ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE);
        void* texture = nullptr;
        void* decodeDevice = nullptr;
        uint32_t subresource = 0;
        int frameWidth = 0, frameHeight = 0;
        ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource,
            &frameWidth, &frameHeight), IMP_OK);
        EXPECT_EQ(decodeDevice, static_cast<void*>(device.Get())) << "decoded on a different device than the server owns";
        EXPECT_EQ(subresource, 0u);
        EXPECT_GE(runCapiInference(texture, frameWidth, frameHeight), 1)
            << "D3D11 inference on the owned device did not detect a face";
        imp_tensor_release(tensor);
    }
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

// Task 4) Source EOS lifecycle: a finite file source reaches EOS deterministically
// and releases every resource with no leak or hang.
TEST_F(P9ServerFixture, FileSourceEosReleasesCleanly) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (!fileExists(videoPath))
        GTEST_SKIP() << "test video not found: " << videoPath;

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", videoPath.c_str());
    imp_video_source_set(src, "width", std::to_string(kModelW).c_str());
    imp_video_source_set(src, "height", std::to_string(kModelH).c_str());
    imp_video_decode_opts_t vopts{};
    vopts.use_d3d11_surface_memory = true;
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);

    int frames = 0;
    bool reachedEos = false;
    for (int i = 0; i < 100000; ++i) {
        imp_tensor_t* tensor = nullptr;
        imp_status_t st = imp_video_read_frame(&tensor, stream, 0);
        if (st != IMP_OK || tensor == nullptr) {
            reachedEos = true;
            break;
        }
        ++frames;
        imp_tensor_release(tensor);
    }
    EXPECT_TRUE(reachedEos) << "finite file source never signalled EOS";
    EXPECT_GE(frames, 1) << "no frames decoded before EOS";
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

// Task 2) Direct camera capture -> D3D11 inference + H.264 MP4 output. Gated on a
// real capture device (OVMS_TEST_CAMERA=1, OVMS_TEST_CAMERA_DEVICE).
TEST_F(P9ServerFixture, CameraCaptureInferenceAndMp4) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (envOr("OVMS_TEST_CAMERA", "").empty())
        GTEST_SKIP() << "set OVMS_TEST_CAMERA=1 to run the live camera test";
    const std::string cameraDevice = envOr("OVMS_TEST_CAMERA_DEVICE", "HP 5MP Camera");

    const std::string outPath = repoRoot() + "\\src\\test\\d3d11_p9_camera_out.mp4";
    std::filesystem::remove(outPath);
    {
        imp_context_t* ctx = nullptr;
        ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, IMP_SOURCE_CAMERA);
        imp_video_source_set(src, "device", cameraDevice.c_str());
        imp_video_source_set(src, "width", "1280");
        imp_video_source_set(src, "height", "720");
        imp_video_branch_t branches[2]{};
        branches[0].width = kModelW;
        branches[0].height = kModelH;
        branches[1].width = 1280;
        branches[1].height = 720;
        branches[1].use_system_memory = true;
        imp_video_decode_opts_t vopts{};
        vopts.use_d3d11_surface_memory = true;
        vopts.branches = branches;
        vopts.branch_count = 2;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
        imp_video_source_destroy(src);

        imp_video_encode_opts_t eopts{};
        eopts.codec = "h264";
        eopts.framerate = 30;
        eopts.output_path = outPath.c_str();
        eopts.output_mode = IMP_OUTPUT_FILE;
        imp_video_encoder_t* enc = nullptr;
        ASSERT_EQ(imp_video_encoder_create(&enc, 1280, 720, ctx, &eopts), IMP_OK) << imp_context_get_error(ctx);

        imp_tensor_t* retainedHost = nullptr;
        const uint8_t* retainedY = nullptr;
        std::vector<uint8_t> retainedPixels;
        for (int frameIndex = 0; frameIndex < 30; ++frameIndex) {
            imp_tensor_t* inferTensor = nullptr;
            ASSERT_EQ(imp_video_read_frame(&inferTensor, stream, 0), IMP_OK);
            ASSERT_NE(inferTensor, nullptr);
            EXPECT_EQ(imp_tensor_get_memory_type(inferTensor), IMP_MEM_D3D11_SURFACE);
            void* texture = nullptr;
            void* decodeDevice = nullptr;
            uint32_t subresource = 0;
            int frameWidth = 0, frameHeight = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(inferTensor, &texture, &decodeDevice,
                &subresource, &frameWidth, &frameHeight), IMP_OK);
            EXPECT_EQ(decodeDevice, static_cast<void*>(device.Get()));
            EXPECT_EQ(subresource, 0u);
            EXPECT_GE(runCapiInference(texture, frameWidth, frameHeight), 0);
            imp_tensor_release(inferTensor);

            imp_tensor_t* hostTensor = nullptr;
            ASSERT_EQ(imp_video_read_frame(&hostTensor, stream, 1), IMP_OK);
            ASSERT_NE(hostTensor, nullptr);
            EXPECT_EQ(imp_tensor_get_memory_type(hostTensor), IMP_MEM_SYSTEM);
            EXPECT_EQ(imp_video_encoder_write(enc, hostTensor), IMP_OK);
            if (frameIndex == 0) {
                const uint8_t* retainedUv = nullptr;
                ASSERT_EQ(imp_tensor_get_nv12_planes(hostTensor, &retainedY, &retainedUv,
                    &frameWidth, &frameHeight), IMP_OK);
                ASSERT_NE(retainedY, nullptr);
                retainedPixels.assign(retainedY, retainedY + 256);
                retainedHost = hostTensor;
            } else {
                imp_tensor_release(hostTensor);
            }
        }
        EXPECT_TRUE(std::equal(retainedPixels.begin(), retainedPixels.end(), retainedY));
        imp_tensor_release(retainedHost);
        imp_video_encoder_close(enc);
        imp_video_close(stream);
        imp_context_destroy(ctx);
    }
    ASSERT_TRUE(fileExists(outPath)) << "camera MP4 was not produced";
    EXPECT_GT(std::filesystem::file_size(outPath), 0u) << "camera MP4 is empty";
    std::filesystem::remove(outPath);
}

// Task 3) RTSP ingest onto the server-owned device. Gated on OVMS_TEST_RTSP_URL.
TEST_F(P9ServerFixture, RtspSourceDecodesOnOwnedDevice) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    const std::string url = envOr("OVMS_TEST_RTSP_URL", "");
    if (url.empty())
        GTEST_SKIP() << "set OVMS_TEST_RTSP_URL to run the RTSP ingest test";

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_URL);
    imp_video_source_set(src, "url", url.c_str());
    imp_video_source_set(src, "width", std::to_string(kModelW).c_str());
    imp_video_source_set(src, "height", std::to_string(kModelH).c_str());
    imp_video_decode_opts_t vopts{};
    vopts.use_d3d11_surface_memory = true;
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);

    imp_tensor_t* tensor = nullptr;
    ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE);
    void* texture = nullptr;
    void* decodeDevice = nullptr;
    uint32_t subresource = 0;
    int fw = 0, fh = 0;
    ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource, &fw, &fh), IMP_OK);
    EXPECT_EQ(decodeDevice, static_cast<void*>(device.Get())) << "RTSP decoded on a different device";
    EXPECT_GE(runCapiInference(texture, fw, fh), 0) << "RTSP D3D11 inference failed";
    imp_tensor_release(tensor);
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

// Task 3) RTSP publication of encoded H.264 to an external RTSP server (e.g.
// MediaMTX). Gated on OVMS_TEST_RTSP_URL (source) + OVMS_TEST_RTSP_OUT (dest).
TEST_F(P9ServerFixture, RtspOutputPublishes) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    const std::string inUrl = envOr("OVMS_TEST_RTSP_URL", "");
    const std::string outUrl = envOr("OVMS_TEST_RTSP_OUT", "");
    if (inUrl.empty() || outUrl.empty())
        GTEST_SKIP() << "set OVMS_TEST_RTSP_URL and OVMS_TEST_RTSP_OUT to run RTSP publish";

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_URL);
    imp_video_source_set(src, "url", inUrl.c_str());
    imp_video_source_set(src, "width", "1280");
    imp_video_source_set(src, "height", "720");
    imp_video_decode_opts_t vopts{};
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);

    imp_video_encode_opts_t eopts{};
    eopts.codec = "h264";
    eopts.framerate = 30;
    eopts.output_path = outUrl.c_str();
    eopts.output_mode = IMP_OUTPUT_RTSP;
    imp_video_encoder_t* enc = nullptr;
    ASSERT_EQ(imp_video_encoder_create(&enc, 1280, 720, ctx, &eopts), IMP_OK) << imp_context_get_error(ctx);

    int written = 0;
    for (int i = 0; i < 60; ++i) {
        imp_tensor_t* tensor = nullptr;
        if (imp_video_read_frame(&tensor, stream, 0) != IMP_OK || tensor == nullptr)
            break;
        if (imp_video_encoder_write(enc, tensor) == IMP_OK)
            ++written;
        imp_tensor_release(tensor);
    }
    EXPECT_GE(written, 1) << "no frames published to RTSP";
    imp_video_encoder_close(enc);
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

}  // namespace
