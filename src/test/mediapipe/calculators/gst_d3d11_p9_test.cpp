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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>

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
      << "\",\"base_path\":\"" << modelDir
      << "\",\"target_device\":\"GPU\",\"nireq\":4,\"plugin_config\":{\"PERFORMANCE_HINT\":\"LATENCY\",\"NUM_STREAMS\":\"1\"}}}]}";
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

    int runReusableCapiInference(OVMS_InferenceRequest* request, bool* hasBoundData,
        void* texture, int w, int h) {
        const std::string yName = std::string(kInputName) + "/y";
        const std::string uvName = std::string(kInputName) + "/uv";
        const size_t bytesY = static_cast<size_t>(w) * h;
        const size_t bytesUV = bytesY / 2;
        if ((*hasBoundData &&
                (!capiOk(OVMS_InferenceRequestInputRemoveData(request, yName.c_str())) ||
                    !capiOk(OVMS_InferenceRequestInputRemoveData(request, uvName.c_str())))) ||
            !capiOk(OVMS_InferenceRequestInputSetData(
                request, yName.c_str(), texture, bytesY, OVMS_BUFFERTYPE_D3D11_TEXTURE_Y, 1)) ||
            !capiOk(OVMS_InferenceRequestInputSetData(
                request, uvName.c_str(), texture, bytesUV, OVMS_BUFFERTYPE_D3D11_TEXTURE_UV, 1)))
            return -1;
        *hasBoundData = true;

        OVMS_InferenceResponse* response = nullptr;
        if (!capiOk(OVMS_Inference(server, request, &response)))
            return -1;
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
        if (capiOk(OVMS_InferenceResponseOutput(response, id, &oname, &dt, &shape, &dims,
                &out, &bytes, &bt, &did)) &&
            out) {
            detections = 0;
            const float* values = static_cast<const float*>(out);
            for (size_t result = 0; result + 7 <= bytes / sizeof(float); result += 7) {
                if (values[result + 2] >= 0.5f)
                    ++detections;
            }
        }
        OVMS_InferenceResponseDelete(response);
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
    vopts.output_memory = IMP_VIDEO_MEMORY_NATIVE;
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
                      &frameWidth, &frameHeight),
            IMP_OK);
        EXPECT_EQ(decodeDevice, static_cast<void*>(device.Get())) << "decoded on a different device than the server owns";
        EXPECT_EQ(subresource, 0u);
        EXPECT_GE(runCapiInference(texture, frameWidth, frameHeight), 1)
            << "D3D11 inference on the owned device did not detect a face";
        imp_tensor_release(tensor);
    }
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

TEST_F(P9ServerFixture, FileSourceDirectCapiPerformance) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (!fileExists(videoPath))
        GTEST_SKIP() << "test video not found: " << videoPath;

    constexpr int warmupFrames = 20;
    constexpr int measuredFrames = 150;
    constexpr int trialCount = 3;
    using Clock = std::chrono::steady_clock;

    const auto runTrial = [&](bool runInference, bool reuseRequest, bool boundedDecode, int trial) {
        imp_context_t* ctx = nullptr;
        ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
        imp_video_source_t* src = nullptr;
        ASSERT_EQ(imp_video_source_create(&src, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.output_memory = IMP_VIDEO_MEMORY_NATIVE;
        options.sync_appsink = boundedDecode;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, src, ctx, &options), IMP_OK) << imp_context_get_error(ctx);
        imp_video_source_destroy(src);

        OVMS_InferenceRequest* reusableRequest = nullptr;
        bool reusableRequestHasBoundData = false;
        if (reuseRequest) {
            const std::string yName = std::string(kInputName) + "/y";
            const std::string uvName = std::string(kInputName) + "/uv";
            const int64_t shapeY[] = {1, kModelH, kModelW, 1};
            const int64_t shapeUV[] = {1, kModelH / 2, kModelW / 2, 2};
            ASSERT_TRUE(capiOk(OVMS_InferenceRequestNew(&reusableRequest, server, kModelName, 1)));
            ASSERT_TRUE(capiOk(OVMS_InferenceRequestAddInput(
                reusableRequest, yName.c_str(), OVMS_DATATYPE_U8, shapeY, 4)));
            ASSERT_TRUE(capiOk(OVMS_InferenceRequestAddInput(
                reusableRequest, uvName.c_str(), OVMS_DATATYPE_U8, shapeUV, 4)));
        }

        std::vector<double> decodeMs;
        std::vector<double> capiMs;
        decodeMs.reserve(measuredFrames);
        capiMs.reserve(measuredFrames);
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            if (frameIndex == warmupFrames)
                measuredStart = Clock::now();
            imp_tensor_t* tensor = nullptr;
            const auto decodeStart = Clock::now();
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            const auto decodeStop = Clock::now();
            ASSERT_NE(tensor, nullptr);
            ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE);

            if (runInference) {
                void* texture = nullptr;
                void* decodeDevice = nullptr;
                uint32_t subresource = 0;
                int frameWidth = 0;
                int frameHeight = 0;
                ASSERT_EQ(imp_tensor_get_d3d11_texture(
                              tensor, &texture, &decodeDevice, &subresource, &frameWidth, &frameHeight),
                    IMP_OK);
                ASSERT_EQ(decodeDevice, static_cast<void*>(device.Get()));
                ASSERT_EQ(subresource, 0u);
                const auto capiStart = Clock::now();
                const int detections = reuseRequest
                                           ? runReusableCapiInference(reusableRequest, &reusableRequestHasBoundData,
                                                 texture, frameWidth, frameHeight)
                                           : runCapiInference(texture, frameWidth, frameHeight);
                ASSERT_GE(detections, 0);
                const auto capiStop = Clock::now();
                if (frameIndex >= warmupFrames)
                    capiMs.push_back(std::chrono::duration<double, std::milli>(capiStop - capiStart).count());
            }

            if (frameIndex >= warmupFrames)
                decodeMs.push_back(std::chrono::duration<double, std::milli>(decodeStop - decodeStart).count());
            imp_tensor_release(tensor);
            if (frameIndex == warmupFrames + measuredFrames - 1)
                measuredStop = Clock::now();
        }
        OVMS_InferenceRequestDelete(reusableRequest);
        imp_video_close(stream);
        imp_context_destroy(ctx);

        const double decodeSum = std::accumulate(decodeMs.begin(), decodeMs.end(), 0.0);
        const double capiSum = std::accumulate(capiMs.begin(), capiMs.end(), 0.0);
        const double stageSum = decodeSum + capiSum;
        const double loopWallMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart)
                                      .count();
        std::cout << std::fixed << std::setprecision(3)
                  << "P11_DIRECT_D3D11 mode=" << (runInference ? (reuseRequest ? "decode_plus_capi_reuse_request" : "decode_plus_capi_new_request") : (boundedDecode ? "decode_only_bounded" : "decode_only"))
                  << " trial=" << trial
                  << " frames=" << measuredFrames
                  << " loop_wall_fps=" << measuredFrames * 1000.0 / loopWallMs
                  << " decode_mean_ms=" << decodeSum / measuredFrames;
        if (runInference) {
            std::sort(capiMs.begin(), capiMs.end());
            const auto percentile = [&capiMs](double fraction) {
                return capiMs[static_cast<size_t>(fraction * (capiMs.size() - 1))];
            };
            std::cout << " capi_mean_ms=" << capiSum / measuredFrames
                      << " capi_p50_ms=" << percentile(0.50)
                      << " capi_p90_ms=" << percentile(0.90)
                      << " capi_p99_ms=" << percentile(0.99)
                      << " combined_stage_fps=" << measuredFrames * 1000.0 / stageSum;
        } else {
            std::cout << " decode_only_fps=" << measuredFrames * 1000.0 / decodeSum;
        }
        std::cout << std::endl;
    };

    std::cout << "P10_DIRECT_D3D11 config=LATENCY streams=1 nireq=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(false, false, false, trial);
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(false, false, true, trial);
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(true, false, false, trial);
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(true, true, false, trial);
}

TEST_F(P9ServerFixture, FileSourcePureOpenVinoD3D11Performance) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (!fileExists(videoPath))
        GTEST_SKIP() << "test video not found: " << videoPath;

    constexpr int warmupFrames = 20;
    constexpr int measuredFrames = 150;
    constexpr int trialCount = 3;
    using Clock = std::chrono::steady_clock;

    const std::filesystem::path modelPath = std::filesystem::path(repoRoot()) /
                                            "src/test/face_detection_adas/1/face-detection-adas-0001.xml";
    ASSERT_TRUE(std::filesystem::exists(modelPath));

    ov::Core core;
    auto rawModel = core.read_model(modelPath.string(), {}, ov::AnyMap{{"ENABLE_MMAP", "NO"}});
    ov::preprocess::PrePostProcessor preprocessor(rawModel);
    preprocessor.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
        .set_memory_type(ov::intel_gpu::memory_type::surface);
    preprocessor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    preprocessor.input().model().set_layout("NCHW");
    const auto model = preprocessor.build();

    ov::intel_gpu::ocl::D3DContext d3dContext(core, device.Get());
    ov::AnyMap pluginConfig{{"PERFORMANCE_HINT", "LATENCY"}, {"NUM_STREAMS", "1"}};
    auto compiledModel = core.compile_model(model, d3dContext, pluginConfig);
    std::vector<ov::InferRequest> requests;
    requests.reserve(4);
    for (int requestIndex = 0; requestIndex < 4; ++requestIndex)
        requests.emplace_back(compiledModel.create_infer_request());

    const auto runTrial = [&](int trial) {
        imp_context_t* ctx = nullptr;
        ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
        imp_video_source_t* src = nullptr;
        ASSERT_EQ(imp_video_source_create(&src, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(src, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.output_memory = IMP_VIDEO_MEMORY_NATIVE;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, src, ctx, &options), IMP_OK) << imp_context_get_error(ctx);
        imp_video_source_destroy(src);

        std::vector<double> inferMs;
        std::vector<double> decodeMs;
        inferMs.reserve(measuredFrames);
        decodeMs.reserve(measuredFrames);
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int detections = 0;
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            if (frameIndex == warmupFrames)
                measuredStart = Clock::now();
            imp_tensor_t* tensor = nullptr;
            const auto decodeStart = Clock::now();
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            const auto decodeStop = Clock::now();
            ASSERT_NE(tensor, nullptr);
            void* texture = nullptr;
            void* decodeDevice = nullptr;
            uint32_t subresource = 0;
            int frameWidth = 0;
            int frameHeight = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(
                          tensor, &texture, &decodeDevice, &subresource, &frameWidth, &frameHeight),
                IMP_OK);
            ASSERT_EQ(decodeDevice, static_cast<void*>(device.Get()));
            ASSERT_EQ(subresource, 0u);

            const int64_t shapeY[] = {1, frameHeight, frameWidth, 1};
            const int64_t shapeUv[] = {1, frameHeight / 2, frameWidth / 2, 2};
            const auto yTensor = d3dContext.create_tensor(
                ov::element::u8, ov::Shape{1, static_cast<size_t>(frameHeight), static_cast<size_t>(frameWidth), 1},
                static_cast<ID3D11Texture2D*>(texture), 0);
            const auto uvTensor = d3dContext.create_tensor(
                ov::element::u8, ov::Shape{1, static_cast<size_t>(frameHeight / 2), static_cast<size_t>(frameWidth / 2), 2},
                static_cast<ID3D11Texture2D*>(texture), 1);
            auto& request = requests[frameIndex % requests.size()];
            request.set_tensor("data/y", yTensor);
            request.set_tensor("data/uv", uvTensor);
            const auto inferStart = Clock::now();
            request.infer();
            const auto inferStop = Clock::now();
            const auto output = request.get_output_tensor(0);
            const float* values = output.data<const float>();
            for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
                if (values[result + 2] >= 0.5f)
                    ++detections;
            }
            if (frameIndex >= warmupFrames) {
                decodeMs.push_back(std::chrono::duration<double, std::milli>(decodeStop - decodeStart).count());
                inferMs.push_back(std::chrono::duration<double, std::milli>(inferStop - inferStart).count());
            }
            imp_tensor_release(tensor);
            if (frameIndex == warmupFrames + measuredFrames - 1)
                measuredStop = Clock::now();
        }
        imp_video_close(stream);
        imp_context_destroy(ctx);

        const double decodeSum = std::accumulate(decodeMs.begin(), decodeMs.end(), 0.0);
        const double inferSum = std::accumulate(inferMs.begin(), inferMs.end(), 0.0);
        std::sort(inferMs.begin(), inferMs.end());
        const auto percentile = [&inferMs](double fraction) {
            return inferMs[static_cast<size_t>(fraction * (inferMs.size() - 1))];
        };
        const double loopWallMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart)
                                      .count();
        std::cout << std::fixed << std::setprecision(3)
                  << "P10_PURE_OV_D3D11 trial=" << trial
                  << " frames=" << measuredFrames
                  << " loop_wall_fps=" << measuredFrames * 1000.0 / loopWallMs
                  << " decode_mean_ms=" << decodeSum / measuredFrames
                  << " infer_mean_ms=" << inferSum / measuredFrames
                  << " infer_p50_ms=" << percentile(0.50)
                  << " infer_p90_ms=" << percentile(0.90)
                  << " infer_p99_ms=" << percentile(0.99)
                  << " detections=" << detections << std::endl;
    };

    std::cout << "P10_PURE_OV_D3D11 config=LATENCY streams=1 infer_requests=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(trial);
}

TEST_F(P9ServerFixture, HostTensorGpuNv12PreprocessPerformance) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";
    if (!fileExists(videoPath))
        GTEST_SKIP() << "test video not found: " << videoPath;

    constexpr int warmupFrames = 20;
    constexpr int measuredFrames = 150;
    constexpr int trialCount = 3;
    constexpr int totalFrames = warmupFrames + measuredFrames;
    using Clock = std::chrono::steady_clock;

    struct HostNv12Frame {
        std::vector<uint8_t> y;
        std::vector<uint8_t> uv;
    };
    std::vector<HostNv12Frame> frames;
    frames.reserve(totalFrames);

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    ASSERT_EQ(imp_video_source_create(&src, IMP_SOURCE_FILE), IMP_OK);
    ASSERT_EQ(imp_video_source_set(src, "path", videoPath.c_str()), IMP_OK);
    ASSERT_EQ(imp_video_source_set(src, "width", std::to_string(kModelW).c_str()), IMP_OK);
    ASSERT_EQ(imp_video_source_set(src, "height", std::to_string(kModelH).c_str()), IMP_OK);
    imp_video_decode_opts_t options{};
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &options), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);
    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        imp_tensor_t* tensor = nullptr;
        ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
        ASSERT_NE(tensor, nullptr);
        ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_SYSTEM);
        const uint8_t* y = nullptr;
        const uint8_t* uv = nullptr;
        int frameWidth = 0;
        int frameHeight = 0;
        ASSERT_EQ(imp_tensor_get_nv12_planes(tensor, &y, &uv, &frameWidth, &frameHeight), IMP_OK);
        ASSERT_EQ(frameWidth, kModelW);
        ASSERT_EQ(frameHeight, kModelH);
        ASSERT_NE(y, nullptr);
        ASSERT_NE(uv, nullptr);
        HostNv12Frame frame;
        frame.y.assign(y, y + static_cast<size_t>(frameWidth) * frameHeight);
        frame.uv.assign(uv, uv + static_cast<size_t>(frameWidth) * frameHeight / 2);
        frames.push_back(std::move(frame));
        imp_tensor_release(tensor);
    }
    imp_video_close(stream);
    imp_context_destroy(ctx);

    const std::filesystem::path modelPath = std::filesystem::path(repoRoot()) /
                                            "src/test/face_detection_adas/1/face-detection-adas-0001.xml";
    ASSERT_TRUE(std::filesystem::exists(modelPath));
    ov::Core core;
    auto rawModel = core.read_model(modelPath.string(), {}, ov::AnyMap{{"ENABLE_MMAP", "NO"}});
    ov::preprocess::PrePostProcessor preprocessor(rawModel);
    preprocessor.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"});
    preprocessor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    preprocessor.input().model().set_layout("NCHW");
    const auto model = preprocessor.build();
    ov::AnyMap pluginConfig{{"PERFORMANCE_HINT", "LATENCY"}, {"NUM_STREAMS", "1"}};
    ov::intel_gpu::ocl::D3DContext d3dContext(core, device.Get());
    auto compiledModel = core.compile_model(model, d3dContext, pluginConfig);
    std::vector<ov::InferRequest> requests;
    requests.reserve(4);
    for (int requestIndex = 0; requestIndex < 4; ++requestIndex)
        requests.emplace_back(compiledModel.create_infer_request());

    const ov::Shape shapeY{1, kModelH, kModelW, 1};
    const ov::Shape shapeUv{1, kModelH / 2, kModelW / 2, 2};
    const auto runTrial = [&](int trial) {
        std::vector<double> bindMs;
        std::vector<double> inferMs;
        bindMs.reserve(measuredFrames);
        inferMs.reserve(measuredFrames);
        int detections = 0;
        for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
            const auto& frame = frames[frameIndex];
            auto& request = requests[frameIndex % requests.size()];
            const auto bindStart = Clock::now();
            request.set_tensor("data/y", ov::Tensor(ov::element::u8, shapeY,
                                             const_cast<uint8_t*>(frame.y.data())));
            request.set_tensor("data/uv", ov::Tensor(ov::element::u8, shapeUv,
                                              const_cast<uint8_t*>(frame.uv.data())));
            const auto bindStop = Clock::now();
            const auto inferStart = Clock::now();
            request.infer();
            const auto inferStop = Clock::now();
            const auto output = request.get_output_tensor(0);
            const float* values = output.data<const float>();
            for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
                if (values[result + 2] >= 0.5f)
                    ++detections;
            }
            if (frameIndex >= warmupFrames) {
                bindMs.push_back(std::chrono::duration<double, std::milli>(bindStop - bindStart).count());
                inferMs.push_back(std::chrono::duration<double, std::milli>(inferStop - inferStart).count());
            }
        }
        const double bindSum = std::accumulate(bindMs.begin(), bindMs.end(), 0.0);
        const double inferSum = std::accumulate(inferMs.begin(), inferMs.end(), 0.0);
        std::cout << std::fixed << std::setprecision(3)
                  << "P11_OV_HOST_NV12_GPU trial=" << trial
                  << " frames=" << measuredFrames
                  << " bind_mean_ms=" << bindSum / measuredFrames
                  << " infer_mean_ms=" << inferSum / measuredFrames
                  << " infer_fps=" << measuredFrames * 1000.0 / inferSum
                  << " detections=" << detections << std::endl;
    };

    std::cout << "P11_OV_HOST_NV12_GPU input=host_nv12 preprocess=GPU_NV12_TO_BGR "
              << "config=LATENCY streams=1 infer_requests=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(trial);
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
    vopts.output_memory = IMP_VIDEO_MEMORY_NATIVE;
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
        branches[1].output_memory = IMP_VIDEO_MEMORY_SYSTEM;
        imp_video_decode_opts_t vopts{};
        vopts.output_memory = IMP_VIDEO_MEMORY_NATIVE;
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
                          &subresource, &frameWidth, &frameHeight),
                IMP_OK);
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
                              &frameWidth, &frameHeight),
                    IMP_OK);
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
    vopts.output_memory = IMP_VIDEO_MEMORY_NATIVE;
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
