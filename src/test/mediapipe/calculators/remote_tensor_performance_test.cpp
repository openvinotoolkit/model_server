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
 * Windows D3D11 source and remote-tensor tests. Direct tests drive the
 * intel_mpi GStreamer/D3D11 pipeline with OpenVINO or the OVMS C-API. The
 * MediaPipe performance test uses GstVideoSourceCalculator followed by
 * GstCapiInferenceCalculator.
 *
 * Covered:
 *   - Device ownership: GStreamer decodes onto the server-owned ID3D11Device
 *     and the ModelManager GPU model runs on that same device (zero host copy).
 *   - Source EOS lifecycle: a finite file source reaches EOS and releases
 *     cleanly.
 *   - Direct camera capture -> D3D11 inference + H.264 MP4 output (gated
 *     on OVMS_TEST_CAMERA).
 *   - RTSP ingest onto the owned device (gated on OVMS_TEST_RTSP_URL).
 *   - RTSP publication of encoded H.264 (gated on OVMS_TEST_RTSP_OUT).
 *
 * Needs Windows + Intel GPU + GStreamer D3D11 + the face_detection_adas fixture
 * and a test video; required setup failures fail the test. Env overrides:
 *   VIDEO_PATH            (default C:\git\model_server\recording_3_raw.avi)
 *   OVMS_REPO_ROOT        (default C:\git\model_server)
 *   OVMS_PERF_MODEL_DIR   (base model directory containing version 1)
 *   OVMS_PERF_MODEL_XML   (model XML; overrides the default under MODEL_DIR)
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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl2.hpp>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/ovms.h"
#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mediapipe_gst_calculators/gst_video_frame_packet.hpp"
#include "src/test/mediapipe/calculators/performance_benchmark_config.hpp"

#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

extern "C" void* OVMS_MPGetExternalServerHandle() {
    ADD_FAILURE() << "test graph must provide the OVMS server through the SERVER side packet";
    return nullptr;
}

namespace {

using Microsoft::WRL::ComPtr;

constexpr const char* kModelName = benchmark_config::kModelName;
constexpr const char* kInputName = benchmark_config::kInputName;
constexpr int kModelW = benchmark_config::kModelWidth;
constexpr int kModelH = benchmark_config::kModelHeight;
constexpr int kPerformanceWarmupFrames = benchmark_config::kWarmupFrames;
constexpr int kPerformanceMeasuredFrames = benchmark_config::kMeasuredFrames;

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int performanceTrialCount() {
    const char* configuredValue = std::getenv("OVMS_PERF_TRIAL_COUNT");
    if (configuredValue == nullptr || *configuredValue == '\0')
        return 3;
    const int parsedValue = std::atoi(configuredValue);
    return parsedValue > 0 ? parsedValue : 3;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string repoRoot() {
    return envOr("OVMS_REPO_ROOT", "C:\\git\\model_server");
}

std::string modelDir() {
    return envOr("OVMS_PERF_MODEL_DIR", repoRoot() + "\\src\\test\\face_detection_adas");
}

std::string modelXmlPath() {
    return envOr("OVMS_PERF_MODEL_XML", modelDir() + "\\1\\face-detection-adas-0001.xml");
}

ComPtr<ID3D11Device> createDevice() {
    ComPtr<ID3D11Device> device;
    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0,
                                   D3D11_SDK_VERSION, device.GetAddressOf(), &got, nullptr);
    return SUCCEEDED(hr) ? device : nullptr;
}

cl_context createOpenCLGpuContext(cl_device_id* device) {
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0)
        return nullptr;
    std::vector<cl_platform_id> platforms(platformCount);
    if (clGetPlatformIDs(platformCount, platforms.data(), nullptr) != CL_SUCCESS)
        return nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != CL_SUCCESS ||
            deviceCount == 0)
            continue;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, nullptr) != CL_SUCCESS)
            continue;
        cl_int error = CL_SUCCESS;
        cl_context context = clCreateContext(nullptr, 1, device, nullptr, nullptr, &error);
        return error == CL_SUCCESS ? context : nullptr;
    }
    return nullptr;
}

void convertNv12ToBgrNchw(const uint8_t* y, const uint8_t* uv, std::vector<float>* output) {
    const size_t planeSize = static_cast<size_t>(kModelW) * kModelH;
    output->resize(3 * planeSize);
    float* blue = output->data();
    float* green = blue + planeSize;
    float* red = green + planeSize;
    for (int row = 0; row < kModelH; ++row) {
        const size_t uvRow = static_cast<size_t>(row / 2) * kModelW;
        for (int column = 0; column < kModelW; ++column) {
            const float luma = static_cast<float>(y[static_cast<size_t>(row) * kModelW + column]) - 16.0f;
            const size_t uvIndex = uvRow + static_cast<size_t>(column / 2) * 2;
            const float u = static_cast<float>(uv[uvIndex]) - 128.0f;
            const float v = static_cast<float>(uv[uvIndex + 1]) - 128.0f;
            const size_t pixel = static_cast<size_t>(row) * kModelW + column;
            blue[pixel] = std::clamp(1.164f * luma + 2.017f * u, 0.0f, 255.0f);
            green[pixel] = std::clamp(1.164f * luma - 0.392f * u - 0.813f * v, 0.0f, 255.0f);
            red[pixel] = std::clamp(1.164f * luma + 1.596f * v, 0.0f, 255.0f);
        }
    }
}

std::string writeTempConfig() {
    const std::string configuredModelDir = std::filesystem::path(modelDir()).generic_string();
    const std::string cfgPath = repoRoot() + "\\src\\test\\remote_tensor_performance_tmp.json";
    std::ofstream f(cfgPath);
    f << "{\"model_config_list\":[{\"config\":{\"name\":\"" << kModelName
            << "\",\"base_path\":\"" << configuredModelDir
            << "\",\"target_device\":\"GPU\",\"nireq\":4,\"plugin_config\":{\"PERFORMANCE_HINT\":\""
            << benchmark_config::kPerformanceHint << "\",\"NUM_STREAMS\":\""
            << benchmark_config::kNumStreams << "\"}}}]}";
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

struct WindowsD3D11RemoteTensorFixture : public ::testing::Test {
    inline static ComPtr<ID3D11Device> suiteDevice;
    ComPtr<ID3D11Device> device;
    OVMS_Server* server = nullptr;
    std::string videoPath;
    void SetUp() override {
        videoPath = envOr("VIDEO_PATH", repoRoot() + "\\recording_3_raw.avi");
        ASSERT_TRUE(imp_video_d3d11_available()) << "D3D11 surface sharing not available";
        ASSERT_TRUE(fileExists(modelXmlPath())) << "model XML not found: " << modelXmlPath();
        if (!suiteDevice)
            suiteDevice = createDevice();
        device = suiteDevice;
        ASSERT_NE(device.Get(), nullptr) << "could not create a hardware ID3D11Device";

        const std::string cfg = writeTempConfig();
        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_TRUE(capiOk(OVMS_ServerSettingsNew(&serverSettings)));
        ASSERT_TRUE(capiOk(OVMS_ModelsSettingsNew(&modelsSettings)));
        ASSERT_TRUE(capiOk(OVMS_ServerSettingsSetGrpcPort(serverSettings, 9719)));
        ASSERT_TRUE(capiOk(OVMS_ModelsSettingsSetConfigPath(modelsSettings, cfg.c_str())));
        ASSERT_TRUE(capiOk(OVMS_ServerNew(&server)));

        // Register the exact same COM device pointer with OVMS and intel_mpi
        // before loading the GPU model. Tests compare the decoder-reported
        // pointer with this device and then infer through the OVMS C-API.
        ASSERT_TRUE(capiOk(OVMS_ServerSetGlobalD3D11Device(server, device.Get())));
        imp_video_set_d3d11_device(device.Get());

        ASSERT_TRUE(capiOk(OVMS_ServerStartFromConfigurationFile(server, serverSettings, modelsSettings)));
    }

    void TearDown() override {
        if (server) {
            OVMS_ServerSetGlobalD3D11Device(server, nullptr);
            imp_video_set_d3d11_device(nullptr);
            OVMS_ServerDelete(server);
        }
        std::filesystem::remove(repoRoot() + "\\src\\test\\remote_tensor_performance_tmp.json");
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
        OVMS_DataType dt = OVMS_DATATYPE_UNDEFINED;
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
        OVMS_DataType dt = OVMS_DATATYPE_UNDEFINED;
        const int64_t* shape = nullptr;
        size_t dims = 0;
        OVMS_BufferType bt = OVMS_BUFFERTYPE_CPU;
        uint32_t did = 0;
        const char* oname = nullptr;
        int detections = -1;
        if (capiOk(OVMS_InferenceResponseOutput(response, id, &oname, &dt, &shape, &dims,
                                                  &out, &bytes, &bt, &did)) && out) {
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

// Verifies pointer identity, not merely adapter identity: intel_mpi must decode
// on the same ID3D11Device registered with OVMS before C-API inference.
TEST_F(WindowsD3D11RemoteTensorFixture, DecoderAndOvmsCapiInferenceUseSameRegisteredD3D11Device) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

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

// Direct path: intel_mpi decode followed by a new OVMS C-API request per frame.
TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithOvmsCapiInferencePerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
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
        options.use_d3d11_surface_memory = true;
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
            measuredStop - measuredStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_DIRECT_D3D11 mode=" << (runInference ? (reuseRequest ? "decode_plus_capi_reuse_request" : "decode_plus_capi_new_request") : (boundedDecode ? "decode_only_bounded" : "decode_only"))
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

    std::cout << "REMOTE_TENSOR_DIRECT_D3D11 config=LATENCY streams=1 nireq=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(true, true, false, trial);
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithD3D11Nv12SurfacePerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;
    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;
    for (int trial = 1; trial <= trialCount; ++trial) {
        imp_context_t* context = nullptr;
        ASSERT_EQ(imp_context_create(&context, nullptr), IMP_OK);
        imp_video_source_t* source = nullptr;
        ASSERT_EQ(imp_video_source_create(&source, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.use_d3d11_surface_memory = true;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, source, context, &options), IMP_OK)
            << imp_context_get_error(context);
        imp_video_source_destroy(source);
        std::vector<double> decodeMs;
        decodeMs.reserve(measuredFrames);
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            imp_tensor_t* tensor = nullptr;
            const auto start = Clock::now();
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            const auto stop = Clock::now();
            ASSERT_NE(tensor, nullptr);
            ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE);
            if (frameIndex >= warmupFrames)
                decodeMs.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            imp_tensor_release(tensor);
        }
        imp_video_close(stream);
        imp_context_destroy(context);
        const double sum = std::accumulate(decodeMs.begin(), decodeMs.end(), 0.0);
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_DECODE_D3D11_NV12 trial=" << trial
                  << " frames=" << measuredFrames
                  << " decode_mean_ms=" << sum / measuredFrames
                  << " decode_fps=" << measuredFrames * 1000.0 / sum << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, OvmsCapiInferenceWithD3D11Nv12Performance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;
    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;
    for (int trial = 1; trial <= trialCount; ++trial) {
        imp_context_t* context = nullptr;
        ASSERT_EQ(imp_context_create(&context, nullptr), IMP_OK);
        imp_video_source_t* source = nullptr;
        ASSERT_EQ(imp_video_source_create(&source, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.use_d3d11_surface_memory = true;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, source, context, &options), IMP_OK)
            << imp_context_get_error(context);
        imp_video_source_destroy(source);
        OVMS_InferenceRequest* request = nullptr;
        const std::string yName = std::string(kInputName) + "/y";
        const std::string uvName = std::string(kInputName) + "/uv";
        const int64_t shapeY[] = {1, kModelH, kModelW, 1};
        const int64_t shapeUV[] = {1, kModelH / 2, kModelW / 2, 2};
        ASSERT_TRUE(capiOk(OVMS_InferenceRequestNew(&request, server, kModelName, 1)));
        ASSERT_TRUE(capiOk(OVMS_InferenceRequestAddInput(
            request, yName.c_str(), OVMS_DATATYPE_U8, shapeY, 4)));
        ASSERT_TRUE(capiOk(OVMS_InferenceRequestAddInput(
            request, uvName.c_str(), OVMS_DATATYPE_U8, shapeUV, 4)));
        bool hasBoundData = false;
        std::vector<double> capiMs;
        capiMs.reserve(measuredFrames);
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            imp_tensor_t* tensor = nullptr;
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            ASSERT_NE(tensor, nullptr);
            void* texture = nullptr;
            void* decodeDevice = nullptr;
            uint32_t subresource = 0;
            int frameWidth = 0;
            int frameHeight = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource,
                &frameWidth, &frameHeight), IMP_OK);
            ASSERT_EQ(decodeDevice, static_cast<void*>(device.Get()));
            ASSERT_EQ(subresource, 0u);
            const auto start = Clock::now();
            ASSERT_GE(runReusableCapiInference(
                          request, &hasBoundData, texture, frameWidth, frameHeight),
                0);
            const auto stop = Clock::now();
            if (frameIndex >= warmupFrames)
                capiMs.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            imp_tensor_release(tensor);
        }
        OVMS_InferenceRequestDelete(request);
        imp_video_close(stream);
        imp_context_destroy(context);
        const double sum = std::accumulate(capiMs.begin(), capiMs.end(), 0.0);
        std::sort(capiMs.begin(), capiMs.end());
        const auto percentile = [&capiMs](double fraction) {
            return capiMs[static_cast<size_t>(fraction * (capiMs.size() - 1))];
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_OVMS_CAPI_D3D11_NV12 trial=" << trial
                  << " frames=" << measuredFrames
                  << " capi_mean_ms=" << sum / measuredFrames
                  << " capi_p50_ms=" << percentile(0.50)
                  << " capi_p90_ms=" << percentile(0.90)
                  << " capi_p99_ms=" << percentile(0.99)
                  << " capi_fps=" << measuredFrames * 1000.0 / sum << std::endl;
    }
}

// Current split architecture with exactly two calculators.
TEST_F(WindowsD3D11RemoteTensorFixture, MediaPipeFileSourceAndOvmsCapiInferenceCalculatorsPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    constexpr int totalFrames = warmupFrames + measuredFrames;
    using Clock = std::chrono::steady_clock;
    const std::string graphText = R"pb(
input_side_packet: "VIDEO_PATH:video_path"
input_side_packet: "WIDTH:width"
input_side_packet: "HEIGHT:height"
input_side_packet: "MAX_FRAMES:max_frames"
input_side_packet: "SERVABLE_NAME:servable_name"
input_side_packet: "SERVABLE_VERSION:servable_version"
input_side_packet: "INPUT_NAME:input_name"
input_side_packet: "SERVER:server"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstVideoSourceCalculator"
  input_side_packet: "VIDEO_PATH:video_path"
  input_side_packet: "WIDTH:width"
  input_side_packet: "HEIGHT:height"
  input_side_packet: "MAX_FRAMES:max_frames"
  output_stream: "FRAME:frame"
}
node {
  calculator: "GstCapiInferenceCalculator"
  input_stream: "FRAME:frame"
  input_side_packet: "SERVABLE_NAME:servable_name"
  input_side_packet: "SERVABLE_VERSION:servable_version"
  input_side_packet: "INPUT_NAME:input_name"
  input_side_packet: "SERVER:server"
  output_stream: "DETECTIONS:detections"
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_TWO_CALCULATOR_GRAPH graph=source_plus_capi_inference "
              << "config=LATENCY streams=1 nireq=4 warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        ::mediapipe::CalculatorGraph graph;
        const auto initializeStart = Clock::now();
        ASSERT_TRUE(graph.Initialize(config).ok());
        const auto initializeStop = Clock::now();
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
            {"width", ::mediapipe::MakePacket<int>(kModelW)},
            {"height", ::mediapipe::MakePacket<int>(kModelH)},
            {"max_frames", ::mediapipe::MakePacket<int>(totalFrames)},
            {"servable_name", ::mediapipe::MakePacket<std::string>(kModelName)},
            {"servable_version", ::mediapipe::MakePacket<int>(1)},
            {"input_name", ::mediapipe::MakePacket<std::string>(kInputName)},
            {"server", ::mediapipe::MakePacket<OVMS_Server*>(server)},
        };
        const auto startRunStart = Clock::now();
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());
        const auto startRunStop = Clock::now();
        Clock::time_point firstOutputStop;
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int frames = 0;
        int detections = 0;
        ::mediapipe::Packet packet;
        while (poller.Next(&packet)) {
            ++frames;
            detections += packet.Get<int>();
            if (frames == 1)
                firstOutputStop = Clock::now();
            if (frames == warmupFrames)
                measuredStart = Clock::now();
            if (frames == totalFrames)
                measuredStop = Clock::now();
        }
        const auto drainStart = Clock::now();
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        const auto drainStop = Clock::now();
        ASSERT_EQ(frames, totalFrames);
        EXPECT_GE(detections, totalFrames);
        const double initializeMs = std::chrono::duration<double, std::milli>(
            initializeStop - initializeStart).count();
        const double sourceOpenMs = std::chrono::duration<double, std::milli>(
            startRunStop - startRunStart).count();
        const double firstFrameMs = std::chrono::duration<double, std::milli>(
            firstOutputStop - startRunStop).count();
        const double steadyMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart).count();
        const double drainMs = std::chrono::duration<double, std::milli>(
            drainStop - drainStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_TWO_CALCULATOR_GRAPH trial=" << trial
                  << " frames=" << measuredFrames
                  << " graph_initialize_ms=" << initializeMs
                  << " source_open_ms=" << sourceOpenMs
                  << " first_frame_ms=" << firstFrameMs
                  << " steady_fps=" << measuredFrames * 1000.0 / steadyMs
                  << " graph_drain_ms=" << drainMs
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, PureOpenVinoInferenceWithD3D11Nv12InCalculatorPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;
    const std::string configuredModelPath = std::filesystem::path(modelXmlPath()).generic_string();
    const std::string graphText = R"pb(
input_stream: "FRAME:frame"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstOpenVinoInferenceCalculator"
  input_stream: "FRAME:frame"
  output_stream: "DETECTIONS:detections"
  options {
    [mediapipe.GstOpenVinoInferenceCalculatorOptions.ext] {
      model_path: ")pb" + configuredModelPath + R"pb("
      input_name: "data"
      performance_hint: "LATENCY"
      num_streams: 1
    }
  }
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_PURE_OV_CALCULATOR input=D3D11_NV12 config=LATENCY streams=1 "
              << "warmup_frames=" << warmupFrames << " measured_frames=" << measuredFrames
              << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        imp_context_t* context = nullptr;
        ASSERT_EQ(imp_context_create(&context, nullptr), IMP_OK);
        imp_video_source_t* source = nullptr;
        ASSERT_EQ(imp_video_source_create(&source, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.use_d3d11_surface_memory = true;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, source, context, &options), IMP_OK)
            << imp_context_get_error(context);
        imp_video_source_destroy(source);

        ::mediapipe::CalculatorGraph graph;
        ASSERT_TRUE(graph.Initialize(config).ok());
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        ASSERT_TRUE(graph.StartRun({}).ok());

        std::vector<double> calculatorMs;
        calculatorMs.reserve(measuredFrames);
        int detections = 0;
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            imp_tensor_t* tensor = nullptr;
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            ASSERT_NE(tensor, nullptr);
            void* texture = nullptr;
            void* decodeDevice = nullptr;
            uint32_t subresource = 0;
            int frameWidth = 0;
            int frameHeight = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource,
                &frameWidth, &frameHeight), IMP_OK);
            ASSERT_EQ(decodeDevice, static_cast<void*>(device.Get()));
            ASSERT_EQ(subresource, 0u);
            ::mediapipe::GstVideoFramePacket frame;
            frame.resource = ::mediapipe::D3D11Frame{reinterpret_cast<uintptr_t>(texture), subresource};
            frame.owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
            frame.width = frameWidth;
            frame.height = frameHeight;

            const auto calculatorStart = Clock::now();
            ASSERT_TRUE(graph.AddPacketToInputStream(
                                  "frame", ::mediapipe::MakePacket<::mediapipe::GstVideoFramePacket>(
                                               std::move(frame))
                                               .At(::mediapipe::Timestamp(frameIndex)))
                            .ok());
            ::mediapipe::Packet output;
            ASSERT_TRUE(poller.Next(&output));
            const auto calculatorStop = Clock::now();
            detections += output.Get<int>();
            if (frameIndex >= warmupFrames) {
                calculatorMs.push_back(std::chrono::duration<double, std::milli>(
                    calculatorStop - calculatorStart).count());
            }
        }
        ASSERT_TRUE(graph.CloseInputStream("frame").ok());
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        imp_video_close(stream);
        imp_context_destroy(context);

        const double calculatorSum = std::accumulate(calculatorMs.begin(), calculatorMs.end(), 0.0);
        std::sort(calculatorMs.begin(), calculatorMs.end());
        const auto percentile = [&calculatorMs](double fraction) {
            return calculatorMs[static_cast<size_t>(fraction * (calculatorMs.size() - 1))];
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_PURE_OV_CALCULATOR trial=" << trial
                  << " frames=" << measuredFrames
                  << " calculator_mean_ms=" << calculatorSum / measuredFrames
                  << " calculator_p50_ms=" << percentile(0.50)
                  << " calculator_p90_ms=" << percentile(0.90)
                  << " calculator_p99_ms=" << percentile(0.99)
                  << " calculator_fps=" << measuredFrames * 1000.0 / calculatorSum
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, OvmsCapiInferenceWithD3D11Nv12InCalculatorPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;
    const std::string graphText = R"pb(
input_stream: "FRAME:frame"
input_side_packet: "SERVABLE_NAME:servable_name"
input_side_packet: "SERVABLE_VERSION:servable_version"
input_side_packet: "INPUT_NAME:input_name"
input_side_packet: "SERVER:server"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstCapiInferenceCalculator"
  input_stream: "FRAME:frame"
  input_side_packet: "SERVABLE_NAME:servable_name"
  input_side_packet: "SERVABLE_VERSION:servable_version"
  input_side_packet: "INPUT_NAME:input_name"
  input_side_packet: "SERVER:server"
  output_stream: "DETECTIONS:detections"
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_OVMS_CAPI_CALCULATOR input=D3D11_NV12 config=LATENCY streams=1 "
              << "warmup_frames=" << warmupFrames << " measured_frames=" << measuredFrames
              << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        imp_context_t* context = nullptr;
        ASSERT_EQ(imp_context_create(&context, nullptr), IMP_OK);
        imp_video_source_t* source = nullptr;
        ASSERT_EQ(imp_video_source_create(&source, IMP_SOURCE_FILE), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "path", videoPath.c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "width", std::to_string(kModelW).c_str()), IMP_OK);
        ASSERT_EQ(imp_video_source_set(source, "height", std::to_string(kModelH).c_str()), IMP_OK);
        imp_video_decode_opts_t options{};
        options.use_d3d11_surface_memory = true;
        imp_video_stream_t* stream = nullptr;
        ASSERT_EQ(imp_video_open(&stream, source, context, &options), IMP_OK)
            << imp_context_get_error(context);
        imp_video_source_destroy(source);

        ::mediapipe::CalculatorGraph graph;
        ASSERT_TRUE(graph.Initialize(config).ok());
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"servable_name", ::mediapipe::MakePacket<std::string>(kModelName)},
            {"servable_version", ::mediapipe::MakePacket<int>(1)},
            {"input_name", ::mediapipe::MakePacket<std::string>(kInputName)},
            {"server", ::mediapipe::MakePacket<OVMS_Server*>(server)},
        };
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());

        std::vector<double> calculatorMs;
        calculatorMs.reserve(measuredFrames);
        int detections = 0;
        for (int frameIndex = 0; frameIndex < warmupFrames + measuredFrames; ++frameIndex) {
            imp_tensor_t* tensor = nullptr;
            ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK) << frameIndex;
            ASSERT_NE(tensor, nullptr);
            void* texture = nullptr;
            void* decodeDevice = nullptr;
            uint32_t subresource = 0;
            int frameWidth = 0;
            int frameHeight = 0;
            ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource,
                &frameWidth, &frameHeight), IMP_OK);
            ASSERT_EQ(decodeDevice, static_cast<void*>(device.Get()));
            ASSERT_EQ(subresource, 0u);
            ::mediapipe::GstVideoFramePacket frame;
            frame.resource = ::mediapipe::D3D11Frame{reinterpret_cast<uintptr_t>(texture), subresource};
            frame.owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
            frame.width = frameWidth;
            frame.height = frameHeight;

            const auto calculatorStart = Clock::now();
            ASSERT_TRUE(graph.AddPacketToInputStream(
                                  "frame", ::mediapipe::MakePacket<::mediapipe::GstVideoFramePacket>(
                                               std::move(frame))
                                               .At(::mediapipe::Timestamp(frameIndex)))
                            .ok());
            ::mediapipe::Packet output;
            ASSERT_TRUE(poller.Next(&output));
            const auto calculatorStop = Clock::now();
            detections += output.Get<int>();
            if (frameIndex >= warmupFrames) {
                calculatorMs.push_back(std::chrono::duration<double, std::milli>(
                    calculatorStop - calculatorStart).count());
            }
        }
        ASSERT_TRUE(graph.CloseInputStream("frame").ok());
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        imp_video_close(stream);
        imp_context_destroy(context);

        const double calculatorSum = std::accumulate(calculatorMs.begin(), calculatorMs.end(), 0.0);
        std::sort(calculatorMs.begin(), calculatorMs.end());
        const auto percentile = [&calculatorMs](double fraction) {
            return calculatorMs[static_cast<size_t>(fraction * (calculatorMs.size() - 1))];
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_OVMS_CAPI_CALCULATOR trial=" << trial
                  << " frames=" << measuredFrames
                  << " calculator_mean_ms=" << calculatorSum / measuredFrames
                  << " calculator_p50_ms=" << percentile(0.50)
                  << " calculator_p90_ms=" << percentile(0.90)
                  << " calculator_p99_ms=" << percentile(0.99)
                  << " calculator_fps=" << measuredFrames * 1000.0 / calculatorSum
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithOpenVinoSourceAndInferenceCalculatorsPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    constexpr int totalFrames = warmupFrames + measuredFrames;
    using Clock = std::chrono::steady_clock;
    const std::string configuredModelPath = std::filesystem::path(modelXmlPath()).generic_string();
    const std::string graphText = R"pb(
input_side_packet: "VIDEO_PATH:video_path"
input_side_packet: "WIDTH:width"
input_side_packet: "HEIGHT:height"
input_side_packet: "MAX_FRAMES:max_frames"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstVideoSourceCalculator"
  input_side_packet: "VIDEO_PATH:video_path"
  input_side_packet: "WIDTH:width"
  input_side_packet: "HEIGHT:height"
  input_side_packet: "MAX_FRAMES:max_frames"
  output_stream: "FRAME:frame"
}
node {
  calculator: "GstOpenVinoInferenceCalculator"
  input_stream: "FRAME:frame"
  output_stream: "DETECTIONS:detections"
  options {
    [mediapipe.GstOpenVinoInferenceCalculatorOptions.ext] {
      model_path: ")pb" + configuredModelPath + R"pb("
      input_name: "data"
      performance_hint: "LATENCY"
      num_streams: 1
    }
  }
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_TWO_CALCULATOR_DIRECT_OV_GRAPH graph=source_plus_openvino_inference "
              << "input=D3D11_NV12 config=LATENCY streams=1 warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        ::mediapipe::CalculatorGraph graph;
        const auto initializeStart = Clock::now();
        ASSERT_TRUE(graph.Initialize(config).ok());
        const auto initializeStop = Clock::now();
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
            {"width", ::mediapipe::MakePacket<int>(kModelW)},
            {"height", ::mediapipe::MakePacket<int>(kModelH)},
            {"max_frames", ::mediapipe::MakePacket<int>(totalFrames)},
        };
        const auto startRunStart = Clock::now();
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());
        const auto startRunStop = Clock::now();
        Clock::time_point firstOutputStop;
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int frames = 0;
        int detections = 0;
        ::mediapipe::Packet packet;
        while (poller.Next(&packet)) {
            ++frames;
            detections += packet.Get<int>();
            if (frames == 1)
                firstOutputStop = Clock::now();
            if (frames == warmupFrames)
                measuredStart = Clock::now();
            if (frames == totalFrames)
                measuredStop = Clock::now();
        }
        const auto drainStart = Clock::now();
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        const auto drainStop = Clock::now();
        ASSERT_EQ(frames, totalFrames);
        EXPECT_GE(detections, totalFrames);
        const double initializeMs = std::chrono::duration<double, std::milli>(
            initializeStop - initializeStart).count();
        const double sourceOpenMs = std::chrono::duration<double, std::milli>(
            startRunStop - startRunStart).count();
        const double firstFrameMs = std::chrono::duration<double, std::milli>(
            firstOutputStop - startRunStop).count();
        const double steadyMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart).count();
        const double drainMs = std::chrono::duration<double, std::milli>(
            drainStop - drainStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_TWO_CALCULATOR_DIRECT_OV_GRAPH trial=" << trial
                  << " frames=" << measuredFrames
                  << " graph_initialize_ms=" << initializeMs
                  << " source_open_ms=" << sourceOpenMs
                  << " first_frame_ms=" << firstFrameMs
                  << " steady_fps=" << measuredFrames * 1000.0 / steadyMs
                  << " graph_drain_ms=" << drainMs
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithOpenVinoSingleCalculatorPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    constexpr int totalFrames = warmupFrames + measuredFrames;
    using Clock = std::chrono::steady_clock;
    const std::string configuredModelPath = std::filesystem::path(modelXmlPath()).generic_string();
    const std::string graphText = R"pb(
input_side_packet: "VIDEO_PATH:video_path"
input_side_packet: "WIDTH:width"
input_side_packet: "HEIGHT:height"
input_side_packet: "MAX_FRAMES:max_frames"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstVideoOpenVinoInferenceCalculator"
  input_side_packet: "VIDEO_PATH:video_path"
  input_side_packet: "WIDTH:width"
  input_side_packet: "HEIGHT:height"
  input_side_packet: "MAX_FRAMES:max_frames"
  output_stream: "DETECTIONS:detections"
  options {
    [mediapipe.GstOpenVinoInferenceCalculatorOptions.ext] {
      model_path: ")pb" + configuredModelPath + R"pb("
      input_name: "data"
      performance_hint: "LATENCY"
      num_streams: 1
    }
  }
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_SINGLE_CALCULATOR_DIRECT_OV graph=decode_plus_openvino_inference "
              << "input=D3D11_NV12 config=LATENCY streams=1 warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        ::mediapipe::CalculatorGraph graph;
        const auto initializeStart = Clock::now();
        ASSERT_TRUE(graph.Initialize(config).ok());
        const auto initializeStop = Clock::now();
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
            {"width", ::mediapipe::MakePacket<int>(kModelW)},
            {"height", ::mediapipe::MakePacket<int>(kModelH)},
            {"max_frames", ::mediapipe::MakePacket<int>(totalFrames)},
        };
        const auto startRunStart = Clock::now();
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());
        const auto startRunStop = Clock::now();
        Clock::time_point firstOutputStop;
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int frames = 0;
        int detections = 0;
        ::mediapipe::Packet packet;
        while (poller.Next(&packet)) {
            ++frames;
            detections += packet.Get<int>();
            if (frames == 1)
                firstOutputStop = Clock::now();
            if (frames == warmupFrames)
                measuredStart = Clock::now();
            if (frames == totalFrames)
                measuredStop = Clock::now();
        }
        const auto drainStart = Clock::now();
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        const auto drainStop = Clock::now();
        ASSERT_EQ(frames, totalFrames);
        EXPECT_GE(detections, totalFrames);
        const double initializeMs = std::chrono::duration<double, std::milli>(
            initializeStop - initializeStart).count();
        const double sourceOpenMs = std::chrono::duration<double, std::milli>(
            startRunStop - startRunStart).count();
        const double firstFrameMs = std::chrono::duration<double, std::milli>(
            firstOutputStop - startRunStop).count();
        const double steadyMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart).count();
        const double drainMs = std::chrono::duration<double, std::milli>(
            drainStop - drainStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_SINGLE_CALCULATOR_DIRECT_OV trial=" << trial
                  << " frames=" << measuredFrames
                  << " graph_initialize_ms=" << initializeMs
                  << " source_open_ms=" << sourceOpenMs
                  << " first_frame_ms=" << firstFrameMs
                  << " steady_fps=" << measuredFrames * 1000.0 / steadyMs
                  << " graph_drain_ms=" << drainMs
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithOpenVinoGstreamerElementPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    constexpr int totalFrames = warmupFrames + measuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;
    const std::string configuredModelPath = std::filesystem::path(modelXmlPath()).generic_string();
    const std::string graphText = R"pb(
input_side_packet: "VIDEO_PATH:video_path"
input_side_packet: "WIDTH:width"
input_side_packet: "HEIGHT:height"
input_side_packet: "MAX_FRAMES:max_frames"
output_stream: "FRAMES:frames"
node {
  calculator: "GstVideoOpenVinoPipelineInferenceCalculator"
  input_side_packet: "VIDEO_PATH:video_path"
  input_side_packet: "WIDTH:width"
  input_side_packet: "HEIGHT:height"
  input_side_packet: "MAX_FRAMES:max_frames"
  output_stream: "FRAMES:frames"
  options {
    [mediapipe.GstOpenVinoInferenceCalculatorOptions.ext] {
      model_path: ")pb" + configuredModelPath + R"pb("
      input_name: "data"
      performance_hint: "LATENCY"
      num_streams: 1
    }
  }
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_GSTREAMER_ELEMENT_DIRECT_OV graph=d3d11convert_plus_ovd3d11infer "
              << "input=D3D11_NV12 config=LATENCY streams=1 warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        ::mediapipe::CalculatorGraph graph;
        ASSERT_TRUE(graph.Initialize(config).ok());
        auto pollerOr = graph.AddOutputStreamPoller("frames");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
            {"width", ::mediapipe::MakePacket<int>(kModelW)},
            {"height", ::mediapipe::MakePacket<int>(kModelH)},
            {"max_frames", ::mediapipe::MakePacket<int>(totalFrames)},
        };
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int frames = 0;
        int inferredFrames = 0;
        ::mediapipe::Packet packet;
        while (poller.Next(&packet)) {
            ++frames;
            inferredFrames += packet.Get<int>();
            if (frames == warmupFrames)
                measuredStart = Clock::now();
            if (frames == totalFrames)
                measuredStop = Clock::now();
        }
        const absl::Status doneStatus = graph.WaitUntilDone();
        ASSERT_TRUE(doneStatus.ok()) << doneStatus.message();
        ASSERT_EQ(frames, totalFrames);
        ASSERT_EQ(inferredFrames, totalFrames);
        const double steadyMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_GSTREAMER_ELEMENT_DIRECT_OV trial=" << trial
                  << " frames=" << measuredFrames
                  << " steady_fps=" << measuredFrames * 1000.0 / steadyMs
                  << " inferred_frames=" << inferredFrames << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeWithOvmsCapiSingleCalculatorPerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    constexpr int totalFrames = warmupFrames + measuredFrames;
    using Clock = std::chrono::steady_clock;
    const std::string graphText = R"pb(
input_side_packet: "VIDEO_PATH:video_path"
input_side_packet: "WIDTH:width"
input_side_packet: "HEIGHT:height"
input_side_packet: "MAX_FRAMES:max_frames"
input_side_packet: "SERVABLE_NAME:servable_name"
input_side_packet: "SERVABLE_VERSION:servable_version"
input_side_packet: "INPUT_NAME:input_name"
input_side_packet: "SERVER:server"
output_stream: "DETECTIONS:detections"
node {
  calculator: "GstVideoCapiInferenceCalculator"
  input_side_packet: "VIDEO_PATH:video_path"
  input_side_packet: "WIDTH:width"
  input_side_packet: "HEIGHT:height"
  input_side_packet: "MAX_FRAMES:max_frames"
  input_side_packet: "SERVABLE_NAME:servable_name"
  input_side_packet: "SERVABLE_VERSION:servable_version"
  input_side_packet: "INPUT_NAME:input_name"
  input_side_packet: "SERVER:server"
  output_stream: "DETECTIONS:detections"
}
)pb";
    const auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(graphText);

    std::cout << "REMOTE_TENSOR_SINGLE_CALCULATOR_OVMS_CAPI graph=decode_plus_capi_inference "
              << "input=D3D11_NV12 config=LATENCY streams=1 warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        ::mediapipe::CalculatorGraph graph;
        ASSERT_TRUE(graph.Initialize(config).ok());
        auto pollerOr = graph.AddOutputStreamPoller("detections");
        ASSERT_TRUE(pollerOr.ok());
        ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());
        const std::map<std::string, ::mediapipe::Packet> sidePackets = {
            {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
            {"width", ::mediapipe::MakePacket<int>(kModelW)},
            {"height", ::mediapipe::MakePacket<int>(kModelH)},
            {"max_frames", ::mediapipe::MakePacket<int>(totalFrames)},
            {"servable_name", ::mediapipe::MakePacket<std::string>(kModelName)},
            {"servable_version", ::mediapipe::MakePacket<int>(1)},
            {"input_name", ::mediapipe::MakePacket<std::string>(kInputName)},
            {"server", ::mediapipe::MakePacket<OVMS_Server*>(server)},
        };
        ASSERT_TRUE(graph.StartRun(sidePackets).ok());
        Clock::time_point measuredStart;
        Clock::time_point measuredStop;
        int frames = 0;
        int detections = 0;
        ::mediapipe::Packet packet;
        while (poller.Next(&packet)) {
            ++frames;
            detections += packet.Get<int>();
            if (frames == warmupFrames)
                measuredStart = Clock::now();
            if (frames == totalFrames)
                measuredStop = Clock::now();
        }
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        ASSERT_EQ(frames, totalFrames);
        EXPECT_GE(detections, totalFrames);
        const double steadyMs = std::chrono::duration<double, std::milli>(
            measuredStop - measuredStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_SINGLE_CALCULATOR_OVMS_CAPI trial=" << trial
                  << " frames=" << measuredFrames
                  << " steady_fps=" << measuredFrames * 1000.0 / steadyMs
                  << " detections=" << detections << std::endl;
    }
}

TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeAndOpenVinoD3D11RemoteTensorInferencePerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    using Clock = std::chrono::steady_clock;

    const std::filesystem::path modelPath = modelXmlPath();
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
        options.use_d3d11_surface_memory = true;
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
            measuredStop - measuredStart).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_PURE_OV_D3D11 trial=" << trial
                  << " frames=" << measuredFrames
                  << " loop_wall_fps=" << measuredFrames * 1000.0 / loopWallMs
                  << " decode_mean_ms=" << decodeSum / measuredFrames
                  << " infer_mean_ms=" << inferSum / measuredFrames
                  << " infer_p50_ms=" << percentile(0.50)
                  << " infer_p90_ms=" << percentile(0.90)
                  << " infer_p99_ms=" << percentile(0.99)
                  << " detections=" << detections << std::endl;
    };

    std::cout << "REMOTE_TENSOR_PURE_OV_D3D11 config=LATENCY streams=1 infer_requests=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(trial);
}

TEST_F(WindowsD3D11RemoteTensorFixture, OpenVinoHostNv12GpuPreprocessingInferencePerformance) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
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

    const std::filesystem::path modelPath = modelXmlPath();
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
                  << "REMOTE_TENSOR_OV_HOST_NV12_GPU trial=" << trial
                  << " frames=" << measuredFrames
                  << " bind_mean_ms=" << bindSum / measuredFrames
                  << " infer_mean_ms=" << inferSum / measuredFrames
                  << " infer_fps=" << measuredFrames * 1000.0 / inferSum
                  << " detections=" << detections << std::endl;
    };

    std::cout << "REMOTE_TENSOR_OV_HOST_NV12_GPU input=host_nv12 preprocess=GPU_NV12_TO_BGR "
              << "config=LATENCY streams=1 infer_requests=4 warmup_frames="
              << warmupFrames << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial)
        runTrial(trial);
}

TEST(OpenVinoGpuPerformance, OpenClRemoteFp32TensorInferencePerformance) {
    constexpr int warmupFrames = kPerformanceWarmupFrames;
    constexpr int measuredFrames = kPerformanceMeasuredFrames;
    const int trialCount = performanceTrialCount();
    constexpr int requestCount = 4;
    const std::string videoPath = envOr("VIDEO_PATH", repoRoot() + "\\recording_3_raw.avi");
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;
    ASSERT_TRUE(fileExists(modelXmlPath())) << "model XML not found: " << modelXmlPath();

    std::vector<std::vector<float>> preparedFrames;
    preparedFrames.reserve(measuredFrames);
    imp_context_t* decodeContext = nullptr;
    ASSERT_EQ(imp_context_create(&decodeContext, nullptr), IMP_OK);
    imp_video_source_t* source = nullptr;
    ASSERT_EQ(imp_video_source_create(&source, IMP_SOURCE_FILE), IMP_OK);
    ASSERT_EQ(imp_video_source_set(source, "path", videoPath.c_str()), IMP_OK);
    ASSERT_EQ(imp_video_source_set(source, "width", std::to_string(kModelW).c_str()), IMP_OK);
    ASSERT_EQ(imp_video_source_set(source, "height", std::to_string(kModelH).c_str()), IMP_OK);
    imp_video_decode_opts_t options{};
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, source, decodeContext, &options), IMP_OK)
        << imp_context_get_error(decodeContext);
    imp_video_source_destroy(source);
    for (int frameIndex = 0; frameIndex < measuredFrames; ++frameIndex) {
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
        preparedFrames.emplace_back();
        convertNv12ToBgrNchw(y, uv, &preparedFrames.back());
        imp_tensor_release(tensor);
    }
    imp_video_close(stream);
    imp_context_destroy(decodeContext);

    ov::Core core;
    auto model = core.read_model(modelXmlPath());
    ASSERT_EQ(model->inputs().size(), 1u);
    cl_device_id deviceId = nullptr;
    cl_context rawContext = createOpenCLGpuContext(&deviceId);
    ASSERT_NE(rawContext, nullptr) << "no OpenCL GPU context";
    std::unique_ptr<_cl_context, decltype(&clReleaseContext)> contextOwner(rawContext, clReleaseContext);
    cl::Context context(rawContext, true);
    cl::CommandQueue queue(context, cl::Device(deviceId));
    ov::intel_gpu::ocl::ClContext ovContext(core, rawContext, 0);
    ov::AnyMap pluginConfig{{"PERFORMANCE_HINT", "LATENCY"}, {"NUM_STREAMS", "1"}};
    auto compiledModel = core.compile_model(model, ovContext, pluginConfig);
    const size_t inputBytes = ov::shape_size(model->input(0).get_shape()) * sizeof(float);
    cl_int error = CL_SUCCESS;
    cl::Buffer inputBuffer(context, CL_MEM_READ_WRITE, inputBytes, nullptr, &error);
    ASSERT_EQ(error, CL_SUCCESS);
    const auto inputTensor = ovContext.create_tensor(
        model->input(0).get_element_type(), model->input(0).get_shape(), inputBuffer);
    const size_t outputBytes = ov::shape_size(model->output(0).get_shape()) * sizeof(float);
    cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputBytes, nullptr, &error);
    ASSERT_EQ(error, CL_SUCCESS);
    const auto outputTensor = ovContext.create_tensor(
        model->output(0).get_element_type(), model->output(0).get_shape(), outputBuffer);
    std::vector<ov::InferRequest> requests;
    requests.reserve(requestCount);
    for (int requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        requests.emplace_back(compiledModel.create_infer_request());
        requests.back().set_tensor(model->input(0).get_any_name(), inputTensor);
        requests.back().set_tensor(model->output(0).get_any_name(), outputTensor);
    }
    const auto uploadFrame = [&](int frameIndex) {
        return queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputBytes,
            preparedFrames[frameIndex % preparedFrames.size()].data());
    };
    for (int frameIndex = 0; frameIndex < warmupFrames; ++frameIndex) {
        ASSERT_EQ(uploadFrame(frameIndex), CL_SUCCESS);
        requests[frameIndex % requests.size()].infer();
    }

    std::cout << "REMOTE_TENSOR_OPENCL_CEILING input=OpenCL_REMOTE_FP32 "
              << "preprocess=BGR_NCHW_CPU_OUTSIDE_TIMER config=LATENCY streams=1 "
              << "infer_requests=" << requests.size() << " warmup_frames=" << warmupFrames
              << " measured_frames=" << measuredFrames << std::endl;
    for (int trial = 1; trial <= trialCount; ++trial) {
        std::vector<double> inferMs;
        inferMs.reserve(measuredFrames);
        const auto wallStart = std::chrono::steady_clock::now();
        for (int frameIndex = 0; frameIndex < measuredFrames; ++frameIndex) {
            ASSERT_EQ(uploadFrame(frameIndex), CL_SUCCESS);
            auto& request = requests[frameIndex % requests.size()];
            const auto inferStart = std::chrono::steady_clock::now();
            request.infer();
            const auto inferStop = std::chrono::steady_clock::now();
            inferMs.push_back(std::chrono::duration<double, std::milli>(inferStop - inferStart).count());
        }
        const auto wallStop = std::chrono::steady_clock::now();
        const double inferSum = std::accumulate(inferMs.begin(), inferMs.end(), 0.0);
        const double wallMs = std::chrono::duration<double, std::milli>(wallStop - wallStart).count();
        std::sort(inferMs.begin(), inferMs.end());
        const auto percentile = [&inferMs](double fraction) {
            return inferMs[static_cast<size_t>(fraction * (inferMs.size() - 1))];
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "REMOTE_TENSOR_OPENCL_CEILING trial=" << trial
                  << " inference_only_fps=" << measuredFrames * 1000.0 / inferSum
                  << " input_transfer_inclusive_fps=" << measuredFrames * 1000.0 / wallMs
                  << " infer_p50_ms=" << percentile(0.50)
                  << " infer_p90_ms=" << percentile(0.90)
                  << " infer_p99_ms=" << percentile(0.99) << std::endl;
    }
}

// A finite file source reaches EOS deterministically and releases every resource.
TEST_F(WindowsD3D11RemoteTensorFixture, FileDecodeReachesEosAndReleasesResources) {
    ASSERT_TRUE(fileExists(videoPath)) << "test video not found: " << videoPath;

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

// Direct camera capture -> D3D11 OVMS C-API inference + H.264 MP4 output.
TEST_F(WindowsD3D11RemoteTensorFixture, CameraDecodeOvmsCapiInferenceAndMp4Encode) {
    if (envOr("OVMS_TEST_CAMERA", "").empty())
        GTEST_SKIP() << "set OVMS_TEST_CAMERA=1 to run the live camera test";
    const std::string cameraDevice = envOr("OVMS_TEST_CAMERA_DEVICE", "HP 5MP Camera");

    const std::string outPath = repoRoot() + "\\src\\test\\d3d11_camera_test_out.mp4";
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

// RTSP ingest onto the registered server-owned device.
TEST_F(WindowsD3D11RemoteTensorFixture, RtspDecodeUsesRegisteredD3D11Device) {
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

// Decode RTSP and publish encoded H.264 to an external RTSP server.
TEST_F(WindowsD3D11RemoteTensorFixture, RtspDecodeAndRepublish) {
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
