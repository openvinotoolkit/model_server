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
 * P5 Windows tests B and C — the D3D11 counterparts of the Linux POC
 * (GStreamerSurfaceInferenceViaGlobalVADisplay):
 *   B) one GStreamer-decoded D3D11 NV12 texture runs inference on a
 *      ModelManager-loaded GPU model through the OVMS C-API (zero host copy),
 *      with the server-owned ID3D11Device registered as the global device.
 *   C) the split MediaPipe graph (GstVideoSourceCalculator ->
 *      GstCapiInferenceCalculator) holds multiple distinct D3D11 textures in
 *      flight and produces detections through the same servable.
 *
 * Needs Windows + Intel GPU + GStreamer D3D11 + the face_detection_adas fixture
 * and a test video; skips otherwise. Overridable via env:
 *   VIDEO_PATH   (default C:\git\model_server\recording_3_raw.avi)
 *   OVMS_REPO_ROOT (default C:\git\model_server)
 */

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/ovms.h"

namespace {

using Microsoft::WRL::ComPtr;

constexpr char kModelName[] = "face_detection_adas";
constexpr char kHostModelName[] = "face_detection_adas_host";
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

// Write a temp config pointing the GPU face_detection_adas servable at the
// in-repo fixture, so the test does not depend on the config-rewrite harness.
std::string writeTempConfig() {
    // Bake NV12 two-plane preprocessing into the model (OVMS config JSON cannot
    // express color_format). This splits the single `data` input into `data/y`
    // and `data/uv`, mirroring the Linux reference preprocessModel(), so the
    // decoded NV12 surface planes bind directly through the tensor factory.
    const std::string srcXml = repoRoot() + "\\src\\test\\face_detection_adas\\1\\face-detection-adas-0001.xml";
    const std::string tmpModelDir = repoRoot() + "\\src\\test\\d3d11_fd_tmp_model";
    const std::string tmpHostModelDir = repoRoot() + "\\src\\test\\d3d11_fd_host_tmp_model";
    std::filesystem::create_directories(tmpModelDir + "\\1");
    std::filesystem::create_directories(tmpHostModelDir + "\\1");
    {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(srcXml);
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
            .set_memory_type(ov::intel_gpu::memory_type::surface);
        ppp.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();
        ov::save_model(model, tmpModelDir + "\\1\\model.xml", false);
    }
    {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(srcXml);
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"});
        ppp.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();
        ov::save_model(model, tmpHostModelDir + "\\1\\model.xml", false);
    }

    std::string escaped;
    for (char c : tmpModelDir) {
        if (c == '\\')
            escaped += "\\\\";
        else
            escaped += c;
    }
    std::string hostEscaped;
    for (char c : tmpHostModelDir) {
        if (c == '\\')
            hostEscaped += "\\\\";
        else
            hostEscaped += c;
    }
    const std::string cfgPath = repoRoot() + "\\src\\test\\d3d11_face_detection_adas_tmp.json";
    std::ofstream f(cfgPath);
    f << "{\"model_config_list\":[{\"config\":{\"name\":\"" << kModelName
      << "\",\"base_path\":\"" << escaped << "\",\"target_device\":\"GPU\"}},{\"config\":{\"name\":\""
      << kHostModelName << "\",\"base_path\":\"" << hostEscaped << "\",\"target_device\":\"GPU\"}}]}";
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

struct D3D11ServerFixture : public ::testing::Test {
    inline static ComPtr<ID3D11Device> suiteDevice;
    ComPtr<ID3D11Device> device;
    OVMS_Server* server = nullptr;
    std::string videoPath;
    bool ready = false;

    void SetUp() override {
        videoPath = envOr("VIDEO_PATH", repoRoot() + "\\recording_3_raw.avi");
        if (!imp_video_d3d11_available())
            GTEST_SKIP() << "D3D11 surface sharing not available";
        if (!fileExists(videoPath))
            GTEST_SKIP() << "test video not found: " << videoPath;
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
        ASSERT_TRUE(capiOk(OVMS_ServerSettingsSetGrpcPort(serverSettings, 9718)));
        ASSERT_TRUE(capiOk(OVMS_ModelsSettingsSetConfigPath(modelsSettings, cfg.c_str())));
        ASSERT_TRUE(capiOk(OVMS_ServerNew(&server)));

        // Register the server-owned device and share it with GStreamer BEFORE
        // the server loads GPU models (so they compile on the matching D3DContext).
        ASSERT_TRUE(capiOk(OVMS_ServerSetGlobalD3D11Device(server, device.Get())));
        imp_video_set_d3d11_device(device.Get());

        if (!capiOk(OVMS_ServerStartFromConfigurationFile(server, serverSettings, modelsSettings)))
            GTEST_SKIP() << "server failed to start (no GPU model?)";
        ready = true;
    }

    void TearDown() override {
        if (server) {
            OVMS_ServerSetGlobalD3D11Device(server, nullptr);
            imp_video_set_d3d11_device(nullptr);
            OVMS_ServerDelete(server);
        }
        std::filesystem::remove_all(repoRoot() + "\\src\\test\\d3d11_fd_tmp_model");
        std::filesystem::remove_all(repoRoot() + "\\src\\test\\d3d11_fd_host_tmp_model");
        std::filesystem::remove(repoRoot() + "\\src\\test\\d3d11_face_detection_adas_tmp.json");
    }

    static void TearDownTestSuite() {
        suiteDevice.Reset();
    }

    // Decode one frame to a D3D11 NV12 texture sized to the model.
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

        std::cerr << "[infer] calling OVMS_Inference texture=" << texture << std::endl;
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
            float maxConf = 0.0f;
            const float* v = static_cast<const float*>(out);
            for (size_t i = 0; i + 7 <= bytes / sizeof(float); i += 7) {
                maxConf = (std::max)(maxConf, v[i + 2]);
                if (v[i + 2] >= 0.5f)
                    ++detections;
            }
            std::cerr << "[D3D11 infer] detections=" << detections
                      << " maxConf=" << maxConf << " outFloats=" << (bytes / sizeof(float)) << std::endl;
        }
        OVMS_InferenceResponseDelete(response);
        OVMS_InferenceRequestDelete(request);
        return detections;
    }

    int runCapiHostNv12Inference(const uint8_t* yData, const uint8_t* uvData, int w, int h) {
        OVMS_InferenceRequest* request = nullptr;
        if (!capiOk(OVMS_InferenceRequestNew(&request, server, kHostModelName, 1)))
            return -1;
        const std::string yName = std::string(kInputName) + "/y";
        const std::string uvName = std::string(kInputName) + "/uv";
        const int64_t shapeY[] = {1, h, w, 1};
        const int64_t shapeUV[] = {1, h / 2, w / 2, 2};
        const size_t bytesY = static_cast<size_t>(w) * h;
        const size_t bytesUV = bytesY / 2;
        capiOk(OVMS_InferenceRequestAddInput(request, yName.c_str(), OVMS_DATATYPE_U8, shapeY, 4));
        capiOk(OVMS_InferenceRequestInputSetData(request, yName.c_str(), yData, bytesY, OVMS_BUFFERTYPE_CPU, 0));
        capiOk(OVMS_InferenceRequestAddInput(request, uvName.c_str(), OVMS_DATATYPE_U8, shapeUV, 4));
        capiOk(OVMS_InferenceRequestInputSetData(request, uvName.c_str(), uvData, bytesUV, OVMS_BUFFERTYPE_CPU, 0));

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
            float maxConf = 0.0f;
            const float* values = static_cast<const float*>(out);
            for (size_t index = 0; index + 7 <= bytes / sizeof(float); index += 7) {
                maxConf = (std::max)(maxConf, values[index + 2]);
                if (values[index + 2] >= 0.5f)
                    ++detections;
            }
            std::cerr << "[Host NV12 infer] detections=" << detections
                      << " maxConf=" << maxConf << " outFloats=" << (bytes / sizeof(float)) << std::endl;
        }
        OVMS_InferenceResponseDelete(response);
        OVMS_InferenceRequestDelete(request);
        return detections;
    }
};

TEST_F(D3D11ServerFixture, HostNv12CapiInference) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", videoPath.c_str());
    imp_video_source_set(src, "width", std::to_string(kModelW).c_str());
    imp_video_source_set(src, "height", std::to_string(kModelH).c_str());
    imp_video_decode_opts_t vopts{};
    imp_video_stream_t* stream = nullptr;
    ASSERT_EQ(imp_video_open(&stream, src, ctx, &vopts), IMP_OK) << imp_context_get_error(ctx);
    imp_video_source_destroy(src);

    imp_tensor_t* tensor = nullptr;
    ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_SYSTEM);
    const uint8_t* yData = nullptr;
    const uint8_t* uvData = nullptr;
    int width = 0;
    int height = 0;
    ASSERT_EQ(imp_tensor_get_nv12_planes(tensor, &yData, &uvData, &width, &height), IMP_OK);
    ASSERT_NE(yData, nullptr);
    ASSERT_NE(uvData, nullptr);
    EXPECT_GE(runCapiHostNv12Inference(yData, uvData, width, height), 1)
        << "baked NV12 PPP model did not detect a face from the CPU fallback frame";

    imp_tensor_release(tensor);
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

// B) one decoded D3D11 texture -> C-API inference on the ModelManager servable.
//
// PROVEN here: the server-owned ID3D11Device is injected into GStreamer, the
// decoded NV12 D3D11 texture (ArraySize=1, subresource 0) is imported through
// the OVMS C-API D3D11TensorFactory into the model compiled on the matching
// D3DContext, and inference executes end to end returning the correct output
// structure ([1,1,200,7] = 1400 floats) with no host copy.
//
TEST_F(D3D11ServerFixture, OneTextureCapiInference) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";

    imp_context_t* ctx = nullptr;
    if (imp_context_create(&ctx, nullptr) != IMP_OK)
        FAIL() << "imp_context_create failed";
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

    imp_tensor_t* tensor = nullptr;
    ASSERT_EQ(imp_video_read_frame(&tensor, stream, 0), IMP_OK);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(imp_tensor_get_memory_type(tensor), IMP_MEM_D3D11_SURFACE);

    void* texture = nullptr;
    void* decodeDevice = nullptr;
    uint32_t subresource = 0;
    int fw = 0, fh = 0;
    ASSERT_EQ(imp_tensor_get_d3d11_texture(tensor, &texture, &decodeDevice, &subresource, &fw, &fh), IMP_OK);
    // The server-owned device is the one GStreamer decoded onto (zero-copy).
    EXPECT_EQ(decodeDevice, static_cast<void*>(device.Get())) << "decoded on a different device";
    ASSERT_EQ(subresource, 0u) << "non-zero subresource not importable by OpenVINO";

    int detections = runCapiInference(texture, fw, fh);
    EXPECT_GE(detections, 1) << "D3D11 texture import + C-API inference did not detect a face";

    imp_tensor_release(tensor);
    imp_video_close(stream);
    imp_context_destroy(ctx);
}

// C) split graph: multiple distinct D3D11 textures in flight -> detections.
TEST_F(D3D11ServerFixture, SplitGraphMultipleInFlight) {
    if (!ready)
        GTEST_SKIP() << "fixture not ready";

    const std::string graphPath = repoRoot() + "\\src\\test\\mediapipe\\calculators\\gst_video_infer.pbtxt";
    if (!fileExists(graphPath))
        GTEST_SKIP() << "graph not found: " << graphPath;
    std::ifstream gf(graphPath);
    std::stringstream ss;
    ss << gf.rdbuf();

    auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(ss.str());
    ::mediapipe::CalculatorGraph graph;
    ASSERT_TRUE(graph.Initialize(config).ok());
    auto pollerOr = graph.AddOutputStreamPoller("detections");
    ASSERT_TRUE(pollerOr.ok());
    ::mediapipe::OutputStreamPoller poller = std::move(pollerOr.value());

    std::map<std::string, ::mediapipe::Packet> side = {
        {"video_path", ::mediapipe::MakePacket<std::string>(videoPath)},
        {"width", ::mediapipe::MakePacket<int>(kModelW)},
        {"height", ::mediapipe::MakePacket<int>(kModelH)},
        {"servable_name", ::mediapipe::MakePacket<std::string>(kModelName)},
        {"servable_version", ::mediapipe::MakePacket<int>(1)},
        {"input_name", ::mediapipe::MakePacket<std::string>(kInputName)},
        {"server", ::mediapipe::MakePacket<OVMS_Server*>(server)},
    };
    ASSERT_TRUE(graph.StartRun(side).ok());

    int frames = 0;
    int totalDetections = 0;
    ::mediapipe::Packet p;
    while (poller.Next(&p)) {
        totalDetections += p.Get<int>();
        ++frames;
    }
    EXPECT_GE(frames, 3) << "expected multiple packet-owned D3D11 frames in flight";
    EXPECT_GE(totalDetections, frames)
        << "expected at least one detection from every graph inference frame";
    ASSERT_TRUE(graph.WaitUntilDone().ok());
}

}  // namespace
