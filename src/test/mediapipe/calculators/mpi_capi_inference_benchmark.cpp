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

#include <fstream>
#include <memory>
#include <gtest/gtest.h>
#include <openvino/pass/serialize.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#endif
#include "src/ovms.h"
#include "mpi_inference_benchmark_utils.hpp"

extern "C" void* OVMS_MPGetExternalServerHandle() {
    ADD_FAILURE() << "Benchmark must supply the SERVER side packet";
    return nullptr;
}

namespace {
namespace bench = inference_benchmark;

bool check(OVMS_Status* status) {
    if (!status)
        return true;
    const char* details = nullptr;
    OVMS_Status* detailsStatus = OVMS_StatusDetails(status, &details);
    ADD_FAILURE() << (details ? details : "OVMS C-API operation failed");
    if (detailsStatus)
        OVMS_StatusDelete(detailsStatus);
    OVMS_StatusDelete(status);
    return false;
}

class MpiCapiBenchmark : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(std::filesystem::exists(bench::modelPath()));
#ifdef _WIN32
        D3D_FEATURE_LEVEL level;
        ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &device_, &level, nullptr)));
        imp_video_set_d3d11_device(device_.Get());
#else
        descriptor_ = open(bench::environment("OVMS_PERF_RENDER_NODE", "/dev/dri/renderD128").c_str(), O_RDWR);
        ASSERT_GE(descriptor_, 0);
        display_ = vaGetDisplayDRM(descriptor_);
        ASSERT_NE(display_, nullptr);
        int major = 0;
        int minor = 0;
        const VAStatus initialize = vaInitialize(display_, &major, &minor);
        initialized_ = initialize == VA_STATUS_SUCCESS;
        ASSERT_TRUE(initialized_) << vaErrorStr(initialize);
        imp_video_set_va_display(display_);
#endif
        const std::string testName = testing::UnitTest::GetInstance()->current_test_info()->name();
        directory_ = std::filesystem::path(bench::environment("TEST_TMPDIR", (bench::root() / "logs").string())) /
                     ("mpi_capi_" + testName);
        std::filesystem::create_directories(directory_ / "model/1");
        ov::Core core;
        ov::serialize(core.read_model(bench::modelPath()), (directory_ / "model/1/model.xml").string());
        rapidjson::Document document;
        document.SetObject();
        auto& allocator = document.GetAllocator();
        rapidjson::Value model(rapidjson::kObjectType);
        model.AddMember("name", rapidjson::Value(benchmark_config::kModelName, allocator), allocator);
        const std::string basePath = (directory_ / "model").generic_string();
        model.AddMember("base_path", rapidjson::Value(basePath.c_str(), allocator), allocator);
        model.AddMember("target_device", "GPU", allocator);
        model.AddMember("nireq", benchmark_config::kRequestCount, allocator);
        rapidjson::Value properties(rapidjson::kObjectType);
        properties.AddMember("PERFORMANCE_HINT", rapidjson::Value(benchmark_config::kPerformanceHint, allocator), allocator);
        properties.AddMember("NUM_STREAMS", rapidjson::Value(benchmark_config::kNumStreams, allocator), allocator);
        model.AddMember("plugin_config", properties, allocator);
        rapidjson::Value entry(rapidjson::kObjectType);
        entry.AddMember("config", model, allocator);
        rapidjson::Value models(rapidjson::kArrayType);
        models.PushBack(entry, allocator);
        document.AddMember("model_config_list", models, allocator);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        ASSERT_TRUE(document.Accept(writer));
        const std::string configPath = (directory_ / "config.json").string();
        {
            std::ofstream config(configPath);
            config << buffer.GetString();
            ASSERT_TRUE(config.good());
        }
        ASSERT_TRUE(check(OVMS_ServerNew(&server_)));
#ifdef _WIN32
        ASSERT_TRUE(check(OVMS_ServerSetGlobalD3D11Device(server_, device_.Get())));
#else
        ASSERT_TRUE(check(OVMS_ServerSetGlobalVADisplay(server_, display_)));
#endif
        OVMS_ServerSettings* rawSettings = nullptr;
        OVMS_ModelsSettings* rawModels = nullptr;
        ASSERT_TRUE(check(OVMS_ServerSettingsNew(&rawSettings)));
        std::unique_ptr<OVMS_ServerSettings, decltype(&OVMS_ServerSettingsDelete)> settings(rawSettings, OVMS_ServerSettingsDelete);
        ASSERT_TRUE(check(OVMS_ModelsSettingsNew(&rawModels)));
        std::unique_ptr<OVMS_ModelsSettings, decltype(&OVMS_ModelsSettingsDelete)> modelsSettings(rawModels, OVMS_ModelsSettingsDelete);
        ASSERT_TRUE(check(OVMS_ServerSettingsSetGrpcPort(settings.get(), 0)));
        ASSERT_TRUE(check(OVMS_ModelsSettingsSetConfigPath(modelsSettings.get(), configPath.c_str())));
        ASSERT_TRUE(check(OVMS_ServerStartFromConfigurationFile(server_, settings.get(), modelsSettings.get())));
    }

    void TearDown() override {
        if (server_)
            OVMS_ServerDelete(server_);
#ifdef _WIN32
        imp_video_set_d3d11_device(nullptr);
#else
        imp_video_set_va_display(nullptr);
        if (initialized_)
            vaTerminate(display_);
        if (descriptor_ >= 0)
            close(descriptor_);
#endif
        if (!directory_.empty())
            std::filesystem::remove_all(directory_);
    }

    void run(bool split) {
        bench::measureGraph(split ? "mpi_capi_two_calculators" : "mpi_capi_one_calculator", true, split,
            {{"server", mediapipe::MakePacket<OVMS_Server*>(server_)},
                {"servable_name", mediapipe::MakePacket<std::string>(benchmark_config::kModelName)},
                {"servable_version", mediapipe::MakePacket<int>(1)},
                {"input_name", mediapipe::MakePacket<std::string>(benchmark_config::kInputName)}});
    }

    OVMS_Server* server_ = nullptr;
    std::filesystem::path directory_;
#ifdef _WIN32
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
#else
    int descriptor_ = -1;
    VADisplay display_ = nullptr;
    bool initialized_ = false;
#endif
};

TEST_F(MpiCapiBenchmark, OneCalculator) { run(false); }
TEST_F(MpiCapiBenchmark, TwoCalculators) { run(true); }
}  // namespace