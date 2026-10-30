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

#include <gtest/gtest.h>

#ifdef _WIN32
TEST(MpiOpenVinoIntegration, ModelManagerCompiledModelSidePacket) {
    GTEST_SKIP() << "ModelManager VA integration test is Linux-only";
}
#else
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "src/modelinstance.hpp"
#include "src/modelinstanceunloadguard.hpp"
#include "src/servable_management/modelmanager.hpp"
#include "mpi_inference_benchmark_utils.hpp"

namespace {
namespace bench = inference_benchmark;

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
}  // namespace

TEST(MpiOpenVinoIntegration, ModelManagerCompiledModelSidePacket) {
    const std::filesystem::path modelPath = bench::root() / "src/test/face_detection_adas";
    if (!std::filesystem::exists(modelPath / "1/face-detection-adas-0001.xml"))
        GTEST_SKIP() << "OVMS face detection test model is not prepared: " << modelPath;

    const std::filesystem::path configPath = std::filesystem::temp_directory_path() /
        ("ovms_mpi_va_config_" + std::to_string(::getpid()) + ".json");
    struct ConfigFileCleanup {
        std::filesystem::path path;
        ~ConfigFileCleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } configFileCleanup{configPath};

    std::ofstream configFile(configPath);
    ASSERT_TRUE(configFile.is_open());
    configFile << R"({"va_display":{"target_device":"GPU.0","render_node":")"
               << bench::environment("OVMS_PERF_RENDER_NODE", "/dev/dri/renderD128")
               << R"("},"model_config_list":[{"config":{"name":"mpi_va","base_path":")"
               << modelPath.string()
               << R"(","target_device":"GPU.0"}}]})";
    configFile.close();
    ASSERT_TRUE(configFile.good());

    NativeVaDisplay vaDisplay;
    ASSERT_TRUE(vaDisplay.open()) << "VA display initialization failed";
    ovms::setGlobalVaDisplay(vaDisplay.display, "GPU.0");
    struct GlobalVaDisplayCleanup {
        ~GlobalVaDisplayCleanup() { ovms::setGlobalVaDisplay(nullptr); }
    } globalVaDisplayCleanup;

    ovms::ModelManager manager;
    ASSERT_EQ(manager.startFromFile(configPath.string()), ovms::StatusCode::OK);

    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    ASSERT_EQ(manager.getModelInstance("mpi_va", 1, modelInstance, unloadGuard), ovms::StatusCode::OK);
    ASSERT_NE(modelInstance->getCompiledModel(), nullptr);
    ASSERT_NE(ovms::globalVaDisplay, nullptr);

    std::shared_ptr<ovms::ModelInstanceUnloadGuard> guard(std::move(unloadGuard));
    auto compiledModel = std::shared_ptr<ov::CompiledModel>(guard, modelInstance->getCompiledModel());
    {
        mediapipe::CalculatorGraphConfig config = bench::graphConfig(false, false);
        config.add_input_side_packet("COMPILED_MODEL:compiled_model");
        auto* node = config.mutable_node(0);
        node->add_input_side_packet("COMPILED_MODEL:compiled_model");
        node->mutable_options()->MutableExtension(mediapipe::GstOpenVinoInferenceCalculatorOptions::ext)->clear_model_path();

        mediapipe::CalculatorGraph graph;
        ASSERT_TRUE(graph.Initialize(config).ok());
        int frames = 0;
        int detections = 0;
        ASSERT_TRUE(graph.ObserveOutputStream("detections", [&](const mediapipe::Packet& packet) {
                             detections += packet.Get<int>();
                             ++frames;
                             return absl::OkStatus();
                         }).ok());
        const absl::Status status = graph.StartRun({
            {"video_path", mediapipe::MakePacket<std::string>(bench::videoPath())},
            {"width", mediapipe::MakePacket<int>(benchmark_config::kModelWidth)},
            {"height", mediapipe::MakePacket<int>(benchmark_config::kModelHeight)},
            {"compiled_model", mediapipe::MakePacket<std::shared_ptr<ov::CompiledModel>>(compiledModel)},
        });
        ASSERT_TRUE(status.ok()) << status;
        ASSERT_TRUE(graph.WaitUntilDone().ok());
        ASSERT_EQ(frames, bench::totalFrames());
        ASSERT_GT(detections, 0);
    }
    compiledModel.reset();
    guard.reset();
    modelInstance.reset();
    manager.join();
}
#endif