//*****************************************************************************
// Copyright 2026 Intel Corporation
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
//*****************************************************************************
#include <cstring>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "src/kfs_frontend/kfs_graph_executor_impl.hpp"
#include "src/logging.hpp"
#include "src/mediapipe_internal/mediapipegraphdefinition.hpp"
#include "src/mediapipe_internal/opencv_calculators_plugin_loader.hpp"
#include "src/status.hpp"
#include "src/utils/env_guard.hpp"

namespace ovms {
namespace {

class TestGraphDefinition : public MediapipeGraphDefinition {
public:
    TestGraphDefinition() :
        MediapipeGraphDefinition("opencv_plugin_test") {}

    Status parse(const std::string& text) {
        chosenConfig = text;
        return validateForConfigLoadableness();
    }
    using MediapipeGraphDefinition::dryInitializeTest;
    using MediapipeGraphDefinition::validateReferencedNodesRegistered;

    const mediapipe::CalculatorGraphConfig& graphConfig() const {
        return config;
    }
};

class NestedOpenCvTestGraph : public mediapipe::Subgraph {
public:
    absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(mediapipe::SubgraphContext*) override {
        mediapipe::CalculatorGraphConfig graph;
        graph.add_input_stream("image");
        graph.add_output_stream("tensors");
        auto* node = graph.add_node();
        node->set_calculator("ImageToTensorCalculator");
        node->add_input_stream("IMAGE:image");
        node->add_output_stream("TENSORS:tensors");
        return graph;
    }
};
REGISTER_MEDIAPIPE_GRAPH(NestedOpenCvTestGraph);

class MissingOpenCvPlugin : public ::testing::Test {
protected:
    EnvGuard environment;

    void SetUp() override {
        spdlog::set_level(spdlog::level::off);
        environment.set("OVMS_OPENCV_CALCULATORS_PLUGIN_PATH", "/ovms/nonexistent-opencv-calculators-plugin.so");
        ASSERT_FALSE(loadOpenCvCalculatorsPlugin());
    }
};

TEST_F(MissingOpenCvPlugin, NonOpenCvGraphStillInitializes) {
    TestGraphDefinition graph;
    ASSERT_TRUE(graph.parse(R"(
        input_stream: "in"
        output_stream: "out"
        node { calculator: "PassThroughCalculator" input_stream: "in" output_stream: "out" }
    )")
                    .ok());
    ASSERT_TRUE(graph.validateReferencedNodesRegistered().ok());
    EXPECT_TRUE(graph.dryInitializeTest().ok());
}

TEST_F(MissingOpenCvPlugin, CalculatorErrorNamesOpenCvAndOperation) {
    TestGraphDefinition graph;
    ASSERT_TRUE(graph.parse(R"(node { calculator: "ImageToTensorCalculator" })").ok());
    const auto status = graph.validateReferencedNodesRegistered();
    EXPECT_EQ(status.getCode(), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
    EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
    EXPECT_NE(status.string().find("ImageToTensorCalculator"), std::string::npos);
}

TEST_F(MissingOpenCvPlugin, ExtensionParseErrorNamesOpenCvAndOperation) {
    TestGraphDefinition graph;
    const auto status = graph.parse(R"(
        node {
            calculator: "ImageToTensorCalculator"
            options { [mediapipe.ImageToTensorCalculatorOptions.ext] { output_tensor_width: 1 } }
        }
    )");
    EXPECT_EQ(status.getCode(), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
    EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
    EXPECT_NE(status.string().find("ImageToTensorCalculator"), std::string::npos);
}

TEST_F(MissingOpenCvPlugin, NestedSubgraphErrorNamesOpenCvAndOperation) {
    TestGraphDefinition graph;
    ASSERT_TRUE(graph.parse(R"(
        input_stream: "in"
        output_stream: "out"
        node { calculator: "NestedOpenCvTestGraph" input_stream: "in" output_stream: "out" }
    )")
                    .ok());
    ASSERT_TRUE(graph.validateReferencedNodesRegistered().ok());
    const auto status = graph.dryInitializeTest();
    EXPECT_EQ(status.getCode(), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
    EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
    EXPECT_NE(status.string().find("ImageToTensorCalculator"), std::string::npos);
}

TEST(OpenCvPluginDeployment, RegistersOptionsAndExecutesDecoderGraph) {
    // Registration is process-lifetime state. Keep the parent process pristine
    // so the missing-plugin tests remain independent of test execution order.
    EXPECT_EXIT(
        {
            const int result = []() {
                UnSetEnvironmentVar("OVMS_OPENCV_CALCULATORS_PLUGIN_PATH");
                if (!loadOpenCvCalculatorsPlugin()) {
                    std::cerr << getOpenCvCalculatorsPluginDiagnostic();
                    return 1;
                }
                if (!mediapipe::CalculatorBaseRegistry::GetRegisteredNames().contains(
                        "OpenCvEncodedImageToImageFrameCalculator")) {
                    std::cerr << "Calculator registration did not reach the host registry";
                    return 2;
                }
                TestGraphDefinition definition;
                const auto parsed = definition.parse(R"(
                    input_stream: "encoded"
                    output_stream: "decoded"
                    node {
                        calculator: "OpenCvEncodedImageToImageFrameCalculator"
                        input_stream: "encoded"
                        output_stream: "decoded"
                        options {
                            [mediapipe.OpenCvEncodedImageToImageFrameCalculatorOptions.ext] {
                                apply_orientation_from_exif_data: false
                            }
                        }
                    }
                )");
                if (!parsed.ok()) {
                    std::cerr << parsed.string();
                    return 3;
                }
                mediapipe::CalculatorGraph graph;
                auto status = graph.Initialize(definition.graphConfig());
                if (!status.ok()) {
                    std::cerr << status;
                    return 4;
                }
                auto poller = graph.AddOutputStreamPoller("decoded");
                if (!poller.ok()) {
                    std::cerr << poller.status();
                    return 5;
                }
                status = graph.StartRun({});
                if (!status.ok()) {
                    std::cerr << status;
                    return 6;
                }
                // A one-pixel PPM avoids another encoder dependency in the host.
                const std::string encoded = std::string("P6\n1 1\n255\n") + "\x0a\x14\x1e";
                status = graph.AddPacketToInputStream("encoded",
                    mediapipe::MakePacket<std::string>(encoded).At(mediapipe::Timestamp(0)));
                if (!status.ok()) {
                    std::cerr << status;
                    return 7;
                }
                status = graph.CloseAllPacketSources();
                if (!status.ok()) {
                    std::cerr << status;
                    return 8;
                }
                mediapipe::Packet packet;
                if (!poller->Next(&packet)) {
                    std::cerr << "Decoder produced no output";
                    return 9;
                }
                const auto& image = packet.Get<mediapipe::ImageFrame>();
                if (image.Width() != 1 || image.Height() != 1 ||
                    image.NumberOfChannels() != 3 ||
                    std::memcmp(image.PixelData(), "\x0a\x14\x1e", 3) != 0) {
                    std::cerr << "Unexpected decoded RGB image";
                    return 10;
                }
                status = graph.WaitUntilDone();
                if (!status.ok()) {
                    std::cerr << status;
                    return 11;
                }
                return 0;
            }();
            _exit(result);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(OpenCvPluginDeployment, CtypesLocalSharedHostExecutesDecoderGraph) {
    const std::string host = GetEnvVar("TEST_SRCDIR") + "/" +
                             GetEnvVar("TEST_WORKSPACE") + "/src/opencv_calculators_local_host.so";
    EXPECT_EXIT(
        {
            UnSetEnvironmentVar("OVMS_OPENCV_CALCULATORS_PLUGIN_PATH");
            const char* script =
                "import ctypes,sys; host=ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL); "
                "sys.exit(host.OVMS_TestLocalOpenCvHost())";
            execl("/usr/bin/python3", "python3", "-c", script, host.c_str(), nullptr);
            _exit(127);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(OpenCvPluginDeployment, CtypesLocalSharedHostReportsMissingOpenCvLibraries) {
    const std::string root = GetEnvVar("TEST_SRCDIR") + "/" + GetEnvVar("TEST_WORKSPACE");
    const std::string host = root + "/src/opencv_calculators_local_host.so";
    const std::string audit = root + "/src/image_utils/libno_opencv_audit.so";
    EXPECT_EXIT(
        {
            UnSetEnvironmentVar("OVMS_OPENCV_CALCULATORS_PLUGIN_PATH");
            SetEnvironmentVar("LD_AUDIT", audit);
            const char* script =
                "import ctypes,sys; host=ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL); "
                "sys.exit(0 if host.OVMS_TestLocalOpenCvHost() == 1 else 1)";
            execl("/usr/bin/python3", "python3", "-c", script, host.c_str(), nullptr);
            _exit(127);
        },
        ::testing::ExitedWithCode(0), "Missing OpenCV/plugin:.*ovms-test-opencv-unavailable");
}

TEST(ImageFrameSerialization, CopiesPaddedRowsWithoutOpenCv) {
    for (const auto format : {mediapipe::ImageFormat::SRGB, mediapipe::ImageFormat::VEC32F1,
             mediapipe::ImageFormat::GRAY16}) {
        auto* frame = new mediapipe::ImageFrame(format, 3, 2, 16);
        const size_t rowBytes = static_cast<size_t>(frame->Width()) * frame->NumberOfChannels() * frame->ChannelSize();
        ASSERT_GT(static_cast<size_t>(frame->WidthStep()), rowBytes);
        std::memset(frame->MutablePixelData(), 0xff, frame->PixelDataSize());
        std::string expected(rowBytes * frame->Height(), '\0');
        for (int row = 0; row < frame->Height(); ++row) {
            for (size_t byte = 0; byte < rowBytes; ++byte) {
                const unsigned char value = static_cast<unsigned char>(row * rowBytes + byte);
                frame->MutablePixelData()[row * frame->WidthStep() + byte] = value;
                expected[row * rowBytes + byte] = value;
            }
        }
        const auto packet = mediapipe::Adopt(frame);
        KFSResponse response;
        ASSERT_TRUE(onPacketReadySerializeImpl("id", "model", "1", "output",
            mediapipe_packet_type_enum::MEDIAPIPE_IMAGE, packet, response)
                        .ok());
        ASSERT_EQ(response.raw_output_contents_size(), 1);
        EXPECT_EQ(response.raw_output_contents(0), expected);
        EXPECT_EQ(response.outputs(0).shape(0), 2);
        EXPECT_EQ(response.outputs(0).shape(1), 3);
        EXPECT_EQ(response.outputs(0).shape(2), frame->NumberOfChannels());
    }
}

}  // namespace
}  // namespace ovms
