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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"

namespace {
constexpr int kDefaultWidth = 672;
constexpr int kDefaultHeight = 384;

bool parsePositiveInt(const char* text, int& value) {
    try {
        size_t parsedLength = 0;
        value = std::stoi(text, &parsedLength);
        return parsedLength == std::string(text).size() && value > 0;
    } catch (const std::exception&) {
        return false;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6 && argc != 7) {
        std::cerr << "Usage: mpi_openvino_video GRAPH.pbtxt VIDEO MODEL.xml [WIDTH HEIGHT [OUTPUT.mp4]]\n";
        return 2;
    }

    int width = kDefaultWidth;
    int height = kDefaultHeight;
    if (argc == 6 && (!parsePositiveInt(argv[4], width) || !parsePositiveInt(argv[5], height))) {
        std::cerr << "WIDTH and HEIGHT must be positive integers\n";
        return 2;
    }

    std::ifstream graphFile(argv[1]);
    if (!graphFile) {
        std::cerr << "Cannot open graph: " << argv[1] << '\n';
        return 1;
    }
    std::ostringstream graphText;
    graphText << graphFile.rdbuf();
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graphText.str());
    bool modelPathSet = false;
    for (int nodeIndex = 0; nodeIndex < config.node_size(); ++nodeIndex) {
        auto* node = config.mutable_node(nodeIndex);
        if (node->calculator() == "GstOpenVinoInferenceCalculator" ||
            node->calculator() == "GstVideoOpenVinoInferenceCalculator") {
            node->mutable_options()->MutableExtension(
                mediapipe::GstOpenVinoInferenceCalculatorOptions::ext)->set_model_path(argv[3]);
            modelPathSet = true;
        }
    }
    if (!modelPathSet) {
        std::cerr << "Graph must contain an OpenVino inference calculator\n";
        return 1;
    }

    mediapipe::CalculatorGraph graph;
    absl::Status status = graph.Initialize(config);
    if (!status.ok()) {
        std::cerr << "Graph initialization failed: " << status.ToString() << '\n';
        return 1;
    }

    uint64_t frameCount = 0;
    uint64_t detectionCount = 0;
    status = graph.ObserveOutputStream("detections", [&](const mediapipe::Packet& packet) {
        const int detections = packet.Get<int>();
        ++frameCount;
        detectionCount += static_cast<uint64_t>(detections);
        std::cout << "frame=" << frameCount << " detections=" << detections << '\n';
        return absl::OkStatus();
    });
    if (!status.ok()) {
        std::cerr << "Output observer setup failed: " << status.ToString() << '\n';
        return 1;
    }

    std::map<std::string, mediapipe::Packet> sidePackets = {
        {"video_path", mediapipe::MakePacket<std::string>(argv[2])},
        {"width", mediapipe::MakePacket<int>(width)},
        {"height", mediapipe::MakePacket<int>(height)},
    };
    const auto hasInputSidePacket = [&config](const std::string& declaration) {
        for (const std::string& configuredPacket : config.input_side_packet()) {
            if (configuredPacket == declaration)
                return true;
        }
        return false;
    };
    if (hasInputSidePacket("FULL_WIDTH:full_width"))
        sidePackets.emplace("full_width", mediapipe::MakePacket<int>(width));
    if (hasInputSidePacket("FULL_HEIGHT:full_height"))
        sidePackets.emplace("full_height", mediapipe::MakePacket<int>(height));
    if (hasInputSidePacket("FULL_VA_SURFACE:full_va_surface"))
        sidePackets.emplace("full_va_surface", mediapipe::MakePacket<bool>(true));
    if (hasInputSidePacket("OUTPUT_PATH:output_path"))
        sidePackets.emplace("output_path", mediapipe::MakePacket<std::string>(argc == 7 ? argv[6] : "mpi_openvino_overlay.mp4"));
    if (hasInputSidePacket("FRAMERATE:framerate"))
        sidePackets.emplace("framerate", mediapipe::MakePacket<int>(30));

    status = graph.StartRun(sidePackets);
    if (!status.ok()) {
        std::cerr << "Graph start failed: " << status.ToString() << '\n';
        return 1;
    }
    status = graph.WaitUntilDone();
    if (!status.ok()) {
        std::cerr << "Graph execution failed: " << status.ToString() << '\n';
        return 1;
    }

    std::cout << "frames=" << frameCount << " total_detections=" << detectionCount << '\n';
    return 0;
}