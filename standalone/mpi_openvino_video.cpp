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

#include <cstdint>
#include <fstream>
#include <iostream>
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
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: mpi_openvino_video GRAPH.pbtxt VIDEO MODEL.xml [WIDTH HEIGHT]\n";
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
    if (config.node_size() != 1) {
        std::cerr << "Expected exactly one inference node in the graph\n";
        return 1;
    }

    auto* options = config.mutable_node(0)->mutable_options()->MutableExtension(
        mediapipe::GstOpenVinoInferenceCalculatorOptions::ext);
    options->set_model_path(argv[3]);

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

    status = graph.StartRun({
        {"video_path", mediapipe::MakePacket<std::string>(argv[2])},
        {"width", mediapipe::MakePacket<int>(width)},
        {"height", mediapipe::MakePacket<int>(height)},
    });
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