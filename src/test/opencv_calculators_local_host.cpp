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

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "src/mediapipe_internal/opencv_calculators_plugin_loader.hpp"

extern "C" __attribute__((visibility("default"))) int OVMS_TestLocalOpenCvHost() {
    if (!ovms::loadOpenCvCalculatorsPlugin()) {
        std::cerr << ovms::getOpenCvCalculatorsPluginDiagnostic();
        return 1;
    }
    mediapipe::CalculatorGraphConfig config;
    config.add_input_stream("encoded");
    config.add_output_stream("decoded");
    auto* node = config.add_node();
    node->set_calculator("OpenCvEncodedImageToImageFrameCalculator");
    node->add_input_stream("encoded");
    node->add_output_stream("decoded");
    mediapipe::CalculatorGraph graph;
    auto status = graph.Initialize(config);
    if (!status.ok()) {
        std::cerr << status;
        return 2;
    }
    auto poller = graph.AddOutputStreamPoller("decoded");
    if (!poller.ok()) {
        std::cerr << poller.status();
        return 3;
    }
    status = graph.StartRun({});
    if (!status.ok()) {
        std::cerr << status;
        return 4;
    }
    const std::string encoded = std::string("P6\n1 1\n255\n") + "\x0a\x14\x1e";
    status = graph.AddPacketToInputStream("encoded",
        mediapipe::MakePacket<std::string>(encoded).At(mediapipe::Timestamp(0)));
    if (!status.ok()) {
        std::cerr << status;
        return 5;
    }
    status = graph.CloseAllPacketSources();
    if (!status.ok()) {
        std::cerr << status;
        return 6;
    }
    mediapipe::Packet packet;
    if (!poller->Next(&packet)) {
        return 7;
    }
    const auto& image = packet.Get<mediapipe::ImageFrame>();
    if (image.Width() != 1 || image.Height() != 1 ||
        image.NumberOfChannels() != 3 ||
        std::memcmp(image.PixelData(), "\x0a\x14\x1e", 3) != 0) {
        return 8;
    }
    status = graph.WaitUntilDone();
    if (!status.ok()) {
        std::cerr << status;
        return 9;
    }
    return 0;
}
