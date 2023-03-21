//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include "calculator_graph.hpp"

#include "../logging.hpp"
#include "../status.hpp"
#include "mediapipe/framework/calculator_graph.h"

namespace ovms {

OVMSCalculatorGraph::OVMSCalculatorGraph() {
}

absl::Status OVMSCalculatorGraph::execute() {
    // google::InitGoogleLogging("--logtostderr=1");
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
                input_stream: "in"
                output_stream: "out"
                node {
                  calculator: "OVMSCalculator"
                  input_stream: "in"
                  output_stream: "out1"
                }
                node {
                  calculator: "OVMSCalculator"
                  input_stream: "out1"
                  output_stream: "out"
                }
            )pb");

    mediapipe::CalculatorGraph graph;
    auto ret = graph.Initialize(config);
    LOG(INFO) << ret;
    SPDLOG_LOGGER_INFO(mediapipe_logger, "Graph initialization");
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
        graph.AddOutputStreamPoller("out"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    SPDLOG_LOGGER_INFO(mediapipe_logger, "Graph start");

    for (int i = 0; i < 10; ++i) {
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            // TODO covnert float to proper tensor?
            "in", mediapipe::MakePacket<float>(0.0f).At(mediapipe::Timestamp(i))));
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));

    mediapipe::Packet packet;
    // Get the output packets string.
    while (poller.Next(&packet)) {
        SPDLOG_LOGGER_INFO(mediapipe_logger, "Result {}", packet.Get<float>());
    }
    return graph.WaitUntilDone();
}

}  // namespace ovms
