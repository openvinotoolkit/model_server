// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A simple example to print out "Hello World!" from a MediaPipe graph.

#include <iostream>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
using std::cout;
using std::endl;

namespace mediapipe {
absl::Status PrintHelloWorld() {
    // Configures a simple graph, which concatenates 2 PassThroughCalculators.
    CalculatorGraphConfig config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
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

    CalculatorGraph graph;
    cout << __FILE__ << ":" << __LINE__ << endl;
    //MP_RETURN_IF_ERROR(graph.Initialize(config));
    auto ret = graph.Initialize(config);
    LOG(INFO) << ret;
    cout << __FILE__ << ":" << __LINE__ << endl;
    ASSIGN_OR_RETURN(OutputStreamPoller poller,
        graph.AddOutputStreamPoller("out"));
    cout << __FILE__ << ":" << __LINE__ << endl;
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    cout << __FILE__ << ":" << __LINE__ << endl;
    // Give 10 input packets that contains the same string "Hello World!".
    for (int i = 0; i < 10; ++i) {
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            // TODO covnert float to proper tensor?
            "in", MakePacket<float>(0.0f).At(Timestamp(i))));
    }
    // Close the input stream "in".
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
    mediapipe::Packet packet;
    // Get the output packets string.
    while (poller.Next(&packet)) {
        LOG(INFO) << packet.Get<float>();
    }
    return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cout << __FILE__ << ":" << __LINE__ << endl;
    auto grph = mediapipe::PrintHelloWorld();
    cout << __FILE__ << ":" << __LINE__ << endl;
    //CHECK(mediapipe::PrintHelloWorld().ok());
    CHECK(grph.ok());
    cout << __FILE__ << ":" << __LINE__ << endl;
    return 0;
}