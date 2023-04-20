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
// TODO to be moved to mediapipe fork
#include <iostream>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
using std::cout;
using std::endl;

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
using std::cout;
using std::endl;

namespace mediapipe {
namespace tf = ::tensorflow;

static absl::Status ExecuteDummy() {
    // Configures a simple graph, which concatenates 2 PassThroughCalculators.
    CalculatorGraphConfig config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                input_stream: "in"
                output_stream: "out"
                node {
                  calculator: "OVMSCalculator"
                  input_stream: "TFTENSOR:in"
                  output_stream: "TFTENSOR:out"
                }
            )pb");

    CalculatorGraph graph;
    cout << __FILE__ << ":" << __LINE__ << endl;
    auto ret = graph.Initialize(config);
    LOG(ERROR) << ret;
    cout << __FILE__ << ":" << __LINE__ << endl;
    ASSIGN_OR_RETURN(OutputStreamPoller poller,
        graph.AddOutputStreamPoller("out"));
    cout << __FILE__ << ":" << __LINE__ << endl;
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    cout << __FILE__ << ":" << __LINE__ << endl;
    // Give 10 input packets that contains the same string "Hello World!".
    for (int i = 0; i < 10; ++i) {
        cout << __FILE__ << ":" << __LINE__ << endl;

        tf::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 10}));
        auto input_tensor_access = input_tensor.tensor<float, 2>();  // 2 since dummy is 2d
        for (int x = 0; x < 10; ++x) {
            input_tensor_access(0, x) = (float)(3 * x);
        }
        auto abstatus = graph.AddPacketToInputStream(
            // TODO covnert float to proper tensor?
            "in", MakePacket<tf::Tensor>(input_tensor).At(Timestamp(i)));
        if (!abstatus.ok()) {
            LOG(ERROR) << "XYZ:   " << abstatus;
        }
        MP_RETURN_IF_ERROR(abstatus);
    }
    cout << __FILE__ << ":" << __LINE__ << endl;
    // Close the input stream "in".
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
    cout << __FILE__ << ":" << __LINE__ << endl;
    mediapipe::Packet packet;
    // Get the output packets string.
    cout << __FILE__ << ":" << __LINE__ << endl;
    while (poller.Next(&packet)) {
        auto received = packet.Get<tf::Tensor>();
        auto received_tensor_access = received.tensor<float, 2>();  // 2 since dummy is 2d
        cout << "Received tensor: [";
        for (int x = 0; x < 10; ++x) {
            cout << received_tensor_access(0, x) << " ";
        }
        cout << " ]" << endl;
    }
    cout << __FILE__ << ":" << __LINE__ << endl;
    return graph.WaitUntilDone();
}

static absl::Status PrintHelloWorld() {
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
    auto ret = graph.Initialize(config);
    LOG(ERROR) << ret;
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
        LOG(ERROR) << packet.Get<float>();
    }
    return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cout << __FILE__ << ":" << __LINE__ << endl;
    auto grph = mediapipe::ExecuteDummy();
    cout << __FILE__ << ":" << __LINE__ << endl;
    CHECK(grph.ok());
    cout << __FILE__ << ":" << __LINE__ << endl;
    return 0;
}
