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
#include "mediapipedemo.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <openvino/openvino.hpp>

#include "../ovms.h"  // NOLINT
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
// added only to deinitialize server here
using std::cout;
using std::endl;

using mediapipe::CalculatorGraph;
using mediapipe::CalculatorGraphConfig;
using mediapipe::MakePacket;
using mediapipe::OutputStreamPoller;
using mediapipe::Packet;
using mediapipe::ParseTextProtoOrDie;
using mediapipe::Timestamp;

// TODO remove before release after integration
static absl::Status ExecuteDummy(size_t requestCount, size_t inputStreamDelayMs) {
    // You have to have 2 different prefixes fo two different input/output streams even if they don't mean anything
    // if model input name does not follow mediapipe convention [A-Z_][A-Z0-9_]* we have to either (1) use model mapping with models or change DAG config. (2) Another option is to use protobuf side packet/option that would map mediapipe input stream TAG with actual model input. (2) seems better as it wouldn't require ingerence in OVMS config just to follow mediapipe convention
    CalculatorGraphConfig config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                input_stream: "in"
                output_stream: "out"
                node {
                  calculator: "OVMSOVCalculator"
                  input_stream: "B:in"
                  output_stream: "A:out"
                  node_options: {
                        [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                          servable_name: "dummy"
                          servable_version: "1"
                          tag_to_input_tensor_names {
                            key: "B"
                            value: "b"
                          }
                          tag_to_output_tensor_names {
                            key: "A"
                            value: "a"
                          }
                          config_path: "/ovms/src/test/mediapipe/config_standard_dummy.json"
                        }
                  }
                }
            )pb");

    CalculatorGraph graph;
    auto ret = graph.Initialize(config);
    LOG(ERROR) << ret;
    ASSIGN_OR_RETURN(OutputStreamPoller poller, graph.AddOutputStreamPoller("out"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    for (size_t i = 0; i < requestCount; ++i) {
        ov::Shape ovShape{1, 10};
        ov::Tensor input_tensor(ov::element::Type_t::f32,
            ovShape);
        float* inputData = (float*)input_tensor.data();
        std::memcpy(input_tensor.data(), reinterpret_cast<void*>(inputData), input_tensor.get_byte_size());
        for (int x = 0; x < 10; ++x) {
            inputData[x] = (float)(i * x);
        }
        auto abstatus = graph.AddPacketToInputStream(
            "in", MakePacket<ov::Tensor>(input_tensor).At(Timestamp(i)));
        if (!abstatus.ok()) {
            LOG(ERROR) << "Failed to add packet to stream:" << abstatus;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(inputStreamDelayMs));
        MP_RETURN_IF_ERROR(abstatus);
    }
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
    Packet packet;
    size_t receivedCount = 0;
    while (poller.Next(&packet)) {
        auto received = packet.Get<ov::Tensor>();
        float* dataOut = (float*)received.data();
        auto timestamp = packet.Timestamp();
        std::stringstream ss;
        ss << "HelloOVMS Received tensor: [";
        for (int x = 0; x < 10; ++x) {
            ss << dataOut[x] << " ";
        }
        ss << " ] receivedCount: " << ++receivedCount << " timestamp: " << timestamp.DebugString() << endl;
        cout << ss.str() << endl;
    }
    auto res = graph.WaitUntilDone();
    // TODO temporary server stopping
    OVMS_Server* cserver = nullptr;
    OVMS_ServerNew(&cserver);
    OVMS_ServerDelete(cserver);
    cout << __FILE__ << ":" << __LINE__ << endl;
    return res;
}

static absl::Status ExecuteAdd(size_t requestCount, size_t inputStreamDelayMs) {
    CalculatorGraphConfig config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                input_stream: "in1"
                input_stream: "in2"
                output_stream: "out"
                node {
                  calculator: "OVMSOVCalculator"
                  input_stream: "INPUT1:in1"
                  input_stream: "INPUT2:in2"
                  output_stream: "SUM:out"
                  node_options: {
                        [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                          servable_name: "add"
                          servable_version: "1"
                          tag_to_input_tensor_names {
                            key: "INPUT1"
                            value: "input1"
                          }
                          tag_to_input_tensor_names {
                            key: "INPUT2"
                            value: "input2"
                          }
                          tag_to_output_tensor_names {
                            key: "SUM"
                            value: "sum"
                          }
                          config_path: "/ovms/src/test/mediapipe/config_standard_add.json"
                        }
                  }
                }
            )pb");

    CalculatorGraph graph;
    auto ret = graph.Initialize(config);
    LOG(ERROR) << ret;
    ASSIGN_OR_RETURN(OutputStreamPoller poller, graph.AddOutputStreamPoller("out"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    for (size_t i = 0; i < requestCount; ++i) {
        ov::Shape ovShape{1, 10};
        ov::Tensor input_tensor(ov::element::Type_t::f32,
            ovShape);
        float* inputData = (float*)input_tensor.data();
        std::memcpy(input_tensor.data(), reinterpret_cast<void*>(inputData), input_tensor.get_byte_size());
        for (int x = 0; x < 10; ++x) {
            inputData[x] = (float)(i * x);
        }
        auto abstatus = graph.AddPacketToInputStream(
            "in1", MakePacket<ov::Tensor>(input_tensor).At(Timestamp(i)));
        if (!abstatus.ok()) {
            LOG(ERROR) << "Failed to add packet to stream:" << abstatus;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(inputStreamDelayMs));
        abstatus = graph.AddPacketToInputStream(
            "in2", MakePacket<ov::Tensor>(input_tensor).At(Timestamp(i)));
        if (!abstatus.ok()) {
            LOG(ERROR) << "Failed to add packet to stream:" << abstatus;
        }
        MP_RETURN_IF_ERROR(abstatus);
    }
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in1"));
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in2"));
    Packet packet;
    size_t receivedCount = 0;
    while (poller.Next(&packet)) {
        auto received = packet.Get<ov::Tensor>();
        float* dataOut = (float*)received.data();
        auto timestamp = packet.Timestamp();
        std::stringstream ss;
        ss << "HelloOVMS Received tensor: [";
        for (int x = 0; x < 10; ++x) {
            ss << dataOut[x] << " ";
        }
        ss << " ] receivedCount: " << ++receivedCount << " timestamp: " << timestamp.DebugString() << endl;
        cout << ss.str() << endl;
    }
    auto res = graph.WaitUntilDone();
    // TODO temporary server stopping
    OVMS_Server* cserver = nullptr;
    OVMS_ServerNew(&cserver);
    OVMS_ServerDelete(cserver);
    cout << __FILE__ << ":" << __LINE__ << endl;
    return res;
}
