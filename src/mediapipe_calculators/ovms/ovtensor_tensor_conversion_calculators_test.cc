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

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace {

TEST(OVTensorTensorConversionCalculatorsTest, CopiesTensorDataAndPreservesShape) {
    const std::string graphProto = R"(
      input_stream: "in"
      output_stream: "out"
      node {
        calculator: "MPTensorToOVTensorCalculator"
        input_stream: "TENSOR:in"
        output_stream: "OVTENSOR:ov_tensor"
      }
      node {
        calculator: "OVTensorToMPTensorCalculator"
        input_stream: "OVTENSOR:ov_tensor"
        output_stream: "TENSOR:out"
      }
    )";
    mediapipe::CalculatorGraph graph;
    const auto config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graphProto);
    ASSERT_TRUE(graph.Initialize(config).ok());

    const mediapipe::Timestamp timestamp(23);
    std::vector<float> observedValues;
    ASSERT_TRUE(graph.ObserveOutputStream("out", [&observedValues, timestamp](const mediapipe::Packet& packet) {
                         EXPECT_EQ(packet.Timestamp(), timestamp);
                         const mediapipe::Tensor& tensor = packet.Get<mediapipe::Tensor>();
                         EXPECT_EQ(tensor.element_type(), mediapipe::Tensor::ElementType::kFloat32);
                         EXPECT_EQ(tensor.shape().dims, std::vector<int>({1, 2}));
                         const float* tensorData = tensor.GetCpuReadView().buffer<float>();
                         observedValues.assign(tensorData, tensorData + 2);
                         return absl::OkStatus();
                     })
            .ok());
    ASSERT_TRUE(graph.StartRun({}).ok());

    mediapipe::Tensor input(mediapipe::Tensor::ElementType::kFloat32,
        mediapipe::Tensor::Shape{std::vector<int>{1, 2}});
    float* inputData = input.GetCpuWriteView().buffer<float>();
    inputData[0] = 3.0f;
    inputData[1] = 7.0f;
    ASSERT_TRUE(graph.AddPacketToInputStream(
                         "in", mediapipe::MakePacket<mediapipe::Tensor>(std::move(input))
                                   .At(timestamp))
            .ok());
    ASSERT_TRUE(graph.CloseAllPacketSources().ok());
    ASSERT_TRUE(graph.WaitUntilDone().ok());
    EXPECT_EQ(observedValues, (std::vector<float>{3.0f, 7.0f}));
}

}  // namespace
