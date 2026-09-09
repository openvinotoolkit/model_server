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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include "absl/strings/str_format.h"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/util/color.pb.h"
#pragma GCC diagnostic pop

#include "src/yolox/ov_yolox_tensors_to_detections_calculator.pb.h"

namespace mediapipe {
namespace {
class OVYoloXTensorsToDetectionsCalculatorTest : public ::testing::Test {
protected:
    std::unique_ptr<CalculatorRunner> MakeRunner(float conf_thresh, float input_size) {
        std::string pbtxt = absl::StrFormat(R"pb(
                                                 node {
                                                   calculator: "OVYoloXTensorsToDetectionsCalculator"
                                                   input_stream: "TENSORS:detection_tensors"
                                                   output_stream: "DETECTIONS:detections"

                                                   node_options: {
                                                           [type.googleapis.com / mediapipe.OVYoloXTensorsToDetectionsCalculatorOptions] {
                                                             conf_thresh: % f
                                                             input_size: % f
                                                           }}
                                                 })pb",
            conf_thresh, input_size);
        return std::make_unique<CalculatorRunner>(pbtxt);
    }
    absl::Status RunOpenOnly(CalculatorRunner& runner) {
        auto tensors = std::make_unique<std::vector<ov::Tensor>>();
        runner.MutableInputs()->Tag("TENSORS").packets.push_back(
            Adopt(tensors.release()).At(Timestamp(0)));
        return runner.Run();
    }
};

// confidence threshold out of bounds

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, ConfThreshBelowZeroFailsOpen) {
    auto runner = MakeRunner(-0.1f, 416.0f);
    EXPECT_FALSE(RunOpenOnly(*runner).ok());
}

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, ConfThreshAboveOneFailsOpen) {
    auto runner = MakeRunner(1.1f, 416.0f);
    EXPECT_FALSE(RunOpenOnly(*runner).ok());
}

// confidence threshold within bounds

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, ConfThreshAtZeroValid) {
    auto runner = MakeRunner(0.0f, 416.0f);
    EXPECT_TRUE(RunOpenOnly(*runner).ok());
}

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, ConfThreshAtOneValid) {
    auto runner = MakeRunner(1.0f, 416.0f);
    EXPECT_TRUE(RunOpenOnly(*runner).ok());
}

// input size out of bounds

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, InputSizeAtZeroFailsOpen) {
    auto runner = MakeRunner(0.5f, 0.0f);
    EXPECT_FALSE(RunOpenOnly(*runner).ok());
}

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, InputSizeNegative_FailsOpen) {
    auto runner = MakeRunner(0.5f, -416.0f);
    EXPECT_FALSE(RunOpenOnly(*runner).ok());
}

// this check passes because input size is in valid range (0,inf)

TEST_F(OVYoloXTensorsToDetectionsCalculatorTest, InputSizeValidPositive_Valid) {
    auto runner = MakeRunner(/*conf_thresh=*/0.5f, /*input_size=*/416.0f);
    EXPECT_TRUE(RunOpenOnly(*runner).ok());
}

}  // namespace
}  // namespace mediapipe
