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

#include "absl/strings/str_format.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/util/color.pb.h"
#pragma GCC diagnostic pop

#include "src/bytetrack/utils/detection_color_by_id_calculator.pb.h"

namespace mediapipe {
namespace {
class DetectionColorByIdCalculatorTest : public ::testing::Test {
protected:
    std::unique_ptr<CalculatorRunner> MakeRunner(float saturation, float value, float thickness = 5.0f) {
        std::string pbtxt = absl::StrFormat(R"pb(
            calculator: "DetectionColorByIdCalculator"
            input_stream: "DETECTIONS:detections"
            output_stream: "RENDER_DATA:render_data"
            node_options: {
                [type.googleapis.com/mediapipe.DetectionColorByIdCalculatorOptions]: {
                    saturation: %f
                    value: %f
                    thickness: %f
                }
            }
        )pb",
            saturation, value, thickness);
        return std::make_unique<CalculatorRunner>(pbtxt);
    }

    absl::Status RunWithDetections(CalculatorRunner& runner, std::vector<Detection> detections) {
        auto input = std::make_unique<std::vector<Detection>>(std::move(detections));
        runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(Adopt(input.release()).At(Timestamp(0)));
        return runner.Run();
    }

    Detection MakeDetection(int id, const std::string& label = "", float score = -1.0f) {
        Detection det;
        det.set_detection_id(id);
        auto* bbox = det.mutable_location_data()->mutable_relative_bounding_box();
        bbox->set_xmin(0.1f);
        bbox->set_ymin(0.1f);
        bbox->set_width(0.2f);
        bbox->set_height(0.2f);
        if (!label.empty()) {
            det.add_label(label);
        }
        if (score >= 0.0f) {
            det.add_score(score);
        }
        return det;
    }

    Detection MakeDetectionWithoutId() {
        Detection det;
        auto* bbox = det.mutable_location_data()->mutable_relative_bounding_box();
        bbox->set_xmin(0.1f);
        bbox->set_ymin(0.1f);
        bbox->set_width(0.2f);
        bbox->set_height(0.2f);
        return det;
    }
};

// These tests should fail because saturation and value go out of bounds

TEST_F(DetectionColorByIdCalculatorTest, SaturationBelowZeroFailsOpen) {
    auto runner = MakeRunner(-0.1f, 0.5f);
    EXPECT_FALSE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, SaturationAboveOneFailsOpen) {
    auto runner = MakeRunner(1.1f, 0.5f);
    EXPECT_FALSE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, ValueBelowZeroFailsOpen) {
    auto runner = MakeRunner(0.5f, -0.1f);
    EXPECT_FALSE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, ValueAboveOneFailsOpen) {
    auto runner = MakeRunner(0.5f, 1.1f);
    EXPECT_FALSE(RunWithDetections(*runner, {}).ok());
}

// These tests should pass because saturation and value are within bounds

TEST_F(DetectionColorByIdCalculatorTest, SaturationAtZeroValid) {
    auto runner = MakeRunner(0.0f, 0.5f);
    EXPECT_TRUE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, SaturationAtOneValid) {
    auto runner = MakeRunner(1.0f, 0.5f);
    EXPECT_TRUE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, ValueAtZeroValid) {
    auto runner = MakeRunner(0.5f, 0.0f);
    EXPECT_TRUE(RunWithDetections(*runner, {}).ok());
}

TEST_F(DetectionColorByIdCalculatorTest, ValueAtOneValid) {
    auto runner = MakeRunner(0.5f, 1.0f);
    EXPECT_TRUE(RunWithDetections(*runner, {}).ok());
}

// test with passing a detection without id

TEST_F(DetectionColorByIdCalculatorTest, SkipsDetectionWithoutId) {
    auto runner = MakeRunner(0.85f, 0.95f);

    std::vector<Detection> detections;
    detections.push_back(MakeDetection(1));
    detections.push_back(MakeDetectionWithoutId());

    ASSERT_TRUE(RunWithDetections(*runner, detections).ok());

    const auto& output = runner->Outputs().Tag("RENDER_DATA").packets;
    ASSERT_EQ(output.size(), 1);
    const auto& render_data = output[0].Get<RenderData>();

    // 1 valid detection -> 2 annotations (box + label). The id-less one is skipped.
    EXPECT_EQ(render_data.render_annotations_size(), 2);
}

TEST_F(DetectionColorByIdCalculatorTest, AllDetectionsMissingId_ProducesEmptyRenderData) {
    auto runner = MakeRunner(0.85f, 0.95f);

    std::vector<Detection> detections;
    detections.push_back(MakeDetectionWithoutId());
    detections.push_back(MakeDetectionWithoutId());

    ASSERT_TRUE(RunWithDetections(*runner, detections).ok());

    const auto& output = runner->Outputs().Tag("RENDER_DATA").packets;
    ASSERT_EQ(output.size(), 1);
    const auto& render_data = output[0].Get<RenderData>();
    // output should be zero because all the non id detections are skipped
    EXPECT_EQ(render_data.render_annotations_size(), 0);
}

}  // namespace
}  // namespace mediapipe
