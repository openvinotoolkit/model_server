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

#include "src/bytetrack/calculators/bytetrack_calculator.pb.h"

namespace mediapipe {
namespace {
class ByteTrackCalculatorTest : public ::testing::Test {
protected:
    std::unique_ptr<CalculatorRunner> MakeRunner(float track_high_threshold, float track_low_threshold, float new_track_threshold, float matching_threshold, int track_buffer, bool fuse_score) {
        std::string pbtxt = absl::StrFormat(R"pb(node {
  calculator: "ByteTrackCalculator"
  input_stream: "DETECTIONS:output_detections"
  output_stream: "DETECTIONS:tracked_detections"
  options: {
    [mediapipe.ByteTrackCalculatorOptions.ext] {
      track_high_threshold:%f
      track_low_threshold:%f
      new_track_threshold:%f
      matching_threshold: %f
      track_buffer: %d
      fuse_score: %v
    }
  }
})pb",
            track_high_threshold, track_low_threshold, new_track_threshold, matching_threshold, track_buffer, fuse_score);
        return std::make_unique<CalculatorRunner>(pbtxt);
    }
};

// ---------- track high threshold tests ----------

TEST_F(ByteTrackCalculatorTest, TrackHighThresholdBelowZeroFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner((-1) * NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

TEST_F(ByteTrackCalculatorTest, TrackHighThresholdAboveOneFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(1 + NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

// ---------- track low threshold tests ----------

TEST_F(ByteTrackCalculatorTest, TrackLowThresholdBelowZeroFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        (-1) * NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

TEST_F(ByteTrackCalculatorTest, TrackLowThresholdAboveOneFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        1 + NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

// ---------- matching threshold tests ----------

TEST_F(ByteTrackCalculatorTest, MatchingThresholdBelowZeroFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        (-1) * NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

TEST_F(ByteTrackCalculatorTest, MatchingThresholdAboveOneFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        1 + NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

// ---------- new track threshold tests ----------

TEST_F(ByteTrackCalculatorTest, NewTrackThresholdBelowZeroFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        (-1) * NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

TEST_F(ByteTrackCalculatorTest, NewTrackThresholdAboveOneFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        1 + NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

// ---------- track_buffer boundary tests ----------

TEST_F(ByteTrackCalculatorTest, TrackBufferZeroIsNotValid) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        0,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

TEST_F(ByteTrackCalculatorTest, TrackBufferNegativeFails) {
    const float NORMAL_TRACK_HIGH_THRESHOLD = 0.7f;
    const float NORMAL_TRACK_LOW_THRESHOLD = 0.5f;
    const float NORMAL_NEW_TRACK_THRESHOLD = 0.3;
    const float NORMAL_MATCHING_THRESHOLD = 0.8;
    const int NORMAL_TRACK_BUFFER = 30;
    auto runner = MakeRunner(NORMAL_TRACK_HIGH_THRESHOLD,
        NORMAL_TRACK_LOW_THRESHOLD,
        NORMAL_NEW_TRACK_THRESHOLD,
        NORMAL_MATCHING_THRESHOLD,
        (-1) * NORMAL_TRACK_BUFFER,
        false);
    auto status = runner->Run();
    EXPECT_FALSE(status.ok());
}

// ---------- fuse_score toggle sanity ----------

TEST_F(ByteTrackCalculatorTest, FuseScoreTrueIsValid) {
    auto runner = MakeRunner(0.7f,
        0.5f,
        0.3f,
        0.8f,
        30,
        true);
    MP_ASSERT_OK(runner->Run());
}
}  // namespace
}  // namespace mediapipe
