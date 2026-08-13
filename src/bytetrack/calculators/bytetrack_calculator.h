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

#ifndef MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_BYTETRACK_CALCULATOR_H_
#define MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_BYTETRACK_CALCULATOR_H_

#include "src/bytetrack/calculators/bytetrack_calculator.pb.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "src/bytetrack/calculators/strack.h"
#include "src/bytetrack/calculators/kalman_filter.h"

namespace mediapipe {
class ByteTrackCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc);
    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;

private:
    ::mediapipe::ByteTrackCalculatorOptions options_;
    std::vector<bytetrack::STrack> tracked_stracks_;
    std::vector<bytetrack::STrack> lost_stracks_;
    std::vector<bytetrack::STrack> removed_stracks_;

    int track_buffer_;
    float det_thresh_;
    float match_thresh_;
    float track_high_thresh_;
    float track_low_thresh_;
    float new_track_thresh_;
    bool fuse_score_;
    int frame_id_ = 0;
    int max_time_lost_ = 30;
    int64_t last_timestamp_ = -1;
    float estimated_fps_ = 30.0f;

    bytetrack::KalmanFilter kalman_filter_;

    static std::vector<bytetrack::STrack*> JointStracks(
        std::vector<bytetrack::STrack*>& a,
        std::vector<bytetrack::STrack*>& b);

    static std::vector<bytetrack::STrack*> SubStracks(
        std::vector<bytetrack::STrack*>& a,
        std::vector<bytetrack::STrack*>& b);

    static std::pair<std::vector<bytetrack::STrack*>, std::vector<bytetrack::STrack*>>
    RemoveDuplicateStracks(
        std::vector<bytetrack::STrack*>& a,
        std::vector<bytetrack::STrack*>& b);
};
}  // namespace mediapipe

#endif