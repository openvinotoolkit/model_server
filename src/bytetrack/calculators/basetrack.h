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

#ifndef SRC_BYTETRACK_CALCULATORS_BASETRACK_H_
#define SRC_BYTETRACK_CALCULATORS_BASETRACK_H_

namespace mediapipe {
namespace bytetrack {

class BaseTrack {
public:
    enum class TrackState { NEW,
        TRACKED,
        LOST,
        REMOVED };

    // mirrors Python's next_id() — static counter owned here
    static int next_id() { return ++count_; }
    static void reset_id() { count_ = 0; }

    // Accessors
    int track_id() const { return track_id_; }
    int frame_id() const { return frame_id_; }
    int start_frame() const { return start_frame_; }
    float score() const { return score_; }
    TrackState state() const { return state_; }
    bool is_activated() const { return is_activated_; }

    void MarkLost() { state_ = TrackState::LOST; }
    void MarkRemoved() { state_ = TrackState::REMOVED; }

protected:
    int track_id_ = 0;
    int frame_id_ = 0;
    int start_frame_ = 0;
    float score_ = 0.f;
    bool is_activated_ = false;
    TrackState state_ = TrackState::NEW;
    static int count_;
};

}  // namespace bytetrack
}  // namespace mediapipe

#endif  // SRC_BYTETRACK_CALCULATORS_BASETRACK_H_
