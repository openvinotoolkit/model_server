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

#ifndef SRC_BYTETRACK_CALCULATORS_STRACK_H_
#define SRC_BYTETRACK_CALCULATORS_STRACK_H_

#include <vector>
#include <string>

#include <Eigen/Dense>

#include "mediapipe/framework/formats/detection.pb.h"
#include "src/bytetrack/calculators/kalman_filter.h"
#include "src/bytetrack/calculators/basetrack.h"

namespace mediapipe {
namespace bytetrack {

using Mean = Eigen::Matrix<float, 1, 8>;
using Cov = Eigen::Matrix<float, 8, 8>;
using MeanMatrix = Eigen::Matrix<float, Eigen::Dynamic, 8>;
using CovMatrix = std::vector<Cov>;

class STrack : public BaseTrack {
public:
    static KalmanFilter shared_kalman;

    explicit STrack(const Detection& det);

    void Activate(KalmanFilter* kalman_filter, int frame_id);
    void ReActivate(const STrack& new_track, int frame_id, bool new_id = false);
    void Update(const STrack& new_track, int frame_id);

    void Predict();
    static void MultiPredict(std::vector<STrack*>& tracks);

    Eigen::Vector4f tlwh() const;
    Eigen::Vector4f tlbr() const;
    // static Eigen::Vector4f TlbrToTlwh(const Eigen::Vector4f& tlbr);
    // static Eigen::Vector4f TlwhToTlbr(const Eigen::Vector4f& tlwh);
    static Eigen::Vector4f TlwhToXyah(const Eigen::Vector4f& tlwh);

    // KalmanState ToProto() const;
    int tracklet_len() const { return tracklet_len_; }
    std::string label() const { return label_; }
    const Mean& mean() const { return mean_; }
    const Cov& cov() const { return cov_; }

private:
    // STrack-only — Kalman state and detection origin
    // score_, is_activated_, tracklet_len_ are inherited from BaseTrack — do NOT redeclare
    Eigen::Vector4f tlwh_;
    std::string label_;
    KalmanFilter* kf_ = nullptr;
    Mean mean_ = Mean::Zero();
    Cov cov_ = Cov::Zero();
    int tracklet_len_ = 0;
};

}  // namespace bytetrack
}  // namespace mediapipe

#endif  // SRC_BYTETRACK_CALCULATORS_STRACK_H_
