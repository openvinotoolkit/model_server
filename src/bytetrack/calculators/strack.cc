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

#include "src/bytetrack/calculators/basetrack.h"

#include <cstddef>
#include <utility>

#include <Eigen/Dense>

#include "mediapipe/framework/formats/detection.pb.h"
#include "src/bytetrack/calculators/kalman_filter.h"
#include "src/bytetrack/calculators/kalman_matrices.h"
#include "src/bytetrack/calculators/strack.h"

namespace mediapipe {
namespace bytetrack {

KalmanFilter STrack::shared_kalman;

STrack::STrack(const Detection& det) {
    const auto& loc = det.location_data();
    score_ = det.score_size() > 0 ? det.score(0) : 0.0f;
    label_ = det.label_size() > 0 ? det.label(0) : "";
    if (loc.format() == LocationData::RELATIVE_BOUNDING_BOX) {
        const auto& rb = loc.relative_bounding_box();
        tlwh_ << rb.xmin(), rb.ymin(), rb.width(), rb.height();
    } else if (loc.format() == LocationData::BOUNDING_BOX) {
        const auto& b = loc.bounding_box();
        tlwh_ << b.xmin(), b.ymin(), b.width(), b.height();
    } else {
        tlwh_.setZero();
    }
}

void STrack::Predict() {
    Mean mean_state = mean();
    if (state() != TrackState::TRACKED) {
        mean_state(7) = 0.f;
    }
    auto [new_mean, new_cov] = kf_->Predict(mean_state, cov());
    mean_ = new_mean;
    cov_ = new_cov;
}

void STrack::Activate(KalmanFilter* kalman_filter, int frame_id) {
    kf_ = kalman_filter;
    track_id_ = next_id();
    auto [new_mean, new_cov] = kf_->Initiate(TlwhToXyah(tlwh_));
    mean_ = new_mean;
    cov_ = new_cov;

    tracklet_len_ = 0;
    state_ = TrackState::TRACKED;
    if (frame_id == 1)
        is_activated_ = true;
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

void STrack::ReActivate(const STrack& new_track, int frame_id, bool new_id) {
    Eigen::Vector4f new_tlwh = new_track.tlwh_;
    auto [new_mean, new_cov] = kf_->Update(mean_, cov_, TlwhToXyah(new_tlwh));
    mean_ = new_mean;
    cov_ = new_cov;
    tracklet_len_ = 0;
    state_ = TrackState::TRACKED;
    is_activated_ = true;

    frame_id_ = frame_id;
    if (new_id)
        track_id_ = next_id();
    score_ = new_track.score();
}

void STrack::Update(const STrack& new_track, int frame_id) {
    frame_id_ = frame_id;
    tracklet_len_ += 1;

    Eigen::Vector4f new_tlwh = new_track.tlwh_;
    auto [new_mean, new_cov] = kf_->Update(mean_, cov_, TlwhToXyah(new_tlwh));
    mean_ = new_mean;
    cov_ = new_cov;

    state_ = TrackState::TRACKED;
    is_activated_ = true;
    score_ = new_track.score();
}

void STrack::MultiPredict(std::vector<STrack*>& tracks) {
    std::size_t n = tracks.size();
    if (n > 0) {
        MeanMatrix multi_mean(n, 8);
        for (int i = 0; i < n; i++) {
            multi_mean.row(i) = tracks[i]->mean_;
        }
        CovMatrix multi_cov(n);
        for (int i = 0; i < n; i++) {
            multi_cov[i] = tracks[i]->cov_;
        }
        for (int i = 0; i < n; i++) {
            if (tracks[i]->state_ != TrackState::TRACKED) {
                multi_mean(i, 7) = 0.f;
            }
        }
        auto [updated_means, updated_covs] = shared_kalman.MultiPredict(multi_mean, multi_cov);
        for (int i = 0; i < n; i++) {
            tracks[i]->mean_ = updated_means.row(i);
            tracks[i]->cov_ = updated_covs[i];
        }
    }
}

Eigen::Vector4f STrack::TlwhToXyah(const Eigen::Vector4f& tlwh) {
    Eigen::Vector4f xyah;
    float x = tlwh(0);
    float y = tlwh(1);
    float w = tlwh(2);
    float h = tlwh(3);

    xyah(0) = x + w / 2.0f;
    xyah(1) = y + h / 2.0f;
    xyah(2) = w / h;
    xyah(3) = h;

    return xyah;
}

Eigen::Vector4f STrack::tlwh() const {
    // Before activation — return the raw detection box
    if (kf_ == nullptr)
        return tlwh_;

    // After activation — reconstruct from Kalman mean
    // mean_ = [cx, cy, ar, h, vx, vy, var, vh]
    Eigen::Vector4f ret;
    ret(0) = mean_(0);
    ret(1) = mean_(1);
    ret(2) = mean_(2) * mean_(3);
    ret(3) = mean_(3);
    ret(0) -= ret(2) / 2.f;
    ret(1) -= ret(3) / 2.f;
    return ret;
}

Eigen::Vector4f STrack::tlbr() const {
    Eigen::Vector4f ret = tlwh();
    ret(2) += ret(0);
    ret(3) += ret(1);
    return ret;
}

}  // namespace bytetrack
}  // namespace mediapipe
