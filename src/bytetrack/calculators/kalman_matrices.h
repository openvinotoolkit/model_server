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
// kalman_matrices.h
#ifndef MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_KALMAN_MATRICES_H_
#define MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_KALMAN_MATRICES_H_

#include <Eigen/Dense>

namespace mediapipe {
namespace bytetrack {

static constexpr float kStdWeightPos = 1.0f / 20.0f;
static constexpr float kStdWeightVel = 1.0f / 160.0f;

inline Eigen::Matrix<float, 8, 8> MakeMotionMatrix() {
    Eigen::Matrix<float, 8, 8> F = Eigen::Matrix<float, 8, 8>::Identity();
    F.block<4, 4>(0, 4) = Eigen::Matrix4f::Identity();
    return F;
}

inline Eigen::Matrix<float, 4, 8> MakeUpdateMatrix() {
    Eigen::Matrix<float, 4, 8> H = Eigen::Matrix<float, 4, 8>::Zero();
    H.block<4, 4>(0, 0) = Eigen::Matrix4f::Identity();
    return H;
}

}  // namespace bytetrack
}  // namespace mediapipe

#endif