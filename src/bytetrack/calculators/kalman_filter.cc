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

#include "src/bytetrack/calculators/kalman_matrices.h"

#include <utility>

#include <Eigen/Dense>

#include "mediapipe/framework/formats/detection.pb.h"
#include "src/bytetrack/calculators/kalman_filter.h"

namespace mediapipe {
namespace bytetrack {

KalmanFilter::KalmanFilter() :
    motion_mat_(MakeMotionMatrix()),
    update_mat_(MakeUpdateMatrix()),
    kStdWeightPos_(kStdWeightPos),
    kStdWeightVel_(kStdWeightVel) {}

std::pair<Mean, Cov> KalmanFilter::Initiate(const Eigen::Vector4f detection) {
    Mean mean;
    mean << detection(0), detection(1), detection(2), detection(3), 0.f, 0.f, 0.f, 0.f;

    Eigen::Matrix<float, 8, 1> std_vals;
    std_vals << 2.f * kStdWeightPos_ * detection(3),
        2.f * kStdWeightPos_ * detection(3),
        1e-2f,
        2.f * kStdWeightPos_ * detection(3),
        10.f * kStdWeightVel_ * detection(3),
        10.f * kStdWeightVel_ * detection(3),
        1e-5f,
        10.f * kStdWeightVel_ * detection(3);

    Cov cov = std_vals.array().square().matrix().asDiagonal();
    return {mean, cov};
}

std::pair<Mean, Cov> KalmanFilter::Predict(const Mean& mean, const Cov& cov) const {
    const float h = mean(3);
    Eigen::Matrix<float, 1, 8> std_pv;
    std_pv << kStdWeightPos_ * h, kStdWeightPos_ * h, 1e-2f, kStdWeightPos_ * h,
        kStdWeightVel_ * h, kStdWeightVel_ * h, 1e-5f, kStdWeightVel_ * h;
    const Cov motion_cov = std_pv.array().square().matrix().asDiagonal();
    const Mean est_mean = mean * motion_mat_.transpose();
    const Cov est_cov = motion_mat_ * cov * motion_mat_.transpose() + motion_cov;

    return {est_mean, est_cov};
}

std::pair<Mean, Cov> KalmanFilter::Update(const Mean& mean, const Cov& cov, const Eigen::Vector4f xyah) const {
    const float x_c = xyah(0);
    const float y_c = xyah(1);
    const float ar = xyah(2);
    const float h = xyah(3);

    Eigen::Matrix<float, 1, 4> measurement;
    measurement << x_c, y_c, ar, h;
    // Innovation covariance in measurement space
    Eigen::Matrix<float, 1, 4> noise_std;
    noise_std << kStdWeightPos_ * mean(3),
        kStdWeightPos_ * mean(3),
        1e-1f,
        kStdWeightPos_ * mean(3);
    const Eigen::Matrix<float, 4, 4> innov_cov = noise_std.array().square().matrix().asDiagonal();

    const Eigen::Matrix<float, 1, 4> proj_mean = mean * update_mat_.transpose();
    const Eigen::Matrix<float, 4, 4> proj_cov = update_mat_ * cov * update_mat_.transpose() + innov_cov;

    // Kalman gain via Cholesky solve: K = (P H^T) (H P H^T + R)^{-1}
    const Eigen::Matrix<float, 8, 4> PHt = cov * update_mat_.transpose();
    const Eigen::Matrix<float, 8, 4> K = proj_cov.llt().solve(PHt.transpose()).transpose();

    const Mean updated_mean = mean + (measurement - proj_mean) * K.transpose();
    const Cov updated_cov = cov - K * proj_cov * K.transpose();

    return {updated_mean, updated_cov};
}

std::pair<MeanMatrix, CovMatrix> KalmanFilter::MultiPredict(const MeanMatrix& means, const CovMatrix& covs) const {
    const int N = means.rows();

    // Build Nx8 std matrix — each row is the std devs for one track
    MeanMatrix std_mat(N, 8);
    for (int i = 0; i < N; ++i) {
        const float h = means(i, 3);
        std_mat.row(i) << kStdWeightPos_ * h, kStdWeightPos_ * h, 1e-2f, kStdWeightPos_ * h,
            kStdWeightVel_ * h, kStdWeightVel_ * h, 1e-5f, kStdWeightVel_ * h;
    }

    // Predicted means: (N,8) @ F^T
    MeanMatrix pred_means = means * motion_mat_.transpose();

    // Predicted covariances per track
    CovMatrix pred_covs(N);
    for (int i = 0; i < N; ++i) {
        const Cov motion_cov = std_mat.row(i)
                                   .array()
                                   .square()
                                   .matrix()
                                   .asDiagonal();
        pred_covs[i] = motion_mat_ * covs[i] * motion_mat_.transpose() + motion_cov;
    }

    return {pred_means, pred_covs};
}

}  // namespace bytetrack
}  // namespace mediapipe
