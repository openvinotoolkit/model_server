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