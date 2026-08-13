#ifndef MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_KALMAN_FILTER_H_
#define MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_KALMAN_FILTER_H_
#include <vector>

#include <Eigen/Dense>

#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {
namespace bytetrack {

using Mean = Eigen::Matrix<float, 1, 8>;
using Cov = Eigen::Matrix<float, 8, 8>;
using MeanMatrix = Eigen::Matrix<float, Eigen::Dynamic, 8>;
using CovMatrix = std::vector<Cov>;
class KalmanFilter {
public:
    KalmanFilter();
    std::pair<Mean, Cov> Initiate(const Eigen::Vector4f detection);
    std::pair<Mean, Cov> Predict(const Mean& mean, const Cov& cov) const;
    std::pair<Mean, Cov> Update(const Mean& mean, const Cov& cov, const Eigen::Vector4f xyah) const;
    std::pair<MeanMatrix, CovMatrix> MultiPredict(const MeanMatrix& means, const CovMatrix& covs) const;

private:
    Eigen::Matrix<float, 8, 8> motion_mat_;
    Eigen::Matrix<float, 4, 8> update_mat_;
    float kStdWeightPos_;
    float kStdWeightVel_;
};

}  // namespace bytetrack
}  // namespace mediapipe

#endif