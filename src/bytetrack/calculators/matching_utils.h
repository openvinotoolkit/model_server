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

#ifndef SRC_BYTETRACK_CALCULATORS_MATCHING_UTILS_H_
#define SRC_BYTETRACK_CALCULATORS_MATCHING_UTILS_H_

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/bytetrack/calculators/strack.h"

namespace mediapipe {
namespace bytetrack {

struct AssignmentResult {
    Eigen::MatrixXi matches;           // shape (K, 2) — col 0 = track idx, col 1 = box idx
    Eigen::VectorXi unmatched_tracks;  // shape (P,)
    Eigen::VectorXi unmatched_boxes;   // shape (Q,)
};

// IoU between two [top, left, bottom, right] boxes
inline float ComputeIoU(const Eigen::Vector4f& a,
    const Eigen::Vector4f& b) {
    float inter_x1 = std::max(a[0], b[0]);  // left
    float inter_y1 = std::max(a[1], b[1]);  // top
    float inter_x2 = std::min(a[2], b[2]);  // right
    float inter_y2 = std::min(a[3], b[3]);  // bottom

    float inter_w = std::max(0.f, inter_x2 - inter_x1);
    float inter_h = std::max(0.f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    if (inter_area == 0.f)
        return 0.f;

    float area_a = (a[2] - a[0]) * (a[3] - a[1]);  // w * h
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter_area / (area_a + area_b - inter_area);
}

inline Eigen::MatrixXf BuildIoUCostMatrix(
    const std::vector<bytetrack::STrack*>& tracks,
    const std::vector<bytetrack::STrack>& detections) {
    int N = tracks.size();
    int M = detections.size();

    Eigen::MatrixXf cost(N, M);

    for (int i = 0; i < N; ++i) {
        auto tb = tracks[i]->tlbr();
        for (int j = 0; j < M; ++j) {
            auto bb = detections[j].tlbr();
            cost(i, j) = 1.f - ComputeIoU(tb, bb);
        }
    }

    return cost;
}

inline Eigen::MatrixXf FuseScore(
    const Eigen::MatrixXf cost_matrix,
    const std::vector<bytetrack::STrack>& detections) {
    if (cost_matrix.size() == 0) {
        return cost_matrix;
    }
    Eigen::MatrixXf iou_sim = 1.0f - cost_matrix.array();

    Eigen::RowVectorXf det_scores(detections.size());
    for (int i = 0; i < (int)detections.size(); i++) {
        det_scores(i) = detections[i].score();
    }

    Eigen::MatrixXf det_scores_mat = det_scores.replicate(cost_matrix.rows(), 1);
    Eigen::MatrixXf fuse_sim = iou_sim.array() * det_scores_mat.array();
    Eigen::MatrixXf fuse_cost = 1.0f - fuse_sim.array();

    return fuse_cost;
}

inline Eigen::MatrixXf BuildIoUCostMatrix(
    const std::vector<bytetrack::STrack*>& a,
    const std::vector<bytetrack::STrack*>& b) {
    int N = a.size();
    int M = b.size();

    Eigen::MatrixXf cost(N, M);

    for (int i = 0; i < N; ++i) {
        auto ta = a[i]->tlbr();
        for (int j = 0; j < M; ++j) {
            auto tb = b[j]->tlbr();
            cost(i, j) = 1.f - ComputeIoU(ta, tb);
        }
    }

    return cost;
}

// inline AssignmentResult LinearAssignment(const Eigen::MatrixXf& cost, float thresh){
//     int N = (int)cost.rows();
//     int M = (int)cost.cols();

//     // Collect candidates below threshold
//     std::vector<std::tuple<float,int,int>> entries;
//     entries.reserve(N * M);
//     for (int i = 0; i < N; ++i)
//         for (int j = 0; j < M; ++j)
//             if (cost(i,j) <= thresh)
//                 entries.emplace_back(cost(i,j), i, j);

//     std::sort(entries.begin(), entries.end());

//     // Greedy assignment
//     std::vector<bool> track_used(N, false);
//     std::vector<bool> box_used(M, false);
//     std::vector<std::pair<int,int>> matched;
//     matched.reserve(std::min(N, M));

//     for (auto& [c, i, j] : entries) {
//         if (!track_used[i] && !box_used[j]) {
//             matched.emplace_back(i, j);
//             track_used[i] = true;
//             box_used[j]   = true;
//         }
//     }

//     // Pack into Eigen outputs
//     int K = (int)matched.size();
//     Eigen::MatrixXi matches(K, 2);          // (K,2) — mirrors np.empty((0,2)) when K=0
//     for (int k = 0; k < K; ++k) {
//         matches(k, 0) = matched[k].first;
//         matches(k, 1) = matched[k].second;
//     }

//     // Count unmatched first, then fill — avoids push_back on Eigen vectors
//     int n_ut = (int)std::count(track_used.begin(), track_used.end(), false);
//     int n_ub = (int)std::count(box_used.begin(),   box_used.end(),   false);

//     Eigen::VectorXi unmatched_tracks(n_ut);
//     Eigen::VectorXi unmatched_boxes(n_ub);

//     for (int i = 0, k = 0; i < N; ++i)
//         if (!track_used[i]) unmatched_tracks(k++) = i;
//     for (int j = 0, k = 0; j < M; ++j)
//         if (!box_used[j])   unmatched_boxes(k++) = j;

//     return {matches, unmatched_tracks, unmatched_boxes};
// }

inline AssignmentResult LinearAssignment(const Eigen::MatrixXf& cost, float thresh) {
    int N = (int)cost.rows();
    int M = (int)cost.cols();

    // Empty matrix early exit — mirrors Python: if cost_matrix.size == 0
    if (N == 0 || M == 0) {
        Eigen::MatrixXi matches(0, 2);
        Eigen::VectorXi u_tracks(N), u_boxes(M);
        for (int i = 0; i < N; ++i)
            u_tracks(i) = i;
        for (int j = 0; j < M; ++j)
            u_boxes(j) = j;
        return {matches, u_tracks, u_boxes};
    }

    // Pad to square S x S — mirrors lap.lapjv extend_cost=True
    int S = std::max(N, M);
    const float INF = 1e9f;

    Eigen::MatrixXf cost_sq = Eigen::MatrixXf::Constant(S, S, INF);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            cost_sq(i, j) = cost(i, j);

    // Dual variables and assignment vectors
    std::vector<float> u(S, 0.f), v(S, 0.f);
    std::vector<int> row2col(S, -1), col2row(S, -1);

    // Phase 1: Column reduction — init v[j] to column minimum
    for (int j = 0; j < S; ++j) {
        int best_i = 0;
        float best_v = cost_sq(0, j);
        for (int i = 1; i < S; ++i) {
            if (cost_sq(i, j) < best_v) {
                best_v = cost_sq(i, j);
                best_i = i;
            }
        }
        v[j] = best_v;
        if (row2col[best_i] == -1) {
            row2col[best_i] = j;
            col2row[j] = best_i;
        }
    }

    // Phase 2: Augmenting row reduction (2 passes)
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < S; ++i) {
            if (row2col[i] != -1)
                continue;
            int j1 = -1, j2 = -1;
            float u1 = INF, u2 = INF;
            for (int j = 0; j < S; ++j) {
                float h = cost_sq(i, j) - v[j];
                if (h < u2) {
                    if (h < u1) {
                        u2 = u1;
                        j2 = j1;
                        u1 = h;
                        j1 = j;
                    } else {
                        u2 = h;
                        j2 = j;
                    }
                }
            }
            u[i] = u1;
            if (col2row[j1] == -1) {
                row2col[i] = j1;
                col2row[j1] = i;
            } else {
                v[j1] -= (u2 - u1);
            }
        }
    }

    // Phase 3: Augmentation via shortest path (Dijkstra with potentials)
    std::vector<float> dist(S);
    std::vector<int> pred(S, -1);
    std::vector<bool> visited(S, false);

    for (int i_start = 0; i_start < S; ++i_start) {
        if (row2col[i_start] != -1)
            continue;

        std::fill(dist.begin(), dist.end(), INF);
        std::fill(pred.begin(), pred.end(), -1);
        std::fill(visited.begin(), visited.end(), false);

        for (int j = 0; j < S; ++j)
            dist[j] = cost_sq(i_start, j) - u[i_start] - v[j];

        int j_end = -1;
        float d_min = INF;

        for (int iter = 0; iter < S; ++iter) {
            // Pick unvisited col with smallest dist
            int j_min = -1;
            d_min = INF;
            for (int j = 0; j < S; ++j)
                if (!visited[j] && dist[j] < d_min) {
                    d_min = dist[j];
                    j_min = j;
                }

            if (j_min == -1)
                break;
            visited[j_min] = true;

            if (col2row[j_min] == -1) {
                j_end = j_min;
                break;
            }

            // Relax edges through the row that owns j_min
            int i_next = col2row[j_min];
            u[i_next] = cost_sq(i_next, j_min) - v[j_min] - d_min;  // update dual
            for (int j = 0; j < S; ++j) {
                if (visited[j])
                    continue;
                float nd = d_min + cost_sq(i_next, j) - u[i_next] - v[j];
                if (nd < dist[j]) {
                    dist[j] = nd;
                    pred[j] = j_min;
                }
            }
        }

        // Update col duals along the path
        for (int j = 0; j < S; ++j)
            if (visited[j])
                v[j] += dist[j] - d_min;
        u[i_start] += d_min;

        // Augment: flip assignments along path back to i_start
        int j_cur = j_end;
        while (j_cur != -1) {
            int i_cur = (pred[j_cur] == -1) ? i_start : col2row[pred[j_cur]];
            col2row[j_cur] = i_cur;
            row2col[i_cur] = j_cur;
            j_cur = pred[j_cur];
        }
    }

    // Extract matches — mirrors: for ix, mx in enumerate(x): if mx >= 0
    // Apply cost_limit=thresh filter here
    std::vector<bool> track_used(N, false);
    std::vector<bool> box_used(M, false);
    std::vector<std::pair<int, int>> matched;

    for (int i = 0; i < N; ++i) {
        int j = row2col[i];
        if (j < M && cost(i, j) <= thresh) {
            matched.emplace_back(i, j);
            track_used[i] = true;
            box_used[j] = true;
        }
    }

    // Pack into AssignmentResult
    int K = (int)matched.size();
    Eigen::MatrixXi matches(K, 2);
    for (int k = 0; k < K; ++k) {
        matches(k, 0) = matched[k].first;
        matches(k, 1) = matched[k].second;
    }

    int n_ut = (int)std::count(track_used.begin(), track_used.end(), false);
    int n_ub = (int)std::count(box_used.begin(), box_used.end(), false);

    Eigen::VectorXi unmatched_tracks(n_ut);
    Eigen::VectorXi unmatched_boxes(n_ub);

    for (int i = 0, k = 0; i < N; ++i)
        if (!track_used[i])
            unmatched_tracks(k++) = i;
    for (int j = 0, k = 0; j < M; ++j)
        if (!box_used[j])
            unmatched_boxes(k++) = j;

    return {matches, unmatched_tracks, unmatched_boxes};
}

}  // namespace bytetrack
}  // namespace mediapipe

#endif  // SRC_BYTETRACK_CALCULATORS_MATCHING_UTILS_H_
