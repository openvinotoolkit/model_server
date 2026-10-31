// Copyright (c) 2026 Intel Corporation
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

#ifndef SRC_TEST_MEDIAPIPE_CALCULATORS_PERFORMANCE_BENCHMARK_CONFIG_HPP_
#define SRC_TEST_MEDIAPIPE_CALCULATORS_PERFORMANCE_BENCHMARK_CONFIG_HPP_

namespace benchmark_config {

constexpr char kModelName[] = "face_detection_adas";
constexpr char kInputName[] = "data";
constexpr int kModelWidth = 672;
constexpr int kModelHeight = 384;
constexpr int kWarmupFrames = 20;
constexpr int kMeasuredFrames = 230;
constexpr int kSampleCount = 10;
constexpr int kRequestCount = 4;
constexpr char kPerformanceHint[] = "LATENCY";
constexpr char kNumStreams[] = "1";

}  // namespace benchmark_config

#endif  // SRC_TEST_MEDIAPIPE_CALCULATORS_PERFORMANCE_BENCHMARK_CONFIG_HPP_