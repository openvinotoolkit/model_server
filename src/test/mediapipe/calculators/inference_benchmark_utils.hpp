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

#ifndef SRC_TEST_MEDIAPIPE_CALCULATORS_INFERENCE_BENCHMARK_UTILS_HPP_
#define SRC_TEST_MEDIAPIPE_CALCULATORS_INFERENCE_BENCHMARK_UTILS_HPP_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include "performance_benchmark_config.hpp"

namespace inference_benchmark {
using Clock = std::chrono::steady_clock;
constexpr int kTotalFrames = benchmark_config::kWarmupFrames + benchmark_config::kMeasuredFrames;
inline const std::string kYInputName = std::string(benchmark_config::kInputName) + "/y";
inline const std::string kUvInputName = std::string(benchmark_config::kInputName) + "/uv";

inline std::string environment(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

inline std::filesystem::path root() {
    return environment("OVMS_REPO_ROOT", std::filesystem::current_path().string());
}

inline std::string modelPath() {
    return environment("OVMS_PERF_MODEL_XML",
        (root() / "src/test/face_detection_adas/1/face-detection-adas-0001.xml").string());
}

inline std::string videoPath() {
    return environment("VIDEO_PATH", (root() / "recording_3_raw.avi").string());
}

inline ov::AnyMap properties() {
    return {{"PERFORMANCE_HINT", benchmark_config::kPerformanceHint},
        {"NUM_STREAMS", benchmark_config::kNumStreams}};
}

inline std::shared_ptr<ov::Model> model(ov::Core& core, bool surface) {
    ov::preprocess::PrePostProcessor processor(core.read_model(modelPath()));
    processor.input().tensor().set_element_type(ov::element::u8).set_spatial_static_shape(benchmark_config::kModelHeight, benchmark_config::kModelWidth).set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"});
    if (surface)
        processor.input().tensor().set_memory_type(ov::intel_gpu::memory_type::surface);
    processor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    processor.input().model().set_layout("NCHW");
    return processor.build();
}

inline std::vector<std::vector<uint8_t>> frames() {
    const std::string path = environment("OVMS_PERF_NV12_PATH", (root() / "logs/benchmark_input.nv12").string());
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Missing NV12 fixture: " + path);
    constexpr size_t frameSize = benchmark_config::kModelWidth * benchmark_config::kModelHeight * 3 / 2;
    std::vector<std::vector<uint8_t>> result;
    for (;;) {
        std::vector<uint8_t> frame(frameSize);
        input.read(reinterpret_cast<char*>(frame.data()), frame.size());
        if (input.gcount() == 0)
            break;
        if (input.gcount() != static_cast<std::streamsize>(frame.size()))
            throw std::runtime_error("Incomplete NV12 frame");
        result.push_back(std::move(frame));
    }
    if (result.size() != kTotalFrames)
        throw std::runtime_error("Benchmark requires the complete 250-frame source, got " + std::to_string(result.size()));
    return result;
}

inline double milliseconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

inline void report(const std::string& name, const std::vector<double>& samples) {
    std::vector<double> fps;
    for (size_t sample = 0; sample < samples.size(); ++sample) {
        fps.push_back(1000.0 / samples[sample]);
        std::cout << std::fixed << std::setprecision(6) << "BENCHMARK case=" << name
                  << " sample=" << sample + 1 << " warmup=" << benchmark_config::kWarmupFrames
                  << " measured=" << benchmark_config::kMeasuredFrames
                  << " consumed=" << kTotalFrames << " eos=1"
                  << " ms_per_frame=" << samples[sample] << " fps=" << fps.back() << '\n';
    }
    const auto summarize = [&](const std::vector<double>& values, const char* unit) {
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0;
        for (double value : values)
            variance += (value - mean) * (value - mean);
        const double deviation = std::sqrt(variance / (values.size() - 1));
        std::cout << "SUMMARY case=" << name << " unit=" << unit << " samples=" << values.size()
                  << " mean=" << mean << " sample_stddev=" << deviation
                  << " cv_percent=" << 100.0 * deviation / mean << std::endl;
    };
    summarize(samples, "ms_per_frame");
    summarize(fps, "fps");
}
}  // namespace inference_benchmark
#endif