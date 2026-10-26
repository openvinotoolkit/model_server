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

#ifndef SRC_TEST_MEDIAPIPE_CALCULATORS_MPI_INFERENCE_BENCHMARK_UTILS_HPP_
#define SRC_TEST_MEDIAPIPE_CALCULATORS_MPI_INFERENCE_BENCHMARK_UTILS_HPP_

#include <map>
#include <memory>
#include <gtest/gtest.h>
#include "mediapipe/framework/calculator_framework.h"
#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mpi/intel_mpi.h"
#include "inference_benchmark_utils.hpp"

namespace inference_benchmark {
struct DecodeStream {
    imp_context_t* context = nullptr;
    imp_video_stream_t* stream = nullptr;
    void open() {
        if (imp_context_create(&context, nullptr) != IMP_OK)
            throw std::runtime_error("MPI context creation failed");
        imp_video_source_t* rawSource = nullptr;
        if (imp_video_source_create(&rawSource, IMP_SOURCE_FILE) != IMP_OK)
            throw std::runtime_error("MPI source creation failed");
        std::unique_ptr<imp_video_source_t, decltype(&imp_video_source_destroy)> source(rawSource, imp_video_source_destroy);
        const std::string path = videoPath();
        const std::string width = std::to_string(benchmark_config::kModelWidth);
        const std::string height = std::to_string(benchmark_config::kModelHeight);
        if (imp_video_source_set(source.get(), "path", path.c_str()) != IMP_OK ||
            imp_video_source_set(source.get(), "width", width.c_str()) != IMP_OK ||
            imp_video_source_set(source.get(), "height", height.c_str()) != IMP_OK)
            throw std::runtime_error("MPI source configuration failed");
        imp_video_decode_opts_t options{};
#ifdef _WIN32
        options.use_d3d11_surface_memory = true;
#else
        options.use_va_surface_memory = true;
#endif
        if (imp_video_open(&stream, source.get(), context, &options) != IMP_OK)
            throw std::runtime_error(imp_context_get_error(context));
    }
    ~DecodeStream() {
        if (stream)
            imp_video_close(stream);
        if (context)
            imp_context_destroy(context);
    }
};

inline mediapipe::CalculatorGraphConfig graphConfig(bool capi, bool split) {
    mediapipe::CalculatorGraphConfig config;
    config.add_output_stream("detections");
    const std::vector<std::string> sourceTags{"VIDEO_PATH:video_path", "WIDTH:width", "HEIGHT:height"};
    const std::vector<std::string> capiTags{"SERVER:server", "SERVABLE_NAME:servable_name",
        "SERVABLE_VERSION:servable_version", "INPUT_NAME:input_name"};
    for (const auto& tag : sourceTags)
        config.add_input_side_packet(tag);
    if (capi) {
        for (const auto& tag : capiTags)
            config.add_input_side_packet(tag);
    }
    auto* source = config.add_node();
    source->set_calculator(split ? "GstVideoSourceCalculator" : (capi ? "GstVideoCapiInferenceCalculator" : "GstVideoOpenVinoInferenceCalculator"));
    for (const auto& tag : sourceTags)
        source->add_input_side_packet(tag);
    source->add_output_stream(split ? "FRAME:frame" : "DETECTIONS:detections");
    auto* inference = source;
    if (split) {
        inference = config.add_node();
        inference->set_calculator(capi ? "GstCapiInferenceCalculator" : "GstOpenVinoInferenceCalculator");
        inference->add_input_stream("FRAME:frame");
        inference->add_output_stream("DETECTIONS:detections");
    }
    if (capi) {
        for (const auto& tag : capiTags)
            inference->add_input_side_packet(tag);
    } else {
        auto* options = inference->mutable_options()->MutableExtension(mediapipe::GstOpenVinoInferenceCalculatorOptions::ext);
        options->set_model_path(modelPath());
        options->set_input_name(benchmark_config::kInputName);
        options->set_performance_hint(benchmark_config::kPerformanceHint);
        options->set_num_streams(std::stoi(benchmark_config::kNumStreams));
    }
    return config;
}

inline void measureGraph(const std::string& name, bool capi, bool split,
    std::map<std::string, mediapipe::Packet> extraPackets = {}) {
    std::vector<double> samples;
    for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
        mediapipe::CalculatorGraph graph;
        auto status = graph.Initialize(graphConfig(capi, split));
        ASSERT_TRUE(status.ok()) << status;
        int frames = 0;
        int detections = 0;
        Clock::time_point start;
        Clock::time_point stop;
        status = graph.ObserveOutputStream("detections", [&](const mediapipe::Packet& packet) {
            const int count = packet.Get<int>();
            if (count < 0)
                return absl::InternalError("Negative detection count");
            detections += count;
            ++frames;
            if (frames == benchmark_config::kWarmupFrames)
                start = Clock::now();
            if (frames == kTotalFrames)
                stop = Clock::now();
            return absl::OkStatus();
        });
        ASSERT_TRUE(status.ok()) << status;
        std::map<std::string, mediapipe::Packet> packets = extraPackets;
        packets.emplace("video_path", mediapipe::MakePacket<std::string>(videoPath()));
        packets.emplace("width", mediapipe::MakePacket<int>(benchmark_config::kModelWidth));
        packets.emplace("height", mediapipe::MakePacket<int>(benchmark_config::kModelHeight));
        status = graph.StartRun(packets);
        ASSERT_TRUE(status.ok()) << status;
        status = graph.WaitUntilDone();
        ASSERT_TRUE(status.ok()) << status;
        ASSERT_EQ(frames, kTotalFrames) << "Source must reach EOS, not a MAX_FRAMES cutoff";
        ASSERT_GT(detections, 0);
        samples.push_back(milliseconds(start, stop) / benchmark_config::kMeasuredFrames);
    }
    report(name, samples);
}
}  // namespace inference_benchmark
#endif