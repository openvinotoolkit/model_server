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

/**
 * P3/P4 split-path graph test — GStreamer/VA source -> NV12 GPU inference.
 *
 * Runs gst_video_infer.pbtxt end to end: the source decodes and emits owned VA
 * frames, the inference node binds the two NV12 planes on the shared context and
 * infers. Sums detections across frames and asserts at least one (parity bar
 * with the standalone/P0 proof), proving the split zero-copy path works.
 *
 * Skips unless GPU + GStreamer + assets are present. Env overrides (defaults
 * match the container):
 *   VIDEO_PATH  (default /ovms/recording_3_raw.avi)
 *   MODEL_PATH  (default /models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml)
 *   GRAPH_PATH  (default /ovms/src/test/mediapipe/calculators/gst_video_infer.pbtxt)
 *   WIDTH/HEIGHT (default 672x384)
 */

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"

namespace {

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int envInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

TEST(GstVideoInferGraph, SplitPathZeroCopyDetections) {
    const std::string video_path = envOr("VIDEO_PATH", "/ovms/recording_3_raw.avi");
    const std::string model_path = envOr(
        "MODEL_PATH",
        "/models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml");
    const std::string graph_path = envOr(
        "GRAPH_PATH",
        "/ovms/src/test/mediapipe/calculators/gst_video_infer.pbtxt");
    const int width = envInt("WIDTH", 672);
    const int height = envInt("HEIGHT", 384);

    if (!fileExists(video_path))
        GTEST_SKIP() << "video not found: " << video_path;
    if (!fileExists(model_path))
        GTEST_SKIP() << "model not found: " << model_path;
    if (!fileExists(graph_path))
        GTEST_SKIP() << "graph not found: " << graph_path;
    if (!imp_video_va_available())
        GTEST_SKIP() << "VA/GPU not available on this host";

    auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(
        readFile(graph_path));

    ::mediapipe::CalculatorGraph graph;
    ASSERT_TRUE(graph.Initialize(config).ok());

    auto poller_or = graph.AddOutputStreamPoller("detections");
    ASSERT_TRUE(poller_or.ok());
    ::mediapipe::OutputStreamPoller poller = std::move(poller_or.value());

    std::map<std::string, ::mediapipe::Packet> side_packets = {
        {"video_path", ::mediapipe::MakePacket<std::string>(video_path)},
        {"width", ::mediapipe::MakePacket<int>(width)},
        {"height", ::mediapipe::MakePacket<int>(height)},
        {"model_path", ::mediapipe::MakePacket<std::string>(model_path)},
    };
    ASSERT_TRUE(graph.StartRun(side_packets).ok());

    int frames = 0;
    int total_detections = 0;
    ::mediapipe::Packet packet;
    while (poller.Next(&packet)) {
        total_detections += packet.Get<int>();
        frames++;
    }

    EXPECT_GT(frames, 0);
    EXPECT_GE(total_detections, 1) << "expected at least one detection on the split path";
    std::cerr << "[GstVideoInferGraph] frames=" << frames
              << " total_detections=" << total_detections << "\n";

    ASSERT_TRUE(graph.WaitUntilDone().ok());
}

}  // namespace
