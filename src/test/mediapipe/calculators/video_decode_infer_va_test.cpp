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
 * P0 graph test — runs video_decode_infer_va.pbtxt end to end.
 *
 * Drives the single-node graph with one TICK packet, feeds VIDEO_PATH/MODEL_PATH/
 * DEVICE side packets, and asserts at least one detection comes back (same bar as
 * the standalone proof).
 *
 * Requires GPU + GStreamer + test assets, so it is skipped unless those are
 * present. Inputs are overridable via env vars (defaults match the container):
 *   VIDEO_PATH  (default /ovms/recording_3_raw.avi)
 *   MODEL_PATH  (default /models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml)
 *   DEVICE      (default GPU)
 *   GRAPH_PATH  (default /ovms/src/test/mediapipe/calculators/video_decode_infer_va.pbtxt)
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

TEST(VideoDecodeInferVaGraph, RunsAndProducesDetections) {
    const std::string video_path = envOr("VIDEO_PATH", "/ovms/recording_3_raw.avi");
    const std::string model_path = envOr(
        "MODEL_PATH",
        "/models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml");
    const std::string device = envOr("DEVICE", "GPU");
    const std::string graph_path = envOr(
        "GRAPH_PATH",
        "/ovms/src/test/mediapipe/calculators/video_decode_infer_va.pbtxt");

    if (!fileExists(video_path))
        GTEST_SKIP() << "video not found: " << video_path;
    if (!fileExists(model_path))
        GTEST_SKIP() << "model not found: " << model_path;
    if (!fileExists(graph_path))
        GTEST_SKIP() << "graph not found: " << graph_path;
    if (device == "GPU" && !imp_video_va_available())
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
        {"model_path", ::mediapipe::MakePacket<std::string>(model_path)},
        {"device", ::mediapipe::MakePacket<std::string>(device)},
    };
    ASSERT_TRUE(graph.StartRun(side_packets).ok());

    ASSERT_TRUE(graph.AddPacketToInputStream(
                         "tick",
                         ::mediapipe::MakePacket<int>(0).At(::mediapipe::Timestamp(0)))
                    .ok());
    ASSERT_TRUE(graph.CloseInputStream("tick").ok());

    ::mediapipe::Packet packet;
    ASSERT_TRUE(poller.Next(&packet));
    int detections = packet.Get<int>();
    EXPECT_GE(detections, 1) << "expected at least one detection";

    ASSERT_TRUE(graph.WaitUntilDone().ok());
}

}  // namespace
