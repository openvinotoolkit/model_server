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
 * P2 graph test — runs gst_video_source.pbtxt and proves per-packet surface
 * ownership: holds several frames in flight at once and asserts their VA
 * surface IDs are distinct and their RemoteTensor planes are valid. Then drains
 * to EOS to confirm clean shutdown.
 *
 * Requires GPU + GStreamer + a test video, so it skips unless those are present.
 * Overridable via env (defaults match the container):
 *   VIDEO_PATH (default /ovms/recording_3_raw.avi)
 *   GRAPH_PATH (default /ovms/src/test/mediapipe/calculators/gst_video_source.pbtxt)
 *   WIDTH/HEIGHT (default 672x384 — face-detection-adas-0001 input)
 */

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/test/mediapipe/calculators/gst_video_frame_packet.hpp"

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

TEST(GstVideoSourceGraph, PerPacketSurfaceOwnership) {
    const std::string video_path = envOr("VIDEO_PATH", "/ovms/recording_3_raw.avi");
    const std::string graph_path = envOr(
        "GRAPH_PATH",
        "/ovms/src/test/mediapipe/calculators/gst_video_source.pbtxt");
    const int width = envInt("WIDTH", 672);
    const int height = envInt("HEIGHT", 384);

    if (!fileExists(video_path))
        GTEST_SKIP() << "video not found: " << video_path;
    if (!fileExists(graph_path))
        GTEST_SKIP() << "graph not found: " << graph_path;
    if (!imp_video_va_available())
        GTEST_SKIP() << "VA/GPU not available on this host";

    auto config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(
        readFile(graph_path));

    ::mediapipe::CalculatorGraph graph;
    ASSERT_TRUE(graph.Initialize(config).ok());

    auto poller_or = graph.AddOutputStreamPoller("frame");
    ASSERT_TRUE(poller_or.ok());
    ::mediapipe::OutputStreamPoller poller = std::move(poller_or.value());

    std::map<std::string, ::mediapipe::Packet> side_packets = {
        {"video_path", ::mediapipe::MakePacket<std::string>(video_path)},
        {"width", ::mediapipe::MakePacket<int>(width)},
        {"height", ::mediapipe::MakePacket<int>(height)},
    };
    ASSERT_TRUE(graph.StartRun(side_packets).ok());

    // Hold several frames simultaneously in flight — this is exactly what a
    // single live_va_sample slot could not do. Their surfaces must be distinct
    // and all valid at the same time.
    constexpr int kInFlight = 4;
    std::vector<::mediapipe::Packet> held;
    for (int i = 0; i < kInFlight; i++) {
        ::mediapipe::Packet packet;
        ASSERT_TRUE(poller.Next(&packet)) << "expected at least " << kInFlight << " frames";
        held.push_back(packet);
    }

    std::vector<uint32_t> surfaces;
    for (const auto& packet : held) {
        const auto& frame = packet.Get<::mediapipe::GstVideoFramePacket>();
        ASSERT_EQ(frame.planes.size(), 2u) << "NV12 must have Y and UV planes";
        EXPECT_GT(frame.planes[0].get_size(), 0u);
        EXPECT_GT(frame.planes[1].get_size(), 0u);
        EXPECT_TRUE(frame.owner != nullptr);
        // Zero-copy proof: planes are real device (remote) handles, not host
        // tensors — remote tensors expose a non-empty params map, and the frame
        // carries a live VA surface id. No host mapping happened on this path.
        EXPECT_FALSE(frame.planes[0].get_params().empty());
        EXPECT_FALSE(frame.planes[1].get_params().empty());
        EXPECT_NE(frame.va_surface_id, 0u);
        surfaces.push_back(frame.va_surface_id);
    }
    // All held surfaces must be distinct (per-packet ownership).
    for (size_t i = 0; i < surfaces.size(); i++)
        for (size_t j = i + 1; j < surfaces.size(); j++)
            EXPECT_NE(surfaces[i], surfaces[j])
                << "held frames " << i << " and " << j << " share a surface";

    // Release the held frames and drain the rest to EOS (clean shutdown).
    held.clear();
    int drained = 0;
    ::mediapipe::Packet packet;
    while (poller.Next(&packet)) {
        drained++;
        if (drained > 100000) break;  // safety cap
    }

    EXPECT_GT(drained, 0);
    ASSERT_TRUE(graph.WaitUntilDone().ok());
}

}  // namespace
