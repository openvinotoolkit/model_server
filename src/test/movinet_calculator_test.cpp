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
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "mediapipe/framework/formats/image_frame.h"
#include "src/graph_metadata.hpp"
#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/live_graph_config.hpp"
#include "src/mediapipe_internal/mediapipegraphexecutor.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/movinet/classification.hpp"
#include "src/movinet/movinet_calculator.pb.h"

namespace ovms {
namespace {
constexpr std::array<size_t, 11> TemporalDepths{2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4};
constexpr size_t PoolInputStart = 1 + std::accumulate(TemporalDepths.begin(), TemporalDepths.end(), size_t{0});
constexpr size_t PoolCount = 16;
constexpr size_t PoolScaleInput = PoolInputStart + PoolCount;
constexpr size_t InputCount = PoolScaleInput + 2;
constexpr size_t PoolingWindowFrames = 32;
constexpr float TemporalWeight = 0.1f;
constexpr float IdleLogit = 0.5f;

class MovinetCalculatorTest : public ::testing::Test {
protected:
    std::filesystem::path directory;
    mediapipe::CalculatorGraphConfig config;
    GraphSidePackets resources;

    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    ("ovms-movinet-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
        std::ofstream(directory / "labels.txt") << "active\nidle\n";

        namespace op = ov::opset13;
        ov::ParameterVector parameters;
        parameters.push_back(std::make_shared<op::Parameter>(ov::element::f32, ov::Shape{1, 3, 172, 172}));
        for (size_t index = 1; index < InputCount; ++index)
            parameters.push_back(std::make_shared<op::Parameter>(ov::element::f32, ov::Shape{1, 1}));
        const auto axes = op::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
        const auto mean = std::make_shared<op::ReduceMean>(parameters[0], axes);
        const auto shape = op::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        const auto current = std::make_shared<op::Reshape>(mean, shape, false);
        const auto sum = std::make_shared<op::Add>(parameters[PoolInputStart], current);
        const auto average = std::make_shared<op::Multiply>(sum, parameters[PoolScaleInput]);
        const auto temporal = std::make_shared<op::Multiply>(
            parameters[1], op::Constant::create(ov::element::f32, ov::Shape{1, 1}, {TemporalWeight}));
        const auto active = std::make_shared<op::Add>(average, temporal);
        const auto logits = std::make_shared<op::Concat>(ov::OutputVector{
            active, op::Constant::create(ov::element::f32, ov::Shape{1, 1}, {IdleLogit})}, 1);
        ov::ResultVector outputs{std::make_shared<op::Result>(logits)};
        for (size_t index = 0; index < TemporalDepths.size() + PoolCount; ++index)
            outputs.push_back(std::make_shared<op::Result>(current));
        ov::serialize(std::make_shared<ov::Model>(outputs, parameters), (directory / "model.xml").string());

        config.add_input_stream("source");
        config.add_output_stream("result");
        config.add_input_side_packet("movinet_resources");
        auto* node = config.add_node();
        node->set_name("movinet");
        node->set_calculator("MovinetCalculator");
        node->add_input_stream("IMAGE:source");
        node->add_output_stream("CLASSIFICATION:result");
        node->add_input_side_packet("MOVINET_RESOURCES:movinet_resources");
        mediapipe::MovinetCalculatorOptions options;
        options.set_model_path("model.xml");
        options.set_labels_path("labels.txt");
        node->add_node_options()->PackFrom(options);
        GraphMetadata metadata;
        bool matched = false;
        for (const auto& initializer : NodeInitializerRegistry::instance().all()) {
            if (initializer->matches(node->calculator())) {
                matched = true;
                const auto status = initializer->initialize(*node, "movinet", directory.string(), resources, metadata, nullptr, nullptr);
                ASSERT_TRUE(status.ok()) << status.string();
            }
        }
        ASSERT_TRUE(matched);
        for (const auto& initializer : NodeInitializerRegistry::instance().all()) {
            const auto status = initializer->prepareLiveGraph(config, resources);
            ASSERT_TRUE(status.ok()) << status.string();
        }
    }

    void TearDown() override {
        resources.clear();
        for (const char* file : {"model.xml", "model.bin", "labels.txt"})
            std::filesystem::remove(directory / file);
        std::filesystem::remove(directory);
    }

    std::unique_ptr<MediapipeGraphExecutor> executor() {
        return std::make_unique<MediapipeGraphExecutor>("movinet", "1", config,
            stream_types_mapping_t{}, stream_types_mapping_t{},
            std::vector<std::string>{}, std::vector<std::string>{}, resources, nullptr, nullptr);
    }

    struct Predictions {
        std::mutex mutex;
        std::condition_variable ready;
        std::vector<VideoClassification> values;

        LiveGraphSession::Observer observer() {
            return [this](const std::string& stream, const LiveGraphPacket& packet) -> Status {
                const auto* prediction = packet.getIf<VideoClassification>();
                if (stream != "classification" || !prediction)
                    return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unexpected classification payload");
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    values.push_back(*prediction);
                }
                ready.notify_all();
                return StatusCode::OK;
            };
        }

        VideoClassification at(size_t index) {
            std::unique_lock<std::mutex> lock(mutex);
            const bool received = ready.wait_for(lock, std::chrono::seconds(10), [&] { return values.size() > index; });
            EXPECT_TRUE(received);
            return received ? values[index] : VideoClassification{"timeout", 0};
        }
    };

    Status send(LiveGraphSession& session, uint8_t value) {
        auto image = std::make_shared<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, 8, 8);
        std::memset(image->MutablePixelData(), value, image->PixelDataSize());
        return session.send("video", LiveGraphPacket::make(std::move(image)));
    }
};

TEST_F(MovinetCalculatorTest, RollsPoolingAndRetainsTemporalHistoryAcrossFrames) {
    auto graphExecutor = executor();
    ASSERT_TRUE(graphExecutor->supportsSession(VideoClassificationLiveGraphProfile));
    Predictions predictions;
    std::unique_ptr<LiveGraphSession> session;
    ASSERT_TRUE(graphExecutor->startSession(VideoClassificationLiveGraphProfile, predictions.observer(), session).ok());
    graphExecutor.reset();
    resources.clear();

    std::deque<float> recent;
    std::deque<float> temporal(TemporalDepths.front(), 0.0f);
    for (size_t index = 0; index < PoolingWindowFrames * 2 + 8; ++index) {
        const bool white = index < PoolingWindowFrames;
        const float current = white ? 1.0f : 0.0f;
        recent.push_back(current);
        if (recent.size() > PoolingWindowFrames)
            recent.pop_front();
        const float mean = std::accumulate(recent.begin(), recent.end(), 0.0f) / recent.size();
        const float difference = mean + TemporalWeight * temporal.front() - IdleLogit;
        temporal.pop_front();
        temporal.push_back(current);
        ASSERT_TRUE(send(*session, white ? 255 : 0).ok());
        const auto prediction = predictions.at(index);
        EXPECT_EQ(prediction.label, difference >= 0 ? "active" : "idle") << index;
        EXPECT_NEAR(prediction.score, 1.0f / (1.0f + std::exp(-std::abs(difference))), 1e-5f) << index;
    }
}

TEST_F(MovinetCalculatorTest, SessionsHaveIndependentStateAndRestartFromZero) {
    auto graphExecutor = executor();
    Predictions firstResults;
    Predictions secondResults;
    Predictions restartedResults;
    std::unique_ptr<LiveGraphSession> first;
    std::unique_ptr<LiveGraphSession> second;
    ASSERT_TRUE(graphExecutor->startSession(VideoClassificationLiveGraphProfile, firstResults.observer(), first).ok());
    ASSERT_TRUE(graphExecutor->startSession(VideoClassificationLiveGraphProfile, secondResults.observer(), second).ok());
    for (size_t index = 0; index <= TemporalDepths.front(); ++index) {
        ASSERT_TRUE(send(*first, 255).ok());
        EXPECT_EQ(firstResults.at(index).label, "active");
        ASSERT_TRUE(send(*second, 0).ok());
        EXPECT_EQ(secondResults.at(index).label, "idle");
    }
    first->cancel();
    first.reset();
    ASSERT_TRUE(graphExecutor->startSession(VideoClassificationLiveGraphProfile, restartedResults.observer(), first).ok());
    ASSERT_TRUE(send(*first, 0).ok());
    const auto restarted = restartedResults.at(0);
    EXPECT_EQ(restarted.label, "idle");
    EXPECT_FLOAT_EQ(restarted.score, secondResults.at(0).score);
    ASSERT_TRUE(send(*second, 0).ok());
    EXPECT_EQ(secondResults.at(TemporalDepths.front() + 1).label, "idle");
}

TEST_F(MovinetCalculatorTest, RejectsInvalidResourcesBeforeOpeningSessions) {
    auto node = config.node(0);
    mediapipe::MovinetCalculatorOptions options;
    options.set_model_path("model.xml");
    options.set_labels_path("missing-labels.txt");
    node.mutable_node_options(0)->PackFrom(options);
    for (const auto& initializer : NodeInitializerRegistry::instance().all()) {
        if (!initializer->matches(node.calculator()))
            continue;
        GraphSidePackets invalid;
        GraphMetadata metadata;
        const auto status = initializer->initialize(node, "invalid", directory.string(), invalid, metadata, nullptr, nullptr);
        EXPECT_FALSE(status.ok());
        EXPECT_TRUE(invalid.movinetResourcesMap.empty());
        EXPECT_FALSE(initializer->prepareLiveGraph(config, invalid).ok());
    }
}
}  // namespace
}  // namespace ovms
