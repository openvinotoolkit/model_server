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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mediapipe/framework/calculator_framework.h"
#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/live_graph_config.hpp"
#include "src/mediapipe_internal/live_graph_session_impl.hpp"
#include "src/mediapipe_internal/mediapipegraphexecutor.hpp"

namespace ovms {
namespace {
class LiveSessionEchoCalculator : public mediapipe::CalculatorBase {
public:
    static absl::Status GetContract(mediapipe::CalculatorContract* contract) {
        contract->Inputs().Index(0).Set<std::string>();
        contract->Outputs().Index(0).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Process(mediapipe::CalculatorContext* context) override {
        context->Outputs().Index(0).AddPacket(context->Inputs().Index(0).Value());
        return absl::OkStatus();
    }
};
REGISTER_CALCULATOR(LiveSessionEchoCalculator);

std::shared_ptr<LiveGraphConfig> echoProfile() {
    auto profile = std::make_shared<LiveGraphConfig>();
    profile->graph.add_input_stream("text");
    profile->graph.add_output_stream("result");
    auto* node = profile->graph.add_node();
    node->set_calculator("LiveSessionEchoCalculator");
    node->add_input_stream("text");
    node->add_output_stream("result");
    profile->inputs.emplace("text", LiveGraphStreamCodec::typed<std::string>());
    profile->outputs.emplace("result", LiveGraphStreamCodec::typed<std::string>());
    return profile;
}

class LiveGraphSessionTest : public ::testing::Test {
protected:
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<LiveGraphPacket> received;
    LiveGraphSession::Observer observer() {
        return [this](const std::string& stream, const LiveGraphPacket& packet) -> Status {
            EXPECT_EQ(stream, "result");
            {
                std::lock_guard<std::mutex> lock(mutex);
                received.push_back(packet);
            }
            ready.notify_all();
            return StatusCode::OK;
        };
    }

    bool waitForPackets(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        return ready.wait_for(lock, std::chrono::seconds(10), [&] { return received.size() >= count; });
    }
};

TEST_F(LiveGraphSessionTest, ChecksTypesNamesAndClosedInputsWithoutPoisoningSession) {
    std::unique_ptr<LiveGraphSession> session;
    ASSERT_TRUE(createLiveGraphSession(echoProfile(), {}, nullptr, observer(), session).ok());
    EXPECT_FALSE(session->send("missing", LiveGraphPacket::make(std::string("wrong stream"))).ok());
    EXPECT_FALSE(session->send("text", LiveGraphPacket::make(42)).ok());
    EXPECT_FALSE(session->closeInput("missing").ok());
    ASSERT_TRUE(session->send("text", LiveGraphPacket::make(std::string("hello"))).ok());
    ASSERT_TRUE(waitForPackets(1));
    ASSERT_TRUE(session->closeInput("text").ok());
    EXPECT_FALSE(session->closeInput("text").ok());
    EXPECT_FALSE(session->send("text", LiveGraphPacket::make(std::string("closed"))).ok());
    session.reset();
    ASSERT_EQ(received.size(), 1);
    ASSERT_NE(received[0].getIf<std::string>(), nullptr);
    EXPECT_EQ(*received[0].getIf<std::string>(), "hello");
    EXPECT_EQ(received[0].getIf<int>(), nullptr);
}

TEST_F(LiveGraphSessionTest, SessionsRetainLeaseAndRunIndependentlyOfExecutor) {
    GraphSidePackets resources;
    resources.liveGraphProfiles.emplace("text.echo", echoProfile());
    auto count = std::make_shared<std::atomic<int64_t>>(0);
    auto activity = std::make_shared<std::atomic<int64_t>>(0);
    auto executor = std::make_unique<MediapipeGraphExecutor>("echo", "1",
        mediapipe::CalculatorGraphConfig{}, stream_types_mapping_t{}, stream_types_mapping_t{},
        std::vector<std::string>{}, std::vector<std::string>{},
        resources, nullptr, nullptr, count, activity);
    EXPECT_TRUE(executor->supportsSession("text.echo"));
    EXPECT_FALSE(executor->supportsSession("missing"));
    std::unique_ptr<LiveGraphSession> first;
    std::unique_ptr<LiveGraphSession> second;
    EXPECT_FALSE(executor->startSession("missing", observer(), first).ok());
    ASSERT_TRUE(executor->startSession("text.echo", observer(), first).ok());
    EXPECT_FALSE(executor->startSession("text.echo", observer(), first).ok());
    ASSERT_TRUE(executor->startSession("text.echo", observer(), second).ok());
    executor.reset();
    EXPECT_EQ(count->load(), 1);
    ASSERT_TRUE(first->send("text", LiveGraphPacket::make(std::string("first"))).ok());
    ASSERT_TRUE(waitForPackets(1));
    first->cancel();
    EXPECT_FALSE(first->send("text", LiveGraphPacket::make(std::string("cancelled"))).ok());
    EXPECT_FALSE(first->closeInput("text").ok());
    first.reset();
    EXPECT_EQ(count->load(), 1);
    ASSERT_TRUE(second->send("text", LiveGraphPacket::make(std::string("second"))).ok());
    ASSERT_TRUE(waitForPackets(2));
    second.reset();
    EXPECT_EQ(count->load(), 0);
    EXPECT_GT(activity->load(), 0);
    ASSERT_EQ(received.size(), 2);
    EXPECT_EQ(*received[0].getIf<std::string>(), "first");
    EXPECT_EQ(*received[1].getIf<std::string>(), "second");
}

TEST_F(LiveGraphSessionTest, RejectsInvalidProfilesAndObservers) {
    std::unique_ptr<LiveGraphSession> session;
    EXPECT_FALSE(createLiveGraphSession(nullptr, {}, nullptr, observer(), session).ok());
    EXPECT_FALSE(createLiveGraphSession(echoProfile(), {}, nullptr, {}, session).ok());
    auto profile = echoProfile();
    profile->inputs.clear();
    EXPECT_FALSE(createLiveGraphSession(profile, {}, nullptr, observer(), session).ok());
    profile = echoProfile();
    profile->outputs.clear();
    EXPECT_FALSE(createLiveGraphSession(profile, {}, nullptr, observer(), session).ok());
    profile = echoProfile();
    profile->inputs.at("text").encode = {};
    EXPECT_FALSE(createLiveGraphSession(profile, {}, nullptr, observer(), session).ok());
    profile = echoProfile();
    profile->outputs.at("result").observe = {};
    EXPECT_FALSE(createLiveGraphSession(profile, {}, nullptr, observer(), session).ok());
    EXPECT_FALSE(session);
}

TEST_F(LiveGraphSessionTest, SurfacesObserverFailure) {
    for (bool throws : {false, true}) {
        std::atomic<bool> called{false};
        std::unique_ptr<LiveGraphSession> session;
        ASSERT_TRUE(createLiveGraphSession(echoProfile(), {}, nullptr,
            [&called, throws](const std::string&, const LiveGraphPacket&) -> Status {
                called = true;
                if (throws)
                    throw std::runtime_error("observer failure");
                return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "observer failure");
            }, session).ok());
        ASSERT_TRUE(session->send("text", LiveGraphPacket::make(std::string("hello"))).ok());
        Status status = StatusCode::OK;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (status.ok() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            status = session->send("text", LiveGraphPacket::make(std::string("check error")));
        }
        EXPECT_TRUE(called);
        EXPECT_FALSE(status.ok());
        EXPECT_NE(status.string().find("observer failure"), std::string::npos);
    }
}
}  // namespace
}  // namespace ovms
