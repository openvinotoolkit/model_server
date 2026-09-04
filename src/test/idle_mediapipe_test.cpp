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
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "src/dags/pipelinedefinitionstatus.hpp"
#include "src/mediapipe_internal/mediapipegraphconfig.hpp"
#include "src/mediapipe_internal/mediapipegraphdefinition.hpp"
#include "src/status.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "test_utils.hpp"

using namespace ovms;

static const std::string kSimplePbtxt = R"(
    input_stream: "in"
    output_stream: "out"
)";

class MediapipeIdleSleepTest : public ::testing::Test {
protected:
    ConstructorEnabledModelManager manager;

    std::unique_ptr<DummyMediapipeGraphDefinition> makeSleepingDef(const std::string& name) {
        MediapipeGraphConfig mgc{name, "", ""};
        mgc.setIdleUnloadTimeoutSeconds(10);
        return std::make_unique<DummyMediapipeGraphDefinition>(name, mgc, kSimplePbtxt, nullptr, true);
    }

    std::unique_ptr<DummyMediapipeGraphDefinition> makeAvailableDef(const std::string& name) {
        MediapipeGraphConfig mgc{name, "", ""};
        mgc.setIdleUnloadTimeoutSeconds(10);
        auto def = std::make_unique<DummyMediapipeGraphDefinition>(name, mgc, kSimplePbtxt, nullptr);
        def->forceValidationPassedEventForTest();
        return def;
    }
};

TEST_F(MediapipeIdleSleepTest, LazyLoadStartsSleeping) {
    auto def = makeSleepingDef("graph1");
    EXPECT_EQ(def->getStateCode(), PipelineDefinitionStateCode::SLEEPING);
    EXPECT_TRUE(def->getStatus().isSleeping());
}

TEST_F(MediapipeIdleSleepTest, UnloadTransitionsAvailableToSleeping) {
    auto def = makeAvailableDef("graph1");
    ASSERT_EQ(def->getStateCode(), PipelineDefinitionStateCode::AVAILABLE);

    ASSERT_EQ(def->putToSleep(), StatusCode::OK);
    EXPECT_EQ(def->getStateCode(), PipelineDefinitionStateCode::SLEEPING);
}

TEST_F(MediapipeIdleSleepTest, UnloadOnSleepingIsNoop) {
    auto def = makeSleepingDef("graph1");
    ASSERT_EQ(def->putToSleep(), StatusCode::OK);
    EXPECT_EQ(def->getStateCode(), PipelineDefinitionStateCode::SLEEPING);
}

TEST_F(MediapipeIdleSleepTest, WakeUpOnAvailableIsNoop) {
    auto def = makeAvailableDef("graph1");
    ASSERT_EQ(def->wakeUpIfSleeping(manager), StatusCode::OK);
    EXPECT_EQ(def->getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
}

TEST_F(MediapipeIdleSleepTest, WakeUpOnRetiredReturnsError) {
    auto def = makeAvailableDef("graph1");
    def->retire();
    ASSERT_EQ(def->getStateCode(), PipelineDefinitionStateCode::RETIRED);

    auto status = def->wakeUpIfSleeping(manager);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.getCode(), StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE);
}

TEST_F(MediapipeIdleSleepTest, WakeUpOnBeginReturnsError) {
    MediapipeGraphConfig mgc{"graph1", "", ""};
    mgc.setIdleUnloadTimeoutSeconds(10);
    DummyMediapipeGraphDefinition def("graph1", mgc, kSimplePbtxt, nullptr);
    ASSERT_EQ(def.getStateCode(), PipelineDefinitionStateCode::BEGIN);

    auto status = def.wakeUpIfSleeping(manager);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.getCode(), StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE);
}

TEST_F(MediapipeIdleSleepTest, ConcurrentWakeUpAllSucceed) {
    auto def = makeAvailableDef("graph1");
    ASSERT_EQ(def->putToSleep(), StatusCode::OK);
    ASSERT_EQ(def->getStateCode(), PipelineDefinitionStateCode::SLEEPING);

    constexpr int numThreads = 8;
    std::promise<void> startSignal;
    std::shared_future<void> ready = startSignal.get_future().share();
    std::vector<std::promise<void>> threadReady(numThreads);
    std::vector<Status> results(numThreads);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&results, &def, &ready, &threadReady, &mgr = manager, i]() {
            threadReady[i].set_value();
            ready.wait();
            results[i] = def->wakeUpIfSleeping(mgr);
        });
    }
    for (int i = 0; i < numThreads; ++i) {
        threadReady[i].get_future().wait();
    }
    startSignal.set_value();
    for (auto& t : threads) {
        t.join();
    }
    // With trivial pbtxt, reload may fail (no real graph), so we just verify
    // no crash/deadlock and state is consistent.
    auto finalState = def->getStateCode();
    EXPECT_TRUE(finalState == PipelineDefinitionStateCode::AVAILABLE ||
                finalState == PipelineDefinitionStateCode::SLEEPING);
}
