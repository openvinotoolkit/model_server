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
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "src/model.hpp"
#include "src/modelinstance.hpp"
#include "src/modelversionstatus.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "environment.hpp"
#include "platform_utils.hpp"
#include "test_utils.hpp"
#include "test_models.hpp"
#include "test_models_configs.hpp"
#include "test_with_temp_dir.hpp"

using namespace ovms;

static const std::string idleModelConfig = R"({
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": ")" + dummy_model_location +
                                           R"(",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ]
})";

class IdleModelManagementTest : public TestWithTempDir {
protected:
    std::string configFilePath;

    void writeConfig(const std::string& content) {
        configFilePath = directoryPath + "/config.json";
        std::ofstream ofs(configFilePath);
        ofs << content;
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        writeConfig(idleModelConfig);
    }
};

TEST_F(IdleModelManagementTest, NonPermanentModelStartsAsSleepingButAppearsAvailable) {
    ConstructorEnabledModelManager manager(30'000'000);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok()) << status.string();
    auto model = manager.findModelByName("dummy");
    ASSERT_NE(model, nullptr);
    auto instance = model->getDefaultModelInstance();
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->getStatus().getState(), ModelVersionState::SLEEPING);

    auto availableNames = manager.getNamesOfAvailableModels();
    EXPECT_NE(std::find(availableNames.begin(), availableNames.end(), "dummy"),
        availableNames.end());
}

TEST_F(IdleModelManagementTest, PermanentGroupModelIsFullyLoaded) {
    std::string permanentConfig = R"({
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": ")" +
                                  dummy_model_location + R"(",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "group_name": "permanent"
                }
            }
        ]
    })";
    writeConfig(permanentConfig);

    ConstructorEnabledModelManager manager(30'000'000);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok()) << status.string();
    auto model = manager.findModelByName("dummy");
    ASSERT_NE(model, nullptr);
    auto instance = model->getDefaultModelInstance();
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(IdleModelManagementTest, SleepingModelSelectedAsDefaultVersion) {
    ConstructorEnabledModelManager manager(30'000'000);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok()) << status.string();

    auto model = manager.findModelByName("dummy");
    ASSERT_NE(model, nullptr);
    auto instance = model->getDefaultModelInstance();
    ASSERT_NE(instance, nullptr);
    EXPECT_TRUE(instance->getStatus().appearsAvailable());
    EXPECT_TRUE(instance->getStatus().isSleeping());
}

class ModelInstanceSleepTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
    }
};

TEST_F(ModelInstanceSleepTest, LazyLoadThenWakeUp) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);

    auto status = instance.wakeUpIfSleeping();
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(ModelInstanceSleepTest, WakeUpThenPutToSleep) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    ASSERT_TRUE(instance.wakeUpIfSleeping().ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);

    instance.putToSleep();
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);
}

TEST_F(ModelInstanceSleepTest, WakeUpWithInvalidPathFails) {
    const std::string nonexistentPath = getGenericFullPathForTmp("/tmp/idle_model_test_nonexistent_path");
    ModelConfig badConfig = DUMMY_MODEL_CONFIG;
    badConfig.setBasePath(nonexistentPath);
    badConfig.setLocalPath(nonexistentPath);

    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(badConfig, true), StatusCode::OK);
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);

    auto status = instance.wakeUpIfSleeping();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);

    // Failed wake-up should remain retryable on every next request.
    status = instance.wakeUpIfSleeping();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);
}

TEST_F(ModelInstanceSleepTest, WakeUpIfAlreadyAvailableIsNoop) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    ASSERT_TRUE(instance.wakeUpIfSleeping().ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);

    ASSERT_TRUE(instance.wakeUpIfSleeping().ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(ModelInstanceSleepTest, WakeUpReportsLoadingStatusWhileReloadInProgress) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_ALL_IDLE("wakeUpIfSleeping() never sets LOADING before reloading");
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    ASSERT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);

    std::atomic<bool> sawLoading{false};
    std::atomic<bool> done{false};
    std::promise<void> pollerStarted;
    std::thread poller([&instance, &sawLoading, &done, &pollerStarted]() {
        pollerStarted.set_value();
        while (!done.load(std::memory_order_relaxed)) {
            if (instance.getStatus().getState() == ModelVersionState::LOADING) {
                sawLoading.store(true, std::memory_order_relaxed);
                break;
            }
        }
    });
    pollerStarted.get_future().wait();
    auto status = instance.wakeUpIfSleeping();
    done.store(true, std::memory_order_relaxed);
    poller.join();

    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(sawLoading.load(std::memory_order_relaxed));
}

TEST_F(ModelInstanceSleepTest, WakeUpOnRetiredModelReturnsError) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    instance.retireModel();
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::END);

    auto status = instance.wakeUpIfSleeping();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.getCode(), StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

TEST_F(ModelInstanceSleepTest, ConcurrentWakeUpAllSucceed) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);

    constexpr int numThreads = 20;
    std::promise<void> startSignal;
    std::shared_future<void> ready = startSignal.get_future().share();
    std::vector<std::promise<void>> threadReady(numThreads);
    std::vector<Status> results(numThreads);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&results, &instance, &ready, &threadReady, i]() {
            threadReady[i].set_value();
            ready.wait();
            results[i] = instance.wakeUpIfSleeping();
        });
    }
    for (int i = 0; i < numThreads; ++i) {
        threadReady[i].get_future().wait();
    }
    startSignal.set_value();
    for (auto& t : threads) {
        t.join();
    }
    for (int i = 0; i < numThreads; ++i) {
        EXPECT_TRUE(results[i].ok()) << "Thread " << i << " failed: " << results[i].string();
    }
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(ModelInstanceSleepTest, RetireThenWakeUpReturnsError) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    ASSERT_TRUE(instance.wakeUpIfSleeping().ok());
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::AVAILABLE);

    instance.retireModel();
    EXPECT_EQ(instance.getStatus().getState(), ModelVersionState::END);

    auto status = instance.wakeUpIfSleeping();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.getCode(), StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

TEST_F(ModelInstanceSleepTest, ConcurrentWakeUpAndPutToSleep) {
    ModelInstance instance("dummy", 1, *ieCore);
    ASSERT_EQ(instance.loadModel(DUMMY_MODEL_CONFIG, true), StatusCode::OK);
    ASSERT_EQ(instance.getStatus().getState(), ModelVersionState::SLEEPING);

    constexpr int numWakers = 20;
    std::promise<void> startSignal;
    std::shared_future<void> ready = startSignal.get_future().share();
    std::vector<std::promise<void>> threadReady(numWakers + 1);
    std::vector<Status> wakeResults(numWakers);
    std::vector<std::thread> threads;
    threads.reserve(numWakers + 1);

    for (int i = 0; i < numWakers; ++i) {
        threads.emplace_back([&wakeResults, &instance, &ready, &threadReady, i]() {
            threadReady[i].set_value();
            ready.wait();
            wakeResults[i] = instance.wakeUpIfSleeping();
        });
    }
    threads.emplace_back([&instance, &ready, &threadReady, numWakers]() {
        threadReady[numWakers].set_value();
        ready.wait();
        instance.putToSleep();
    });

    for (int i = 0; i <= numWakers; ++i) {
        threadReady[i].get_future().wait();
    }
    startSignal.set_value();
    for (auto& t : threads) {
        t.join();
    }

    auto finalState = instance.getStatus().getState();
    EXPECT_TRUE(finalState == ModelVersionState::AVAILABLE ||
                finalState == ModelVersionState::SLEEPING);
}
