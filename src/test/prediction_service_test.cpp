//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <stdlib.h>

#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "test_utils.hpp"

using testing::Each;
using testing::Eq;

class ConstructorEnabledModelManager : public ovms::ModelManager {
public:
    ConstructorEnabledModelManager() :
        ModelManager() {}
    ~ConstructorEnabledModelManager() {
        SPDLOG_INFO("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        models.clear();
        SPDLOG_INFO("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    }
};

class TestPredict : public ::testing::Test {
public:
    void SetUp() {
        ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
        const int initialBatchSize = 1;
        config.setBatchSize(initialBatchSize);
        // TODO dirty hack to avoid initializing config
        setenv("NIREQ", "2", 1);
    }
    void TearDown() {
        SPDLOG_ERROR("TEAR_DOWN");
    }
    /**
     * @brief This function should mimic most closely predict request to check for thread safety
     */
    void performPredict(const std::string modelName,
        const ovms::model_version_t modelVersion,
        const size_t batchSize,
        std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance = nullptr,
        std::unique_ptr<std::future<void>> waitBeforePerformInference = nullptr);

    void deserialize(const std::vector<float>& input, InferenceEngine::InferRequest& inferRequest, std::shared_ptr<ovms::ModelInstance> modelInstance) {
        auto blob = InferenceEngine::make_shared_blob<float>(
            modelInstance->getInputsInfo().at(DUMMY_MODEL_INPUT_NAME)->getTensorDesc(),
            const_cast<float*>(reinterpret_cast<const float*>(input.data())));
        inferRequest.SetBlob(DUMMY_MODEL_INPUT_NAME, blob);
    }

    void serializeAndCheck(int outputSize, InferenceEngine::InferRequest& inferRequest) {
        std::vector<float> output(outputSize);
        ASSERT_THAT(output, Each(Eq(0.)));
        auto blobOutput = inferRequest.GetBlob(DUMMY_MODEL_OUTPUT_NAME);
        std::memcpy(output.data(), blobOutput->cbuffer(), outputSize * sizeof(float));
        EXPECT_THAT(output, Each(Eq(2.)));
    }

    void testConcurrentPredicts(const int initialBatchSize, const uint waitingBeforePerformInferenceCount, const uint waitingBeforeGettingModelCount) {
        ASSERT_GE(20, waitingBeforePerformInferenceCount);
        // TODO dirty hack to avoid initializing config
        setenv("NIREQ", "20", 1);
        ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

        std::vector<std::promise<void>> releaseWaitBeforeGettingModelInstance(waitingBeforeGettingModelCount);
        std::vector<std::promise<void>> releaseWaitBeforePerformInference(waitingBeforePerformInferenceCount);

        std::vector<std::thread> predictsWaitingBeforeGettingModelInstance;
        std::vector<std::thread> predictsWaitingBeforeInference;
        for (auto i = 0u; i < waitingBeforeGettingModelCount; ++i) {
            predictsWaitingBeforeGettingModelInstance.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforeGettingModelInstance, i]() {
                        performPredict(config.getName(), config.getVersion(), initialBatchSize + (i % 3),
                            std::move(
                                std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance[i].get_future())));
                    }));
        }
        for (auto i = 0u; i < waitingBeforePerformInferenceCount; ++i) {
            predictsWaitingBeforeInference.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforePerformInference, i]() {
                        performPredict(config.getName(), config.getVersion(), initialBatchSize,
                            nullptr,
                            std::move(
                                std::make_unique<std::future<void>>(releaseWaitBeforePerformInference[i].get_future())));
                    }));
        }
        // sleep to allow all threads to initialize
        std::this_thread::sleep_for(std::chrono::seconds(2));
        for (auto& promise : releaseWaitBeforeGettingModelInstance) {
            promise.set_value();
        }
        for (auto& promise : releaseWaitBeforePerformInference) {
            promise.set_value();
        }
        for (auto& thread : predictsWaitingBeforeGettingModelInstance) {
            thread.join();
        }
        for (auto& thread : predictsWaitingBeforeInference) {
            thread.join();
        }
    }

public:
    ConstructorEnabledModelManager manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ~TestPredict() {
        std::cout << "Destructor of TestPredict()" << std::endl;
    }
};

void TestPredict::performPredict(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const size_t batchSize,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference) {
    // only validation is skipped
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstancePredictRequestsHandlesCountGuard> modelInstancePredictRequestsHandlesCountGuard;

    if (waitBeforeGettingModelInstance) {
        std::cout << "Waiting before getModelInstance. Batch size:" << batchSize << std::endl;
        waitBeforeGettingModelInstance->get();
    }
    ASSERT_EQ(getModelInstance(manager, modelName, modelVersion, modelInstance, modelInstancePredictRequestsHandlesCountGuard), ovms::StatusCode::OK);
    ASSERT_EQ(assureModelInstanceLoadedWithProperBatchSize(*modelInstance, batchSize, modelInstancePredictRequestsHandlesCountGuard), ovms::StatusCode::OK);
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance->getInferRequestsQueue();
    ovms::ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    std::vector<float> input(DUMMY_MODEL_INPUT_SIZE * batchSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));
    deserialize(input, inferRequest, modelInstance);
    if (waitBeforePerformInference) {
        std::cout << "Waiting before performInfernce. Batch size:" << batchSize << std::endl;
        waitBeforePerformInference->get();
    }
    auto status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    serializeAndCheck(DUMMY_MODEL_OUTPUT_SIZE * batchSize, inferRequest);
}

TEST_F(TestPredict, SuccesfullOnDummyModel) {
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    performPredict(config.getName(), config.getVersion(), 1);
}

TEST_F(TestPredict, SuccesfullReloadFromAlreadyLoadedWithNewBatchSize) {
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const int initialBatchSize = config.getBatchSize();
    config.setBatchSize(initialBatchSize);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    performPredict(config.getName(), config.getVersion(), initialBatchSize);

    auto newBatchSize = config.getBatchSize() + 1;
    ASSERT_NE(initialBatchSize, newBatchSize);
    performPredict(config.getName(), config.getVersion(), newBatchSize);
}

TEST_F(TestPredict, SuccesfullReloadWhen1InferRequestJustBeforePredict) {
    // FIRST LOAD MODEL WITH BS=1
    const int initialBatchSize = 1;
    const int newBatchSize = 2;
    config.setBatchSize(initialBatchSize);
    // TODO dirty hack to avoid initializing config
    setenv("NIREQ", "2", 1);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    std::promise<void> releaseWaitBeforePerformInference;
    std::thread t(
        [this, &releaseWaitBeforePerformInference]() {
            performPredict(config.getName(), config.getVersion(), initialBatchSize,
                nullptr,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInference.get_future())));
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstancePredictRequestsHandlesCountGuard> modelInstancePredictRequestsHandlesCountGuard1;
    ASSERT_EQ(getModelInstance(manager, config.getName(), config.getVersion(), modelInstance, modelInstancePredictRequestsHandlesCountGuard1), ovms::StatusCode::OK);

    std::thread t2([modelInstance, newBatchSize, &modelInstancePredictRequestsHandlesCountGuard1]() { modelInstance->reloadModel(newBatchSize, modelInstancePredictRequestsHandlesCountGuard1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    releaseWaitBeforePerformInference.set_value();
    t2.join();

    ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance->getInferRequestsQueue();
    ovms::ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);

    std::vector<float> input(DUMMY_MODEL_INPUT_SIZE * newBatchSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));
    deserialize(input, inferRequest, modelInstance);
    ASSERT_EQ(performInference(inferRequestsQueue, executingInferId, inferRequest), ovms::StatusCode::OK);
    serializeAndCheck(DUMMY_MODEL_OUTPUT_SIZE, inferRequest);
    t.join();
}

TEST_F(TestPredict, SuccesfullReloadWhen1InferRequestJustBeforeGettingModelInstance) {
    // FIRST LOAD MODEL WITH BS=1
    // request inference with BS=2
    // in meantime start predict request with BS1
    const int initialBatchSize = 1;
    const int newBatchSize = 2;
    config.setBatchSize(initialBatchSize);
    // TODO dirty hack to avoid initializing config
    setenv("NIREQ", "2", 1);
    auto status = manager.reloadModelWithVersions(config);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    std::promise<void> releaseWaitBeforeGettingModelInstance;
    std::thread secondPredictRequest(
        [this, &releaseWaitBeforeGettingModelInstance]() {
            performPredict(config.getName(), config.getVersion(), initialBatchSize,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance.get_future())));
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstancePredictRequestsHandlesCountGuard> modelInstancePredictRequestsHandlesCountGuard;
    ASSERT_EQ(getModelInstance(manager, config.getName(), config.getVersion(), modelInstance, modelInstancePredictRequestsHandlesCountGuard),
        ovms::StatusCode::OK);
    std::thread assureProperBSLoadedThread([modelInstance, newBatchSize, &modelInstancePredictRequestsHandlesCountGuard]() {
        auto status = assureModelInstanceLoadedWithProperBatchSize(*modelInstance, newBatchSize, modelInstancePredictRequestsHandlesCountGuard);
        ASSERT_EQ(status, ovms::StatusCode::OK);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Now notifying second predict to get model with proper batchsize." << std::endl;
    releaseWaitBeforeGettingModelInstance.set_value();
    assureProperBSLoadedThread.join();

    ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance->getInferRequestsQueue();
    // exception from keeping the same as predict path - using unique_ptr to keep order of destructors the same
    auto executingStreamIdGuard = std::make_unique<ovms::ExecutingStreamIdGuard>(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard->getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);

    std::vector<float> input(DUMMY_MODEL_INPUT_SIZE * newBatchSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));

    deserialize(input, inferRequest, modelInstance);
    ASSERT_EQ(performInference(inferRequestsQueue, executingInferId, inferRequest), ovms::StatusCode::OK);
    serializeAndCheck(DUMMY_MODEL_OUTPUT_SIZE * newBatchSize, inferRequest);

    std::cout << "Now releasing blockade from reloading." << std::endl;
    executingStreamIdGuard.reset();
    modelInstancePredictRequestsHandlesCountGuard.reset();
    secondPredictRequest.join();
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeGettingModelInstance) {
    const int initialBatchSize = 1;
    config.setBatchSize(initialBatchSize);

    const uint waitingBeforePerformInferenceCount = 0;
    const uint waitingBeforeGettingModelCount = 9;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeInference) {
    const int initialBatchSize = 1;
    config.setBatchSize(initialBatchSize);

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 0;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestAtDifferentStages) {
    const int initialBatchSize = 1;
    config.setBatchSize(initialBatchSize);

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 9;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}
