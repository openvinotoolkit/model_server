//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include "../executingstreamidguard.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../processing_spec.hpp"
#include "test_utils.hpp"

using testing::Each;
using testing::Eq;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
class TestPredict : public ::testing::Test {
public:
    void SetUp() {
        ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
        const int initialBatchSize = 1;
        config.setBatchSize(initialBatchSize);
        config.setNireq(2);
    }
    void TearDown() {
        spdlog::error("TEAR_DOWN");
    }
    /**
     * @brief This function should mimic most closely predict request to check for thread safety
     */
    void performPredict(const std::string modelName,
        const ovms::model_version_t modelVersion,
        const tensorflow::serving::PredictRequest& request,
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
        ASSERT_EQ(blobOutput->byteSize(), outputSize * sizeof(float));
        std::memcpy(output.data(), blobOutput->cbuffer(), outputSize * sizeof(float));
        EXPECT_THAT(output, Each(Eq(2.)));
    }

    void testConcurrentPredicts(const int initialBatchSize, const uint waitingBeforePerformInferenceCount, const uint waitingBeforeGettingModelCount) {
        ASSERT_GE(20, waitingBeforePerformInferenceCount);
        config.setNireq(20);
        ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

        std::vector<std::promise<void>> releaseWaitBeforeGettingModelInstance(waitingBeforeGettingModelCount);
        std::vector<std::promise<void>> releaseWaitBeforePerformInference(waitingBeforePerformInferenceCount);

        std::vector<std::thread> predictsWaitingBeforeGettingModelInstance;
        std::vector<std::thread> predictsWaitingBeforeInference;
        for (auto i = 0u; i < waitingBeforeGettingModelCount; ++i) {
            predictsWaitingBeforeGettingModelInstance.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforeGettingModelInstance, i]() {
                        tensorflow::serving::PredictRequest request = preparePredictRequest(
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::shape_t, tensorflow::DataType>{{(initialBatchSize + (i % 3)), 10}, tensorflow::DataType::DT_FLOAT}}});

                        performPredict(config.getName(), config.getVersion(), request,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance[i].get_future())));
                    }));
        }
        for (auto i = 0u; i < waitingBeforePerformInferenceCount; ++i) {
            predictsWaitingBeforeInference.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforePerformInference, i]() {
                        tensorflow::serving::PredictRequest request = preparePredictRequest(
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::shape_t, tensorflow::DataType>{{initialBatchSize, 10}, tensorflow::DataType::DT_FLOAT}}});

                        performPredict(config.getName(), config.getVersion(), request, nullptr,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInference[i].get_future())));
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

    void testConcurrentBsChanges(const int initialBatchSize, const uint numberOfThreads) {
        ASSERT_GE(20, numberOfThreads);
        config.setNireq(20);
        ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

        std::vector<std::promise<void>> releaseWaitBeforeGettingModelInstance(numberOfThreads);
        std::vector<std::thread> predictThreads;
        for (auto i = 0u; i < numberOfThreads; ++i) {
            predictThreads.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforeGettingModelInstance, i]() {
                        tensorflow::serving::PredictRequest request = preparePredictRequest(
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::shape_t, tensorflow::DataType>{{(initialBatchSize + i), 10}, tensorflow::DataType::DT_FLOAT}}});
                        performPredict(config.getName(), config.getVersion(), request,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance[i].get_future())));
                    }));
        }
        // sleep to allow all threads to initialize
        std::this_thread::sleep_for(std::chrono::seconds(2));
        for (auto& promise : releaseWaitBeforeGettingModelInstance) {
            promise.set_value();
        }

        for (auto& thread : predictThreads) {
            thread.join();
        }
    }

    static void checkOutputShape(const tensorflow::serving::PredictResponse& response, const ovms::shape_t& shape) {
        ASSERT_EQ(response.outputs().count("a"), 1);
        const auto& output_tensor = response.outputs().at("a");
        ASSERT_EQ(output_tensor.tensor_shape().dim_size(), shape.size());
        for (unsigned int i = 0; i < shape.size(); i++) {
            EXPECT_EQ(output_tensor.tensor_shape().dim(i).size(), shape[i]);
        }
    }

    ovms::Status performInferenceWithRequest(const tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response) {
        std::shared_ptr<ovms::ModelInstance> model;
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
        auto status = manager.getModelInstance("dummy", 0, model, unload_guard);
        if (!status.ok()) {
            return status;
        }

        response.Clear();
        return model->infer(&request, &response, unload_guard);
    }

    ovms::Status performInferenceWithShape(tensorflow::serving::PredictResponse& response, const ovms::shape_t& shape = {1, 10}, const tensorflow::DataType precision = tensorflow::DataType::DT_FLOAT) {
        auto request = preparePredictRequest(
            {{DUMMY_MODEL_INPUT_NAME, std::tuple<ovms::shape_t, tensorflow::DataType>{shape, precision}}});
        return performInferenceWithRequest(request, response);
    }

    ovms::Status performInferenceWithBatchSize(tensorflow::serving::PredictResponse& response, int batchSize = 1, const tensorflow::DataType precision = tensorflow::DataType::DT_FLOAT) {
        ovms::shape_t shape = {batchSize, 10};
        auto request = preparePredictRequest(
            {{DUMMY_MODEL_INPUT_NAME, std::tuple<ovms::shape_t, tensorflow::DataType>{shape, precision}}});
        return performInferenceWithRequest(request, response);
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
    const tensorflow::serving::PredictRequest& request,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference) {
    // only validation is skipped
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;

    auto& tensorProto = request.inputs().find("b")->second;
    size_t batchSize = tensorProto.tensor_shape().dim(0).size();
    size_t inputSize = 1;
    for (int i = 0; i < tensorProto.tensor_shape().dim_size(); i++) {
        inputSize *= tensorProto.tensor_shape().dim(i).size();
    }

    if (waitBeforeGettingModelInstance) {
        std::cout << "Waiting before getModelInstance. Batch size: " << batchSize << std::endl;
        waitBeforeGettingModelInstance->get();
    }
    ASSERT_EQ(manager.getModelInstance(modelName, modelVersion, modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    if (waitBeforePerformInference) {
        std::cout << "Waiting before performInfernce." << std::endl;
        waitBeforePerformInference->get();
    }
    ovms::Status validationStatus = modelInstance->validate(&request, nullptr);
    ASSERT_TRUE(validationStatus == ovms::StatusCode::OK ||
                validationStatus == ovms::StatusCode::RESHAPE_REQUIRED ||
                validationStatus == ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
    ASSERT_EQ(modelInstance->reloadModelIfRequired(validationStatus, &request, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    ovms::ExecutingStreamIdGuard executingStreamIdGuard(modelInstance->getInferRequestsQueue());
    InferenceEngine::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    std::vector<float> input(inputSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));
    deserialize(input, inferRequest, modelInstance);
    auto status = modelInstance->performInference(inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    size_t outputSize = batchSize * DUMMY_MODEL_OUTPUT_SIZE;
    serializeAndCheck(outputSize, inferRequest);
}

TEST_F(TestPredict, SuccesfullOnDummyModel) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    performPredict(config.getName(), config.getVersion(), request);
}

TEST_F(TestPredict, SuccesfullReloadFromAlreadyLoadedWithNewBatchSize) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const int initialBatchSize = config.getBatchSize();
    config.setBatchSize(initialBatchSize);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    performPredict(config.getName(), config.getVersion(), request);
}

TEST_F(TestPredict, SuccesfullReloadWhen1InferenceInProgress) {
    //  FIRST LOAD MODEL WITH BS=1
    tensorflow::serving::PredictRequest requestBs1 = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    tensorflow::serving::PredictRequest requestBs2 = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{2, 10}, tensorflow::DataType::DT_FLOAT}}});

    config.setBatchingParams("auto");
    config.setNireq(2);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

    std::promise<void> releaseWaitBeforePerformInferenceBs1, releaseWaitBeforeGetModelInstanceBs2;
    std::thread t1(
        [this, &requestBs1, &releaseWaitBeforePerformInferenceBs1]() {
            performPredict(config.getName(), config.getVersion(), requestBs1, nullptr,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInferenceBs1.get_future())));
        });
    std::thread t2(
        [this, &requestBs2, &releaseWaitBeforeGetModelInstanceBs2]() {
            performPredict(config.getName(), config.getVersion(), requestBs2,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGetModelInstanceBs2.get_future())),
                nullptr);
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    releaseWaitBeforePerformInferenceBs1.set_value();
    releaseWaitBeforeGetModelInstanceBs2.set_value();
    t1.join();
    t2.join();
}

TEST_F(TestPredict, SuccesfullReloadWhen1InferenceAboutToStart) {
    //  FIRST LOAD MODEL WITH BS=1
    tensorflow::serving::PredictRequest requestBs1 = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    tensorflow::serving::PredictRequest requestBs2 = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{2, 10}, tensorflow::DataType::DT_FLOAT}}});

    config.setBatchingParams("auto");
    config.setNireq(2);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

    std::promise<void> releaseWaitBeforeGetModelInstanceBs1, releaseWaitBeforePerformInferenceBs2;
    std::thread t1(
        [this, &requestBs1, &releaseWaitBeforeGetModelInstanceBs1]() {
            performPredict(config.getName(), config.getVersion(), requestBs1,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGetModelInstanceBs1.get_future())),
                nullptr);
        });
    std::thread t2(
        [this, &requestBs2, &releaseWaitBeforePerformInferenceBs2]() {
            performPredict(config.getName(), config.getVersion(), requestBs2, nullptr,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInferenceBs2.get_future())));
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    releaseWaitBeforePerformInferenceBs2.set_value();
    releaseWaitBeforeGetModelInstanceBs1.set_value();
    t1.join();
    t2.join();
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeGettingModelInstance) {
    const int initialBatchSize = 1;
    config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 0;
    const uint waitingBeforeGettingModelCount = 9;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeInference) {
    const int initialBatchSize = 1;
    config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 0;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TEST_F(TestPredict, SuccesfullReloadWhenSeveralInferRequestAtDifferentStages) {
    const int initialBatchSize = 1;
    config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 9;
    testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TEST_F(TestPredict, SuccesfullReloadForMultipleThreadsDifferentBS) {
    const int initialBatchSize = 2;
    config.setBatchingParams("auto");

    const uint numberOfThreads = 5;
    testConcurrentBsChanges(initialBatchSize, numberOfThreads);
}

TEST_F(TestPredict, SuccesfullReshapeViaRequestOnDummyModel) {
    // Prepare model manager with dynamic shaped dummy model, originally loaded with 1x10 shape
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);

    // Get dummy model instance
    std::shared_ptr<ovms::ModelInstance> model;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    auto status = manager.getModelInstance("dummy", 0, model, unload_guard);

    // Prepare request with 1x5 shape, expect reshape
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 5}, tensorflow::DataType::DT_FLOAT}}});

    tensorflow::serving::PredictResponse response;

    // Do the inference
    ASSERT_EQ(model->infer(&request, &response, unload_guard), ovms::StatusCode::OK);

    // Expect reshape to 1x5
    ASSERT_EQ(response.outputs().count("a"), 1);
    auto& output_tensor = (*response.mutable_outputs())["a"];
    ASSERT_EQ(output_tensor.tensor_shape().dim_size(), 2);
    EXPECT_EQ(output_tensor.tensor_shape().dim(0).size(), 1);
    EXPECT_EQ(output_tensor.tensor_shape().dim(1).size(), 5);
}

/**
 * Scenario - perform inferences with different shapes and model reload via config.json change
 * 
 * 1. Load model with shape=auto, initial internal shape (1,10)
 * 2. Do the inference with (1,12) shape - expect status OK and result (1,12)
 * 3. Reshape model to fixed=(1,11) with config.json change
 * 4. Do the inference with (1,12) shape - expect status INVALID_SHAPE
 * 5. Do the inference with (1,11) shape - expect status OK and result (1,11)
 * 6. Reshape model back to shape=auto, initial internal shape (1,10)
 * 7. Do the inference with (1,12) shape - expect status OK and result (1,12)
 */
TEST_F(TestPredict, ReshapeViaRequestAndConfigChange) {
    using namespace ovms;

    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    tensorflow::serving::PredictResponse response;

    // Perform reshape to (1,12) using request
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), StatusCode::OK);
    checkOutputShape(response, {1, 12});

    // Reshape with model reload to Fixed=(1,11)
    config.setBatchingParams("0");
    config.parseShapeParameter("(1,11)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    // Cannot do the inference with (1,12)
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), StatusCode::INVALID_SHAPE);

    // Successfull inference with (1,11)
    ASSERT_EQ(performInferenceWithShape(response, {1, 11}), StatusCode::OK);
    checkOutputShape(response, {1, 11});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    // Perform reshape to (1,12) using request
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), StatusCode::OK);
    checkOutputShape(response, {1, 12});
}

/**
 * Scenario - perform inferences with different batch size and model reload via config.json change
 * 
 * 1. Load model with bs=auto, initial internal shape (1,10)
 * 2. Do the inference with (3,10) shape - expect status OK and result (3,10)
 * 3. Change model batch size to fixed=4 with config.json change
 * 4. Do the inference with (3,10) shape - expect status INVALID_BATCH_SIZE
 * 5. Do the inference with (4,10) shape - expect status OK and result (4,10)
 * 6. Reshape model back to batchsize=auto, initial internal shape (1,10)
 * 7. Do the inference with (3,10) shape - expect status OK and result (3,10)
 */
TEST_F(TestPredict, ChangeBatchSizeViaRequestAndConfigChange) {
    using namespace ovms;

    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    tensorflow::serving::PredictResponse response;

    // Perform batch size change to 3 using request
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), StatusCode::OK);
    checkOutputShape(response, {3, 10});

    // Change batch size with model reload to Fixed=4
    config.setBatchingParams("4");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    // Cannot do the inference with (3,10)
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), StatusCode::INVALID_BATCH_SIZE);

    // Successfull inference with (4,10)
    ASSERT_EQ(performInferenceWithBatchSize(response, 4), StatusCode::OK);
    checkOutputShape(response, {4, 10});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    // Perform batch change to 3 using request
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), StatusCode::OK);
    checkOutputShape(response, {3, 10});
}
#pragma GCC diagnostic pop
