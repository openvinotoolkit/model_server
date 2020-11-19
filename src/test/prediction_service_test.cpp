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

#include "prediction_service_test.hpp"

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
    auto status = ovms::getModelInstance(manager, "dummy", 0, model, unload_guard);

    // Prepare request with 1x5 shape, expect reshape
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 5}, tensorflow::DataType::DT_FLOAT}}});

    tensorflow::serving::PredictResponse response;

    // Do the inference
    ASSERT_EQ(inference(*model, &request, &response, unload_guard), ovms::StatusCode::OK);

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
