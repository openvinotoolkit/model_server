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
#include <fstream>
#include <filesystem>
#include <cstring>

#include <stdlib.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <inference_engine.hpp>

#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "test_utils.hpp"

class TestPredict : public ::testing::Test {};

using testing::Each;
using testing::Eq;

void performPredict(ovms::ModelInstance& modelInstance, const int inputSize) {
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance.getInferRequestsQueue();
    ovms::ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);

    std::vector<float> input(inputSize);
    std::generate(input.begin(), input.end(), [](){ return 1.;});
    ASSERT_THAT(input, Each(Eq(1.)));

    // deserialize
    auto blob = InferenceEngine::make_shared_blob<float>(
        modelInstance.getInputsInfo().at(DUMMY_MODEL_INPUT_NAME)->getTensorDesc(),
        const_cast<float*>(reinterpret_cast<const float*>(input.data())));
    inferRequest.SetBlob(DUMMY_MODEL_INPUT_NAME, blob);

    // perform inference
    auto status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // serialize
    std::vector<float> output(inputSize);
    ASSERT_THAT(output, Each(Eq(0.)));
    auto blobOutput = inferRequest.GetBlob(DUMMY_MODEL_OUTPUT_NAME);
    std::memcpy(output.data(), blobOutput->cbuffer(), inputSize * sizeof(float));
    EXPECT_THAT(output, Each(Eq(2.)));
}

TEST_F(TestPredict, SuccesfullOnDummyModel) {
    ovms::ModelInstance modelInstance;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    // TODO dirty hack to avoid initializing config
    setenv("NIREQ", "1", 1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    performPredict(modelInstance, DUMMY_MODEL_INPUT_SIZE);
}


TEST_F(TestPredict, SuccesfullReloadFromAlreadyLoadedWithNewBatchSize) {
    ovms::ModelInstance modelInstance;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const int initialBatchSize = 1;
    config.setBatchSize(initialBatchSize);
    // TODO dirty hack to avoid initializing config
    setenv("NIREQ", "1", 1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    performPredict(modelInstance, DUMMY_MODEL_INPUT_SIZE * initialBatchSize);
    auto newBatchSize = config.getBatchSize() + 1;
    EXPECT_EQ(modelInstance.reloadModel(newBatchSize), ovms::StatusCode::OK);
    performPredict(modelInstance, DUMMY_MODEL_INPUT_SIZE * newBatchSize);
}
