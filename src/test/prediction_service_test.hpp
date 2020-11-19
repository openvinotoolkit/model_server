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
#pragma once

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

#include "../executinstreamidguard.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
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
        std::unique_ptr<std::future<void>> waitBeforePerformInference = nullptr) {
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
        ASSERT_EQ(getModelInstance(manager, modelName, modelVersion, modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);

        if (waitBeforePerformInference) {
            std::cout << "Waiting before performInfernce." << std::endl;
            waitBeforePerformInference->get();
        }
        ovms::Status validationStatus = modelInstance->validate(&request);
        ASSERT_TRUE(validationStatus == ovms::StatusCode::OK ||
                    validationStatus == ovms::StatusCode::RESHAPE_REQUIRED ||
                    validationStatus == ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
        ASSERT_EQ(reloadModelIfRequired(validationStatus, *modelInstance, &request, modelInstanceUnloadGuard), ovms::StatusCode::OK);

        ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance->getInferRequestsQueue();
        ovms::ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
        int executingInferId = executingStreamIdGuard.getId();
        InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
        std::vector<float> input(inputSize);
        std::generate(input.begin(), input.end(), []() { return 1.; });
        ASSERT_THAT(input, Each(Eq(1.)));
        deserialize(input, inferRequest, modelInstance);
        auto status = performInference(inferRequestsQueue, executingInferId, inferRequest);
        ASSERT_EQ(status, ovms::StatusCode::OK);
        size_t outputSize = batchSize * DUMMY_MODEL_OUTPUT_SIZE;
        serializeAndCheck(outputSize, inferRequest);
    }

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
        auto status = ovms::getModelInstance(manager, "dummy", 0, model, unload_guard);
        if (!status.ok()) {
            return status;
        }

        response.Clear();
        return ovms::inference(*model, &request, &response, unload_guard);
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
