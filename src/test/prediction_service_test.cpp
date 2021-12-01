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
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <stdlib.h>

#include "../deserialization.hpp"
#include "../executingstreamidguard.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../sequence_processing_spec.hpp"
#include "../serialization.hpp"
#include "test_utils.hpp"

using testing::Each;
using testing::Eq;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
void serializeAndCheck(int outputSize, ov::runtime::InferRequest& inferRequest, const std::string& outputName, const ovms::tensor_map_t& outputsInfo) {
    std::vector<float> output(10);
    tensorflow::serving::PredictResponse response;
    auto status = serializePredictResponse_2(inferRequest, outputsInfo, &response);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    std::memcpy(output.data(), (float*)response.outputs().at(outputName).tensor_content().data(), DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    EXPECT_THAT(output, Each(Eq(1.)));
}
class TestPredict : public ::testing::Test {
public:
    void SetUp() {
        ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
        const int initialBatchSize = 1;
        config.setBatchSize(initialBatchSize);
        config.setNireq(2);
    }
    /**
     * @brief This function should mimic most closely predict request to check for thread safety
     */
    void performPredict(const std::string modelName,
        const ovms::model_version_t modelVersion,
        const tensorflow::serving::PredictRequest& request,
        std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance = nullptr,
        std::unique_ptr<std::future<void>> waitBeforePerformInference = nullptr);

    void testConcurrentPredicts(const int initialBatchSize, const uint waitingBeforePerformInferenceCount, const uint waitingBeforeGettingModelCount) {
        ASSERT_GE(20, waitingBeforePerformInferenceCount);
        config.setNireq(20);
        ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

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
        ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

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

    static void checkOutputShape(const tensorflow::serving::PredictResponse& response, const ovms::shape_t& shape, const std::string& outputName = "a") {
        ASSERT_EQ(response.outputs().count(outputName), 1);
        const auto& output_tensor = response.outputs().at(outputName);
        ASSERT_EQ(output_tensor.tensor_shape().dim_size(), shape.size());
        for (unsigned int i = 0; i < shape.size(); i++) {
            EXPECT_EQ(output_tensor.tensor_shape().dim(i).size(), shape[i]);
        }
    }

    static void checkOutputValues(const tensorflow::serving::PredictResponse& response, const std::vector<float>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME) {
        ASSERT_EQ(response.outputs().count(outputName), 1);
        const auto& output_tensor = response.outputs().at(outputName);
        float* buffer = (float*)output_tensor.tensor_content().data();
        std::vector<float> actualValues(buffer, buffer + output_tensor.tensor_content().size() / sizeof(float));
        ASSERT_EQ(0, std::memcmp(actualValues.data(), expectedValues.data(), expectedValues.size() * sizeof(float)))
            << readableError(expectedValues.data(), actualValues.data(), expectedValues.size() * sizeof(float));
    }

    ovms::Status performInferenceWithRequest(const tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response, const std::string& servableName = "dummy") {
        std::shared_ptr<ovms::ModelInstance> model;
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
        auto status = manager.getModelInstance(servableName, 0, model, unload_guard);
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

    ovms::Status performInferenceWithImageInput(tensorflow::serving::PredictResponse& response, const std::vector<size_t>& shape, const std::vector<float>& data = {}, const std::string& servableName = "increment_1x3x4x5", int batchSize = 1, const tensorflow::DataType precision = tensorflow::DataType::DT_FLOAT) {
        auto request = preparePredictRequest(
            {{INCREMENT_1x3x4x5_MODEL_INPUT_NAME, std::tuple<ovms::shape_t, tensorflow::DataType>{shape, precision}}}, data);
        return performInferenceWithRequest(request, response, servableName);
    }

    ovms::Status performInferenceWithBinaryImageInput(tensorflow::serving::PredictResponse& response, const std::string& inputName, const std::string& servableName = "increment_1x3x4x5", int batchSize = 1) {
        auto request = prepareBinaryPredictRequest(inputName, batchSize);
        return performInferenceWithRequest(request, response, servableName);
    }

public:
    ConstructorEnabledModelManager manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ~TestPredict() {
        std::cout << "Destructor of TestPredict()" << std::endl;
    }
};

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) :
        ModelInstance("UNUSED_NAME", 42, ieCore, ieCore_2) {}
    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
        return validate(request);
    }
};

void performPrediction(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const tensorflow::serving::PredictRequest& request,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference,
    ovms::ModelManager& manager,
    const std::string& inputName,
    const std::string& outputName) {
    // only validation is skipped
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;

    auto& tensorProto = request.inputs().at(inputName);
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
    ovms::Status validationStatus = (std::static_pointer_cast<MockModelInstance>(modelInstance))->mockValidate(&request);
    ASSERT_TRUE(validationStatus == ovms::StatusCode::OK ||
                validationStatus == ovms::StatusCode::RESHAPE_REQUIRED ||
                validationStatus == ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
    ASSERT_EQ(modelInstance->reloadModelIfRequired(validationStatus, &request, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    ovms::ExecutingStreamIdGuard_2 executingStreamIdGuard(modelInstance->getInferRequestsQueue_2());
    ov::runtime::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    ovms::InputSink_2<ov::runtime::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;

    auto status = ovms::deserializePredictRequest_2<ovms::ConcreteTensorProtoDeserializator_2>(request, modelInstance->getInputsInfo(), inputSink, isPipeline);
    status = modelInstance->performInference_2(inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    size_t outputSize = batchSize * DUMMY_MODEL_OUTPUT_SIZE;
    serializeAndCheck(outputSize, inferRequest, outputName, modelInstance->getOutputsInfo());
}
void TestPredict::performPredict(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const tensorflow::serving::PredictRequest& request,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference) {
    performPrediction(modelName,
        modelVersion,
        request,
        std::move(waitBeforeGettingModelInstance),
        std::move(waitBeforePerformInference),
        this->manager,
        DUMMY_MODEL_INPUT_NAME,
        DUMMY_MODEL_OUTPUT_NAME);
}

TEST_F(TestPredict, SuccesfullOnDummyModel) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);

    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    performPredict(config.getName(), config.getVersion(), request);
}

static const char* oneDummyWithMappedInputConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"input_tensor": "(1,10) "}
            }
        }
    ]
})";

class TestPredictWithMapping : public TestWithTempDir {
protected:
    std::string ovmsConfig;
    std::string modelPath;
    std::string configFilePath;
    std::string mappingConfigPath;
    const std::string dummyModelInputMapping = "input_tensor";
    const std::string dummyModelOutputMapping = "output_tensor";

public:
    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        const std::string modelPathToReplace{"/ovms/src/test/dummy"};
        auto it = ovmsConfig.find(modelPathToReplace);
        if (it != std::string::npos) {
            ovmsConfig.replace(ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), modelPath);
        }
        configFilePath = directoryPath + "/ovms_config.json";
    }
    void SetUp() {
        TestWithTempDir::SetUp();
        modelPath = directoryPath + "/dummy/";
        mappingConfigPath = modelPath + "1/mapping_config.json";
        SetUpConfig(oneDummyWithMappedInputConfig);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        createConfigFileWithContent(R"({
            "inputs": {"b":"input_tensor"},
            "outputs": {"a": "output_tensor"}
        })",
            mappingConfigPath);
    }
};

TEST_F(TestPredictWithMapping, SuccesfullOnDummyModelWithMapping) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{dummyModelInputMapping,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(configFilePath);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    performPrediction(config.getName(), config.getVersion(), request, nullptr, nullptr, manager, dummyModelInputMapping, dummyModelOutputMapping);
}

TEST_F(TestPredict, SuccesfullReloadFromAlreadyLoadedWithNewBatchSize) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const int initialBatchSize = config.getBatchSize();
    config.setBatchSize(initialBatchSize);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
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
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

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
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

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

// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_SuccesfullReshapeViaRequestOnDummyModel) {
    // Prepare model manager with dynamic shaped dummy model, originally loaded with 1x10 shape
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

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
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_ReshapeViaRequestAndConfigChange) {
    using namespace ovms;

    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    // Perform reshape to (1,12) using request
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 12});

    // Reshape with model reload to Fixed=(1,11)
    config.setBatchingParams("0");
    config.parseShapeParameter("(1,11)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Cannot do the inference with (1,12)
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), ovms::StatusCode::INVALID_SHAPE);

    // Successfull inference with (1,11)
    ASSERT_EQ(performInferenceWithShape(response, {1, 11}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 11});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform reshape to (1,12) using request
    ASSERT_EQ(performInferenceWithShape(response, {1, 12}), ovms::StatusCode::OK);
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
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_ChangeBatchSizeViaRequestAndConfigChange) {
    using namespace ovms;

    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    // Perform batch size change to 3 using request
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), ovms::StatusCode::OK);
    checkOutputShape(response, {3, 10});

    // Change batch size with model reload to Fixed=4
    config.setBatchingParams("4");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Cannot do the inference with (3,10)
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), ovms::StatusCode::INVALID_BATCH_SIZE);

    // Successfull inference with (4,10)
    ASSERT_EQ(performInferenceWithBatchSize(response, 4), ovms::StatusCode::OK);
    checkOutputShape(response, {4, 10});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("auto");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform batch change to 3 using request
    ASSERT_EQ(performInferenceWithBatchSize(response, 3), ovms::StatusCode::OK);
    checkOutputShape(response, {3, 10});
}

/**
 * Scenario - perform inference with NHWC input layout changed via config.json.
 * 
 * 1. Load model with layout=nhwc, initial internal layout: nchw
 * 2. Do the inference with (1,4,5,3) shape - expect status OK and result (1,3,4,5)
 * 3. Do the inference with (1,3,4,5) shape - expect INVALID_SHAPE
 * 4. Remove layout setting
 * 5. Do the inference with (1,4,5,3) shape - expect status OK and result (1,3,4,5): model keeps old nhwc setting
 * 6. Do the inference with (1,3,4,5) shape - expect INVALID_SHAPE
 * 7. Add layout setting to nchw
 * 8. Do the inference with (1,3,4,5) shape - expect status OK and result (1,3,4,5):
 * 9. Do the inference with (1,4,5,3) shape - expect INVALID_SHAPE
 */
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_PerformInferenceChangeModelInputLayout) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    // Perform inference with NHWC layout, ensure status OK and correct results
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 5.0, 3.0, 6.0, 4.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NCHW layout, ensure error
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);

    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NHWC layout, ensure status OK and correct results
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 5.0, 3.0, 6.0, 4.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NCHW layout, ensure error
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);

    // Prepare model with layout changed back to nchw
    ASSERT_EQ(config.parseLayoutParameter("nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and correct results
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NHWC layout, ensure error
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);
}

/**
 * Scenario - change output layout of model and perform inference. Check results if in correct order.
 *
 * 1. Load model with output layout=nhwc, initial internal layout: nchw
 * 2. Do the inference with (1,3,4,5) shape - expect status OK and result in NHWC layout
 * 3. Remove layout setting
 * 4. Do the inference with (1,3,4,5) shape - expect status OK and result in NHWC layout
 * 5. Roll back layout setting to internal nchw
 * 6. Do the inference with (1,3,4,5) shape - expect status OK and result in NCHW layout
 */
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_PerformInferenceChangeModelOutputLayout) {
    using namespace ovms;

    // Prepare model with changed output layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nhwc\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    // Perform inference with NCHW layout, ensure status OK and results in NHWC order
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 1, 2, 3}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 4.0, 6.0, 3.0, 5.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and results still in NHWC order
    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 1, 2, 3}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 4.0, 6.0, 3.0, 5.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Change output layout back to original nchw.
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nchw\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    ASSERT_EQ(performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

TEST_F(TestPredict, ErrorWhenLayoutSetForMissingTensor) {
    ovms::ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("{\"invalid_tensor_name\":\"nhwc\"}"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::CONFIG_LAYOUT_IS_NOT_IN_NETWORK);
}

TEST_F(TestPredict, NetworkNotLoadedWhenLayoutAndDimsInconsistent) {
    // Dummy has 2 dimensions: (1,10), changing layout to NHWC should fail
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::NETWORK_NOT_LOADED);
}

/**
 * Scenario - change input layout of model and perform inference with binary input. Check results.
 *
 * 1. Load model with input layout=nhwc, initial internal layout: nchw
 * 2. Do the inference with single binary image tensor - expect status OK and result in NCHW layout
 * 3. Set layout setting to internal nchw
 * 4. Do the inference with single binary image tensor - expect status UNSUPPORTED_LAYOUT
 * 5. Set back layout setting to nhwc
 * 6. Do the inference with single binary image tensor - expect status OK and result in NCHW layout
 */
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_PerformInferenceWithBinaryInputChangeModelInputLayout) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter("nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with binary input, ensure validation rejects the request due to NCHW setting
    ASSERT_EQ(performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::UNSUPPORTED_LAYOUT);

    // Switch back to nhwc
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with binary input, ensure status OK after switching layout and correct results
    ASSERT_EQ(performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::OK);
    checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/**
 * Scenario - send binary input request to model accepting auto batch size.
 *
 * 1. Load model with input layout=nhwc, batch_size=auto, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with batch=5 binary image tensor - expect status OK and result in NCHW layout
 */
// TODO: Enable once Layout Change becomes available using OV 2.0.
TEST_F(TestPredict, DISABLED_PerformInferenceWithBinaryInputBatchSizeAuto) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;

    const int batchSize = 5;
    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5", batchSize), ovms::StatusCode::OK);
    checkOutputShape(response, {5, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/**
 * Scenario - send binary input request with no shape set.
 *
 * 1. Load model with input layout=nhwc, batch_size=auto, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with binary image tensor with no shape set - expect status INVALID_NO_OF_SHAPE_DIMENSIONS
 */

TEST_F(TestPredict, PerformInferenceWithBinaryInputNoInputShape) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    tensorflow::serving::PredictResponse response;
    tensorflow::serving::PredictRequest request;
    auto& tensor = (*request.mutable_inputs())[INCREMENT_1x3x4x5_MODEL_INPUT_NAME];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);
    tensor.add_string_val(image_bytes.get(), filesize);
    tensor.set_dtype(tensorflow::DataType::DT_STRING);

    // Perform inference with binary input, ensure status INVALID_NO_OF_SHAPE_DIMENSIONS
    ASSERT_EQ(performInferenceWithRequest(request, response, "increment_1x3x4x5"), ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

#pragma GCC diagnostic pop
