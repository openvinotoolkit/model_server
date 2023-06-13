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
#include <openvino/openvino.hpp>
#include <stdlib.h>

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/inferenceparameter.hpp"
#include "../capi_frontend/inferencerequest.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../capi_frontend/inferencetensor.hpp"
#include "../deserialization.hpp"
#include "../executingstreamidguard.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelversion.hpp"
#include "../prediction_service_utils.hpp"
#include "../sequence_processing_spec.hpp"
#include "../serialization.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "test_utils.hpp"

using testing::Each;
using testing::ElementsAre;
using testing::Eq;

using ovms::Buffer;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::StatusCode;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
static void serializeAndCheck(int outputSize, ov::InferRequest& inferRequest, const std::string& outputName, const ovms::tensor_map_t& outputsInfo) {
    std::vector<float> output(outputSize);
    tensorflow::serving::PredictResponse response;
    ovms::OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_SERVABLE_NAME, UNUSED_MODEL_VERSION, outputsInfo, &response, ovms::getTensorInfoName);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    std::memcpy(output.data(), (float*)response.outputs().at(outputName).tensor_content().data(), outputSize * sizeof(float));
    EXPECT_THAT(output, Each(Eq(1.)));
}

static ovms::Status getOutput(const KFSResponse& response, const std::string& name, KFSOutputTensorIteratorType& it, size_t& bufferId) {
    it = response.outputs().begin();
    bufferId = 0;
    while (it != response.outputs().end()) {
        if (it->name() == name) {
            break;
        }
        ++it;
        ++bufferId;
    }
    if (it != response.outputs().end()) {
        return StatusCode::OK;
    }
    return StatusCode::INVALID_MISSING_INPUT;
}

static ovms::Status getOutput(const TFSResponseType& response, const std::string& name, TFSOutputTensorIteratorType& it, size_t& bufferId) {
    it = response.outputs().find(name);
    if (it != response.outputs().end()) {
        return StatusCode::OK;
    }
    return StatusCode::INVALID_MISSING_INPUT;
}

using inputs_info_elem_t = std::pair<std::string, std::tuple<ovms::signed_shape_t, ovms::Precision>>;
static size_t calculateByteSize(const inputs_info_elem_t& e) {
    auto& [inputName, shapeDatatypeTuple] = e;
    auto& [shape, precision] = shapeDatatypeTuple;
    size_t shapeProduct = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return shapeProduct * ovms::DataTypeToByteSize(ovms::getPrecisionAsOVMSDataType(precision));
}
template <typename RequestType>
class Preparer {
    std::vector<std::unique_ptr<std::vector<float>>> dataKeeper;

public:
    void preparePredictRequest(RequestType& request, inputs_info_t requestInputs) {
        ::preparePredictRequest(request, requestInputs);
    }
};
template <>
void Preparer<ovms::InferenceRequest>::preparePredictRequest(ovms::InferenceRequest& request, inputs_info_t requestInputs) {
    auto inputWithGreatestRequirements = std::max_element(requestInputs.begin(), requestInputs.end(), [](const inputs_info_elem_t& a, const inputs_info_elem_t& b) {
        return calculateByteSize(a) < calculateByteSize(b);
    });
    size_t byteSizeToPreserve = calculateByteSize(*inputWithGreatestRequirements);
    auto& currentData = dataKeeper.emplace_back(std::make_unique<std::vector<float>>(byteSizeToPreserve));
    memset(reinterpret_cast<void*>(const_cast<float*>(currentData->data())), '1', byteSizeToPreserve);
    ::preparePredictRequest(request, requestInputs, *currentData);
}

template <typename Pair,
    typename RequestType = typename Pair::first_type,
    typename ResponseType = typename Pair::second_type>
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
        const RequestType& request,
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
                        RequestType request;
                        Preparer<RequestType> preparer;
                        preparer.preparePredictRequest(request,
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::signed_shape_t, ovms::Precision>{{(initialBatchSize + (i % 3)), 10}, ovms::Precision::FP32}}});

                        performPredict(config.getName(), config.getVersion(), request,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance[i].get_future())));
                    }));
        }
        for (auto i = 0u; i < waitingBeforePerformInferenceCount; ++i) {
            predictsWaitingBeforeInference.emplace_back(
                std::thread(
                    [this, initialBatchSize, &releaseWaitBeforePerformInference, i]() {
                        RequestType request;
                        Preparer<RequestType> preparer;
                        preparer.preparePredictRequest(request,
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::signed_shape_t, ovms::Precision>{{initialBatchSize, 10}, ovms::Precision::FP32}}});

                        performPredict(config.getName(), config.getVersion(), request, nullptr,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInference[i].get_future())));
                    }));
        }
        // sleep to allow all threads to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
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
                        RequestType request;
                        Preparer<RequestType> preparer;
                        preparer.preparePredictRequest(request,
                            {{DUMMY_MODEL_INPUT_NAME,
                                std::tuple<ovms::signed_shape_t, ovms::Precision>{{(initialBatchSize + i), 10}, ovms::Precision::FP32}}});
                        performPredict(config.getName(), config.getVersion(), request,
                            std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGettingModelInstance[i].get_future())));
                    }));
        }
        // sleep to allow all threads to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        for (auto& promise : releaseWaitBeforeGettingModelInstance) {
            promise.set_value();
        }

        for (auto& thread : predictThreads) {
            thread.join();
        }
    }

    void checkOutputShape(const ResponseType& response, const ovms::signed_shape_t& shape, const std::string& outputName = "a");

    static void checkOutputValuesU8(const TFSResponseType& response, const std::vector<uint8_t>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME, bool checkRaw = true) {
        ASSERT_EQ(response.outputs().count(outputName), 1);
        const auto& output_tensor = response.outputs().at(outputName);
        uint8_t* buffer = reinterpret_cast<uint8_t*>(const_cast<char*>(output_tensor.tensor_content().data()));
        std::vector<uint8_t> actualValues(buffer, buffer + output_tensor.tensor_content().size() / sizeof(uint8_t));
        ASSERT_EQ(actualValues.size(), expectedValues.size());
        ASSERT_EQ(0, std::memcmp(actualValues.data(), expectedValues.data(), expectedValues.size() * sizeof(uint8_t)))
            << readableError(expectedValues.data(), actualValues.data(), expectedValues.size() * sizeof(uint8_t));
    }
    static void checkOutputValues(const TFSResponseType& response, const std::vector<float>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME) {
        ASSERT_EQ(response.outputs().count(outputName), 1);
        const auto& output_tensor = response.outputs().at(outputName);
        float* buffer = reinterpret_cast<float*>(const_cast<char*>(output_tensor.tensor_content().data()));
        std::vector<float> actualValues(buffer, buffer + output_tensor.tensor_content().size() / sizeof(float));
        ASSERT_EQ(0, std::memcmp(actualValues.data(), expectedValues.data(), expectedValues.size() * sizeof(float)))
            << readableError(expectedValues.data(), actualValues.data(), expectedValues.size() * sizeof(float));
    }
    static void checkOutputValuesU8(const ovms::InferenceResponse& res, const std::vector<uint8_t>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME, bool checkRaw = true) {
        FAIL() << "not supported";
    }
    static void checkOutputValues(const ovms::InferenceResponse& res, const std::vector<float>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME) {
        InferenceResponse& response = const_cast<InferenceResponse&>(res);
        size_t outputCount = response.getOutputCount();
        ASSERT_GE(1, outputCount);
        size_t outputId = 0;
        while (outputId < outputCount) {
            const std::string* cppName;
            InferenceTensor* tensor;
            auto status = response.getOutput(outputId, &cppName, &tensor);
            ASSERT_EQ(status, StatusCode::OK) << status.string();
            ASSERT_NE(nullptr, tensor);
            ASSERT_NE(nullptr, cppName);
            if (outputName == *cppName) {
                const Buffer* buffer = tensor->getBuffer();
                ASSERT_NE(nullptr, buffer);
                ASSERT_EQ(expectedValues.size() * sizeof(float), buffer->getByteSize());
                float* bufferRaw = reinterpret_cast<float*>(const_cast<void*>(buffer->data()));
                ASSERT_EQ(0, std::memcmp(bufferRaw, expectedValues.data(), expectedValues.size() * sizeof(float)))
                    << readableError(expectedValues.data(), bufferRaw, expectedValues.size() * sizeof(float));
                return;
            }
            ++outputId;
        }
        ASSERT_TRUE(false) << "did not found output with name: " << outputName;
    }
    static void checkOutputValues(const KFSResponse& response, const std::vector<float>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME) {
        KFSOutputTensorIteratorType it;
        size_t bufferId;
        auto status = getOutput(response, outputName, it, bufferId);
        ASSERT_TRUE(status.ok()) << "Couln't find output:" << outputName;
        if (response.raw_output_contents().size() > 0) {
            float* buffer = reinterpret_cast<float*>(const_cast<char*>(response.raw_output_contents(bufferId).data()));
            ASSERT_EQ(0, std::memcmp(buffer, expectedValues.data(), expectedValues.size() * sizeof(float)))
                << readableError(expectedValues.data(), buffer, expectedValues.size() * sizeof(float));
        } else {
            auto& responseOutput = *it;
            if (responseOutput.datatype() == "FP32") {
                for (size_t i = 0; i < expectedValues.size(); i++) {
                    ASSERT_EQ(responseOutput.contents().fp32_contents()[i], expectedValues[i]);
                }
            } else if (responseOutput.datatype() == "BYTES") {
                ASSERT_EQ(0, std::memcmp(&responseOutput.contents().bytes_contents(), expectedValues.data(), expectedValues.size() * sizeof(float)));
            }
        }
    }
    static void checkOutputValuesU8(const KFSResponse& response, const std::vector<uint8_t>& expectedValues, const std::string& outputName = INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME, bool checkRaw = true) {
        KFSOutputTensorIteratorType it;
        size_t bufferId;
        auto status = getOutput(response, outputName, it, bufferId);
        ASSERT_TRUE(status.ok()) << "Couln't find output:" << outputName;
        if (checkRaw) {
            ASSERT_GT(response.raw_output_contents().size(), 0);
            uint8_t* buffer = reinterpret_cast<uint8_t*>(const_cast<char*>(response.raw_output_contents(bufferId).data()));
            ASSERT_EQ(response.raw_output_contents(bufferId).size(), expectedValues.size());
            ASSERT_EQ(0, std::memcmp(buffer, expectedValues.data(), expectedValues.size() * sizeof(uint8_t)))
                << readableError(expectedValues.data(), buffer, expectedValues.size() * sizeof(uint8_t));
        } else {
            auto& responseOutput = *it;
            ASSERT_EQ(responseOutput.datatype(), "UINT8") << "other precision testing not supported";
            ASSERT_EQ(expectedValues.size(), responseOutput.contents().uint_contents().size());
            for (size_t i = 0; i < expectedValues.size(); i++) {
                ASSERT_EQ(expectedValues[i], responseOutput.contents().uint_contents(i))
                    << "Wrong value at index " << i << ", expected: " << expectedValues[i] << " actual: " << responseOutput.contents().uint_contents(i);
            }
        }
    }

    ovms::Status performInferenceWithRequest(const RequestType& request, ResponseType& response, const std::string& servableName = "dummy") {
        std::shared_ptr<ovms::ModelInstance> model;
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
        auto status = manager.getModelInstance(servableName, 0, model, unload_guard);
        if (!status.ok()) {
            return status;
        }
        response.Clear();
        return model->infer(&request, &response, unload_guard);
    }

    ovms::Status performInferenceWithShape(ResponseType& response, const ovms::signed_shape_t& shape = {1, 10}, const ovms::Precision precision = ovms::Precision::FP32) {
        RequestType request;
        Preparer<RequestType> preparer;
        preparer.preparePredictRequest(request,
            {{DUMMY_MODEL_INPUT_NAME, std::tuple<ovms::signed_shape_t, ovms::Precision>{shape, precision}}});
        return performInferenceWithRequest(request, response);
    }

    ovms::Status performInferenceWithBatchSize(ResponseType& response, int batchSize = 1, const ovms::Precision precision = ovms::Precision::FP32, const size_t batchSizePosition = 0) {
        ovms::signed_shape_t shape = {1, 10};
        shape[batchSizePosition] = batchSize;
        RequestType request;
        Preparer<RequestType> preparer;
        preparer.preparePredictRequest(request,
            {{DUMMY_MODEL_INPUT_NAME, std::tuple<ovms::signed_shape_t, ovms::Precision>{shape, precision}}});
        return performInferenceWithRequest(request, response);
    }

    ovms::Status performInferenceWithImageInput(ResponseType& response, const ovms::signed_shape_t& shape, const std::vector<float>& data = {}, const std::string& servableName = "increment_1x3x4x5", int batchSize = 1, const ovms::Precision precision = ovms::Precision::FP32) {
        RequestType request;
        Preparer<RequestType> preparer;
        if (data.size()) {
            preparePredictRequest(request,
                {{INCREMENT_1x3x4x5_MODEL_INPUT_NAME, std::tuple<ovms::signed_shape_t, ovms::Precision>{shape, precision}}}, data);
        } else {
            preparer.preparePredictRequest(request,
                {{INCREMENT_1x3x4x5_MODEL_INPUT_NAME, std::tuple<ovms::signed_shape_t, ovms::Precision>{shape, precision}}});
        }
        return performInferenceWithRequest(request, response, servableName);
    }

    ovms::Status performInferenceWithBinaryImageInput(ResponseType& response, const std::string& inputName, const std::string& servableName = "increment_1x3x4x5", int batchSize = 1) {
        RequestType request;
        prepareBinaryPredictRequest(request, inputName, batchSize);
        return performInferenceWithRequest(request, response, servableName);
    }

public:
    ConstructorEnabledModelManager manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ~TestPredict() {
        std::cout << "Destructor of TestPredict()" << std::endl;
    }
};

template <>
void TestPredict<TFSInterface>::checkOutputShape(const TFSResponseType& response, const ovms::signed_shape_t& shape, const std::string& outputName) {
    ASSERT_EQ(response.outputs().count(outputName), 1);
    const auto& output_tensor = response.outputs().at(outputName);
    ASSERT_EQ(output_tensor.tensor_shape().dim_size(), shape.size());
    for (unsigned int i = 0; i < shape.size(); i++) {
        EXPECT_EQ(output_tensor.tensor_shape().dim(i).size(), shape[i]);
    }
}

template <>
void TestPredict<CAPIInterface>::checkOutputShape(const ovms::InferenceResponse& cresponse, const ovms::signed_shape_t& shape, const std::string& outputName) {
    size_t outputCount = cresponse.getOutputCount();
    EXPECT_GE(1, outputCount);
    size_t outputId = 0;
    while (outputId < outputCount) {
        const std::string* cppName;
        InferenceTensor* tensor;
        InferenceResponse& response = const_cast<InferenceResponse&>(cresponse);
        auto status = response.getOutput(outputId, &cppName, &tensor);
        EXPECT_EQ(status, StatusCode::OK) << status.string();
        EXPECT_NE(nullptr, tensor);
        EXPECT_NE(nullptr, cppName);
        if (outputName == *cppName) {
            auto resultShape = tensor->getShape();
            EXPECT_EQ(shape.size(), resultShape.size());
            for (size_t i = 0; i < shape.size(); ++i) {
                EXPECT_EQ(resultShape[i], shape[i]);
            }
        }
        ++outputId;
    }
    return;
}

template <>
void TestPredict<KFSInterface>::checkOutputShape(const KFSResponse& response, const ovms::signed_shape_t& shape, const std::string& outputName) {
    auto it = response.outputs().begin();
    size_t bufferId;
    auto status = getOutput(response, outputName, it, bufferId);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(it->shape().size(), shape.size());
    for (unsigned int i = 0; i < shape.size(); i++) {
        EXPECT_EQ(it->shape()[i], shape[i]);
    }
}

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance(UNUSED_SERVABLE_NAME, UNUSED_MODEL_VERSION, ieCore) {}
    template <typename RequestType>
    const ovms::Status mockValidate(const RequestType* request) {
        return validate(request);
    }
};
const size_t DUMMY_DIM_POS = 1;
template <typename RequestType>
static size_t extractDummyOutputSize(const RequestType& request);

template <>
size_t extractDummyOutputSize(const TFSPredictRequest& request) {
    auto it = request.inputs().begin();
    EXPECT_NE(it, request.inputs().end());
    auto shape = it->second.tensor_shape();
    return shape.dim(DUMMY_DIM_POS).size();
}
template <>
size_t extractDummyOutputSize(const KFSRequest& request) {
    auto it = request.inputs().begin();
    EXPECT_NE(it, request.inputs().end());
    auto shape = it->shape();
    return shape[DUMMY_DIM_POS];
}
template <>
size_t extractDummyOutputSize(const ovms::InferenceRequest& request) {
    return request.getRequestShapes().begin()->second[DUMMY_DIM_POS];
}

template <typename RequestType>
static void performPrediction(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const RequestType& request,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference,
    ovms::ModelManager& manager,
    const std::string& inputName,
    const std::string& outputName) {
    // only validation is skipped
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;

    auto bsPositionIndex = 0;
    auto requestBatchSizeOpt = ovms::getRequestBatchSize(&request, bsPositionIndex);
    ASSERT_TRUE(requestBatchSizeOpt);
    ASSERT_TRUE(requestBatchSizeOpt.value().isStatic());
    auto requestBatchSize = requestBatchSizeOpt.value().getStaticValue();

    if (waitBeforeGettingModelInstance) {
        std::cout << "Waiting before getModelInstance. Batch size: " << requestBatchSize << std::endl;
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
    auto requestShapes = ovms::getRequestShapes(&request);
    ASSERT_EQ(modelInstance->reloadModelIfRequired(validationStatus, requestBatchSize, requestShapes, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    ovms::ExecutingStreamIdGuard executingStreamIdGuard(modelInstance->getInferRequestsQueue(), modelInstance->getMetricReporter());
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    ovms::InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;

    auto status = ovms::deserializePredictRequest<ovms::ConcreteTensorProtoDeserializator>(request, modelInstance->getInputsInfo(), inputSink, isPipeline);
    status = modelInstance->performInference(inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    size_t outputSize = requestBatchSize * extractDummyOutputSize(request);
    serializeAndCheck(outputSize, inferRequest, outputName, modelInstance->getOutputsInfo());
}
template <typename Pair,
    typename RequestType,
    typename ResponseType>
void TestPredict<Pair, RequestType, ResponseType>::performPredict(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const RequestType& request,
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

using MyTypes = ::testing::Types<TFSInterface, KFSInterface, CAPIInterface>;
TYPED_TEST_SUITE(TestPredict, MyTypes);

TYPED_TEST(TestPredict, SuccesfullOnDummyModel) {
    typename TypeParam::first_type request;
    Preparer<typename TypeParam::first_type> preparer;
    preparer.preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);

    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    this->performPredict(config.getName(), config.getVersion(), request);
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
                "nireq": 10,
                "shape": {"input_tensor": "(1,10) "}
            }
        }
    ]
})";

static const char* oneDummyWithMappedInputSpecificAutoShapeConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 10,
                "shape": {"input_tensor": "auto"}
            }
        }
    ]
})";
static const char* oneDummyWithMappedInputAnonymousAutoShapeConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 10,
                "shape": "auto"
            }
        }
    ]
})";
template <typename RequestType>
class TestPredictWithMapping : public TestWithTempDir {
public:
    std::string ovmsConfig;
    std::string modelPath;
    std::string configFilePath;
    std::string mappingConfigPath;
    const std::string dummyModelInputMapping = "input_tensor";
    const std::string dummyModelOutputMapping = "output_tensor";

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
    }
    void SetUp(const std::string& configContent) {
        modelPath = directoryPath + "/dummy/";
        mappingConfigPath = modelPath + "1/mapping_config.json";
        SetUpConfig(configContent);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        createConfigFileWithContent(R"({
            "inputs": {"b":"input_tensor"},
            "outputs": {"a": "output_tensor"}
        })",
            mappingConfigPath);
    }
};

TYPED_TEST_SUITE(TestPredictWithMapping, MyTypes);

TYPED_TEST(TestPredictWithMapping, SuccesfullOnDummyModelWithMapping) {
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type request;
    preparer.preparePredictRequest(request,
        {{this->dummyModelInputMapping,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    this->SetUp(oneDummyWithMappedInputConfig);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(this->configFilePath);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    performPrediction(config.getName(), config.getVersion(), request, nullptr, nullptr, manager, this->dummyModelInputMapping, this->dummyModelOutputMapping);
}

TYPED_TEST(TestPredictWithMapping, SuccesfullOnPassthrough_2D_U8ModelWithMapping) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "String inputs not supported for C-API";
    this->modelPath = this->directoryPath + "/passthrough/";
    this->mappingConfigPath = this->modelPath + "1/mapping_config.json";
    std::filesystem::copy("/ovms/src/test/passthrough", this->modelPath, std::filesystem::copy_options::recursive);
    this->ovmsConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "passhtrough_u8",
                "base_path": "/ovms/src/test/passthrough"
            }
        }
    ]
})";
    const std::string modelPathToReplace{"/ovms/src/test/passthrough"};
    auto it = this->ovmsConfig.find(modelPathToReplace);
    ASSERT_NE(it, std::string::npos);
    this->ovmsConfig.replace(this->ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), this->modelPath);
    this->configFilePath = this->directoryPath + "/ovms_config.json";
    createConfigFileWithContent(this->ovmsConfig, this->configFilePath);
    createConfigFileWithContent(R"({
        "outputs": {"copy:0": "copy:0_string"}
    })",
        this->mappingConfigPath);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(this->configFilePath);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    typename TypeParam::first_type request;
    prepareInferStringRequest(request, PASSTHROUGH_MODEL_INPUT_NAME, {"String_123", "", "zebra"});
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(manager.getModelInstance("passhtrough_u8", 1, modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    assertStringResponse(response, {"String_123", "", "zebra"}, "copy:0_string");
}

TYPED_TEST(TestPredictWithMapping, SuccesfullOnDummyModelWithMappingSpecificShapeAuto) {
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type request;
    preparer.preparePredictRequest(request,
        {{this->dummyModelInputMapping,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 5}, ovms::Precision::FP32}}});
    this->SetUp(oneDummyWithMappedInputSpecificAutoShapeConfig);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    auto status = config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    status = manager.loadConfig(this->configFilePath);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    performPrediction(config.getName(), config.getVersion(), request, nullptr, nullptr, manager, this->dummyModelInputMapping, this->dummyModelOutputMapping);
}
TYPED_TEST(TestPredictWithMapping, SuccesfullOnDummyModelWithMappingAnonymousShapeAuto) {
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type request;
    preparer.preparePredictRequest(request,
        {{this->dummyModelInputMapping,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 5}, ovms::Precision::FP32}}});
    this->SetUp(oneDummyWithMappedInputAnonymousAutoShapeConfig);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    auto status = config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    status = manager.loadConfig(this->configFilePath);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    performPrediction(config.getName(), config.getVersion(), request, nullptr, nullptr, manager, this->dummyModelInputMapping, this->dummyModelOutputMapping);
}

TYPED_TEST(TestPredict, SuccesfullReloadFromAlreadyLoadedWithNewBatchSize) {
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type request;
    preparer.preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const auto initialBatchSize = config.getBatchSize();
    config.setBatchSize(initialBatchSize);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    this->performPredict(config.getName(), config.getVersion(), request);
}

TYPED_TEST(TestPredict, SuccesfullReloadWhen1InferenceInProgress) {
    //  FIRST LOAD MODEL WITH BS=1
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type requestBs1;
    preparer.preparePredictRequest(requestBs1,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    typename TypeParam::first_type requestBs2;
    preparer.preparePredictRequest(requestBs2,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 10}, ovms::Precision::FP32}}});

    this->config.setBatchingParams("auto");
    this->config.setNireq(2);
    ASSERT_EQ(this->manager.reloadModelWithVersions(this->config), ovms::StatusCode::OK_RELOADED);

    std::promise<void> releaseWaitBeforePerformInferenceBs1, releaseWaitBeforeGetModelInstanceBs2;
    std::thread t1(
        [this, &requestBs1, &releaseWaitBeforePerformInferenceBs1]() {
            this->performPredict(this->config.getName(), this->config.getVersion(), requestBs1, nullptr,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInferenceBs1.get_future())));
        });
    std::thread t2(
        [this, &requestBs2, &releaseWaitBeforeGetModelInstanceBs2]() {
            this->performPredict(this->config.getName(), this->config.getVersion(), requestBs2,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGetModelInstanceBs2.get_future())),
                nullptr);
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    releaseWaitBeforePerformInferenceBs1.set_value();
    releaseWaitBeforeGetModelInstanceBs2.set_value();
    t1.join();
    t2.join();
}

TYPED_TEST(TestPredict, SuccesfullReloadWhen1InferenceAboutToStart) {
    //  FIRST LOAD MODEL WITH BS=1
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type requestBs2;
    preparer.preparePredictRequest(requestBs2,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 10}, ovms::Precision::FP32}}});
    typename TypeParam::first_type requestBs1;
    preparer.preparePredictRequest(requestBs1,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});

    this->config.setBatchingParams("auto");
    this->config.setNireq(2);
    ASSERT_EQ(this->manager.reloadModelWithVersions(this->config), ovms::StatusCode::OK_RELOADED);

    std::promise<void> releaseWaitBeforeGetModelInstanceBs1, releaseWaitBeforePerformInferenceBs2;
    std::thread t1(
        [this, &requestBs1, &releaseWaitBeforeGetModelInstanceBs1]() {
            this->performPredict(this->config.getName(), this->config.getVersion(), requestBs1,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforeGetModelInstanceBs1.get_future())),
                nullptr);
        });
    std::thread t2(
        [this, &requestBs2, &releaseWaitBeforePerformInferenceBs2]() {
            this->performPredict(this->config.getName(), this->config.getVersion(), requestBs2, nullptr,
                std::move(std::make_unique<std::future<void>>(releaseWaitBeforePerformInferenceBs2.get_future())));
        });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    releaseWaitBeforePerformInferenceBs2.set_value();
    releaseWaitBeforeGetModelInstanceBs1.set_value();
    t1.join();
    t2.join();
}

TYPED_TEST(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeGettingModelInstance) {
    const int initialBatchSize = 1;
    this->config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 0;
    const uint waitingBeforeGettingModelCount = 9;
    this->testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TYPED_TEST(TestPredict, SuccesfullReloadWhenSeveralInferRequestJustBeforeInference) {
    const int initialBatchSize = 1;
    this->config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 0;
    this->testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TYPED_TEST(TestPredict, SuccesfullReloadWhenSeveralInferRequestAtDifferentStages) {
    const int initialBatchSize = 1;
    this->config.setBatchingParams("auto");

    const uint waitingBeforePerformInferenceCount = 9;
    const uint waitingBeforeGettingModelCount = 9;
    this->testConcurrentPredicts(initialBatchSize, waitingBeforePerformInferenceCount, waitingBeforeGettingModelCount);
}

TYPED_TEST(TestPredict, SuccesfullReloadForMultipleThreadsDifferentBS) {
    const int initialBatchSize = 2;
    this->config.setBatchingParams("auto");

    const uint numberOfThreads = 5;
    this->testConcurrentBsChanges(initialBatchSize, numberOfThreads);
}
TYPED_TEST(TestPredict, SuccesfullReshapeViaRequestOnDummyModel) {
    // Prepare model this->manager with dynamic shaped dummy model, originally loaded with 1x10 shape
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Prepare request with 1x5 shape, expect reshape
    Preparer<typename TypeParam::first_type> preparer;
    typename TypeParam::first_type request;
    preparer.preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 5}, ovms::Precision::FP32}}});

    typename TypeParam::second_type response;

    // Do the inference
    auto status = this->performInferenceWithRequest(request, response, "dummy");
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // Expect reshape to 1x5
    this->checkOutputShape(response, {1, 5}, DUMMY_MODEL_OUTPUT_NAME);
}

/*
 * Scenario - perform inferences with different shapes and model reload via this->config.json change
 *
 * 1. Load model with shape=auto, initial internal shape (1,10)
 * 2. Do the inference with (1,12) shape - expect status OK and result (1,12)
 * 3. Reshape model to fixed=(1,11) with this->config.json change
 * 4. Do the inference with (1,12) shape - expect status INVALID_SHAPE
 * 5. Do the inference with (1,11) shape - expect status OK and result (1,11)
 * 6. Reshape model back to shape=auto, initial internal shape (1,10)
 * 7. Do the inference with (1,12) shape - expect status OK and result (1,12)
 *
 */
TYPED_TEST(TestPredict, ReshapeViaRequestAndConfigChange) {
    using namespace ovms;

    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform reshape to (1,12) using request
    ASSERT_EQ(this->performInferenceWithShape(response, {1, 12}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 12});

    // Reshape with model reload to Fixed=(1,11)
    config.setBatchingParams("0");
    config.parseShapeParameter("(1,11)");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Cannot do the inference with (1,12)
    ASSERT_EQ(this->performInferenceWithShape(response, {1, 12}), ovms::StatusCode::INVALID_SHAPE);

    // Successfull inference with (1,11)
    ASSERT_EQ(this->performInferenceWithShape(response, {1, 11}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 11});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("0");
    config.parseShapeParameter("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform reshape to (1,12) using request
    ASSERT_EQ(this->performInferenceWithShape(response, {1, 12}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 12});
}
/*
 * Scenario - perform inferences with different batch size and model reload via this->config.json change
 *
 * 1. Load model with bs=auto, initial internal shape (1,10)
 * 2. Do the inference with (3,10) shape - expect status OK and result (3,10)
 * 3. Change model batch size to fixed=4 with this->config.json change
 * 4. Do the inference with (3,10) shape - expect status INVALID_BATCH_SIZE
 * 5. Do the inference with (4,10) shape - expect status OK and result (4,10)
 * 6. Reshape model back to batchsize=auto, initial internal shape (1,10)
 * 7. Do the inference with (3,10) shape - expect status OK and result (3,10)
 */
TYPED_TEST(TestPredict, ChangeBatchSizeViaRequestAndConfigChange) {
    using namespace ovms;
    // Prepare model with shape=auto (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    this->config.setBatchingParams("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(this->config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform batch size change to 3 using request
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 3), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 10});

    // Change batch size with model reload to Fixed=4
    this->config.setBatchingParams("4");
    ASSERT_EQ(this->manager.reloadModelWithVersions(this->config), ovms::StatusCode::OK_RELOADED);

    // Cannot do the inference with (3,10)
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 3), ovms::StatusCode::INVALID_BATCH_SIZE);

    // Successfull inference with (4,10)
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 4), ovms::StatusCode::OK);
    this->checkOutputShape(response, {4, 10});

    // Reshape back to AUTO, internal shape is (1,10)
    this->config.setBatchingParams("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(this->config), ovms::StatusCode::OK_RELOADED);

    // Perform batch change to 3 using request
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 3), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 10});
}

/*
 * Scenario - perform inference with NHWC input layout changed via this->config.json.
 *
 * 1. Load model with layout=nhwc:nchw, initial internal layout: nchw, initial shape=(1,3,4,5)
 * 2. Do the inference with (1,4,5,3) shape - expect status OK and result (1,3,4,5)
 * 3. Do the inference with (1,3,4,5) shape - expect INVALID_SHAPE
 * 4. Remove layout setting
 * 5. Do the inference with (1,3,4,5) shape - expect status OK and result (1,3,4,5)
 * 6. Do the inference with (1,4,5,3) shape - expect INVALID_SHAPE
 * 7. Adding layout setting to nchw
 * 8. Do the inference with (1,3,4,5) shape - expect status OK and result (1,3,4,5)
 * 9. Do the inference with (1,4,5,3) shape - expect INVALID_SHAPE
 */
TYPED_TEST(TestPredict, PerformInferenceChangeModelInputLayout) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with NHWC layout, ensure status OK and correct results
    auto status = this->performInferenceWithImageInput(response, {1, 4, 5, 3});
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    // Perform inference with NCHW layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::INVALID_SHAPE);

    // Reload model with layout setting removed, model is back to NCHW
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NHWC layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 4, 5, 3}), ovms::StatusCode::INVALID_SHAPE);

    // Prepare model with layout changed back to nchw
    ASSERT_EQ(config.parseLayoutParameter("nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure OK
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NHWC layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 4, 5, 3}), ovms::StatusCode::INVALID_SHAPE);
}
/*
 * Scenario - perform inference with NHWC input layout changed and shape changed via this->config.json.
 *
 * 1. Load model with layout=nchw:nhwc and shape=(1,1,2,3), initial internal layout: nchw, initial shape=(1,3,4,5)
 * 2. Do the inference with (1,1,2,3) shape - expect status OK and result (1,3,1,2)
 * 3. Do the inference with (1,3,1,2) shape - expect INVALID_SHAPE
 * 4. Remove layout setting
 * 5. Do the inference with (1,1,2,3) shape - expect status OK and result (1,3,1,2)
 * 6. Do the inference with (1,3,1,2) shape - expect INVALID_SHAPE
 * 7. Adding layout setting to nchw
 * 8. Do the inference with (1,3,1,2) shape - expect status OK and result (1,3,1,2)
 * 9. Do the inference with (1,1,2,3) shape - expect INVALID_SHAPE
 */
TYPED_TEST(TestPredict, PerformInferenceChangeModelInputLayoutAndShape) {
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with NHWC layout, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 5.0, 3.0, 6.0, 4.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NCHW layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);

    // Reload model with layout setting removed, model is back to NCHW
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NHWC layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);

    // Prepare model with layout changed back to nchw
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure OK
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Perform inference with NHWC layout, ensure error
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::INVALID_SHAPE);
}

/**
 * Scenario - change output layout of model and perform inference.
 *
 * 1. Load model with output layout=nhwc:nchw, initial internal layout: nchw
 * 2. Do the inference with (1,3,4,5) shape - expect status OK and result in NHWC layout
 * 3. Remove layout setting
 * 4. Do the inference with (1,3,4,5) shape - expect status OK and result in NCHW layout
 * 5. Roll back layout setting to internal nchw
 * 6. Do the inference with (1,3,4,5) shape - expect status OK and result in NCHW layout
 */
TYPED_TEST(TestPredict, PerformInferenceChangeModelOutputLayout) {
    using namespace ovms;

    // Prepare model with changed output layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nhwc:nchw\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with NCHW layout, ensure status OK and results in NHWC order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 4, 5, 3}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and results still in NHWC order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Change output layout back to original nchw.
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nchw\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 4, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/*
 * Scenario - change output layout of model, modify shape and perform inference. Check results if in correct order.
 *
 * 1. Load model with output layout=nhwc:nchw, shape (1,1,2,3) initial internal layout: nchw
 * 2. Do the inference with (1,3,4,5) shape - expect status OK and result in NHWC layout
 * 3. Remove layout setting
 * 4. Do the inference with (1,3,4,5) shape - expect status OK and result in NCHW layout
 * 5. Roll back layout setting to internal nchw
 * 6. Do the inference with (1,3,4,5) shape - expect status OK and result in NCHW layout
 */
TYPED_TEST(TestPredict, PerformInferenceChangeModelOutputLayoutAndShape) {
    using namespace ovms;

    // Prepare model with changed output layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nhwc:nchw\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with NCHW layout, ensure status OK and results in NHWC order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 1, 2, 3}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 4.0, 6.0, 3.0, 5.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter(""), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NCHW layout, ensure status OK and results still in NHWC order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Change output layout back to original nchw.
    ASSERT_EQ(config.parseLayoutParameter(std::string("{\"") + INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME + std::string("\":\"nchw\"}")), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 3, 1, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/* Scenario - change input layout and changing batch size at runtime. Expect shape dimension order to stay the same.
 *
 * 1. Load model with output layout=nhwc:nchw, native unchanged shape (1,4,5,3) initial internal layout: nchw
 * 2. Do the inference with (1,4,5,3) shape - expect status OK and result in NCHW layout
 * 3. Change batch size setting to 10
 * 4. Do the inference with (10,4,5,3) shape - expect status OK and result in NCHW layout
 * 5. Change batch size setting to 15
 * 6. Do the inference with (15,4,5,3) shape - expect status OK and result in NCHW layout
 * */
TYPED_TEST(TestPredict, PerformInferenceChangeModelLayoutAndKeepChangingBatchSize) {
    using namespace ovms;

    // Prepare model with changed output layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with NHWC layout, ensure status OK and results in NHWC order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {1, 4, 5, 3}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Reload model with batch size changed
    config.setBatchingParams("10");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NHWC layout batch=10, ensure status OK and results still in NCHW order
    ASSERT_EQ(this->performInferenceWithImageInput(response, {10, 4, 5, 3}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {10, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Change bs to 15
    config.setBatchingParams("15");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with NHWC layout batch=15, ensure status OK and input/output layout has not changed
    ASSERT_EQ(this->performInferenceWithImageInput(response, {15, 4, 5, 3}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {15, 3, 4, 5}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

TYPED_TEST(TestPredict, ErrorWhenLayoutSetForMissingTensor) {
    ovms::ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("{\"invalid_tensor_name\":\"nhwc\"}"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::CONFIG_LAYOUT_IS_NOT_IN_MODEL);
}

TYPED_TEST(TestPredict, NetworkNotLoadedWhenLayoutAndDimsInconsistent) {
    // Dummy has 2 dimensions: (1,10), changing layout to NHWC should fail
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::MODEL_NOT_LOADED);
}

/* Scenario - change input layout of model and perform inference with binary input. Check results.
 *
 * 1. Load model with input layout=nhwc, initial internal layout: nchw
 * 2. Do the inference with single binary image tensor - expect status OK and result in NCHW layout
 * 3. Set layout setting to internal nchw
 * 4. Do the inference with single binary image tensor - expect status UNSUPPORTED_LAYOUT
 * 5. Set back layout setting to nhwc
 * 6. Do the inference with single binary image tensor - expect status OK and result in NCHW layout
 * */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputChangeModelInputLayout) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";

    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    // Reload model with layout setting removed
    ASSERT_EQ(config.parseLayoutParameter("nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with binary input, ensure validation rejects the request due to NCHW setting
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::UNSUPPORTED_LAYOUT);

    // Switch back to nhwc
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseShapeParameter("(1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform inference with binary input, ensure status OK after switching layout and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/* Scenario - perform inference with binary input with witdth exceeding shape range when model shape is dynamic. Check results.
 *
 * 1. Load model with dynamic shape and input layout=nhwc, initial internal layout: nchw
 * 2. Do the inference with single binary image tensor with witdth exceeding shape range - expect status OK and reshaped output tensor
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputAndShapeDynamic) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    // binary input shape is [1,1,1,3] so it should be resized to the nearest border which is in this case [1,1,2,3]
    ASSERT_EQ(config.parseShapeParameter("(1,1,2:5,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5"), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}
/*
 * Scenario - send binary input request to model accepting auto batch size.
 *
 * 1. Load model with input layout=nhwc, batch_size=auto, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with batch=5 binary image tensor - expect status OK and result in NCHW layout
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputBatchSizeAuto) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(config.parseShapeParameter("(1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    const int batchSize = 5;
    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5", batchSize), ovms::StatusCode::OK);
    this->checkOutputShape(response, {5, 3, 1, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0, 37.0, 37.0, 28.0, 28.0, 238.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/* Scenario - send binary input request with no shape set.
 *
 * 1. Load model with input layout=nhwc, batch_size=auto, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with binary image tensor with no shape set - expect status INVALID_NO_OF_SHAPE_DIMENSIONS
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputNoInputShape) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(config.parseShapeParameter("(1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::first_type request;
    typename TypeParam::second_type response;
    prepareBinaryPredictRequestNoShape(request, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, 1);

    // Perform inference with binary input, ensure status INVALID_NO_OF_SHAPE_DIMENSIONS
    ASSERT_EQ(this->performInferenceWithRequest(request, response, "increment_1x3x4x5"), ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

/*
 * Scenario - perform inference with with batch size set to auto and batch size not matching on position other than first
 *
 * 1. Load model with bs=auto, layout=b=>cn,a=>cn initial internal shape (1,10)
 * 2. Do the inference with (1,30) shape - expect status OK and result (1,30)
 * 3. Change model batch size to fixed=4 with config.json change
 * 4. Do the inference with (1,30) shape - expect status INVALID_BATCH_SIZE
 * 5. Do the inference with (1,4) shape - expect status OK and result (1,4)
 * 6. Reshape model back to batchsize=auto, initial internal shape (1,10)
 * 7. Do the inference with (1,30) shape - expect status OK and result (1,30)
 * 8. Do the inference with (30,10) shape - expect status INVALID_SHAPE
 */
TYPED_TEST(TestPredict, ChangeBatchSizeViaRequestAndConfigChangeArbitraryPosition) {
    using namespace ovms;
    size_t batchSizePosition = 1;  //  [0:C, 1:N]

    // Prepare model with bs=auto, layout=b=>cn,a=>cn (initially (1,10) shape)
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(config.parseLayoutParameter("{\"b\":\"cn\",\"a\":\"cn\"}"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform batch size change to 30 using request
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 30, ovms::Precision::FP32, batchSizePosition), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 30});

    // Change batch size with model reload to Fixed=4
    config.setBatchingParams("4");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Cannot do the inference with (1,30)
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 30, ovms::Precision::FP32, batchSizePosition), ovms::StatusCode::INVALID_BATCH_SIZE);

    // Successfull inference with (1,4)
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 4, ovms::Precision::FP32, batchSizePosition), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 4});

    // Reshape back to AUTO, internal shape is (1,10)
    config.setBatchingParams("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Perform batch change to 30 using request
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 30, ovms::Precision::FP32, batchSizePosition), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 30});

    // Ensure cannot change batch size with first dimension
    batchSizePosition = 0;
    ASSERT_EQ(this->performInferenceWithBatchSize(response, 30, ovms::Precision::FP32, batchSizePosition), ovms::StatusCode::INVALID_SHAPE);
}

/* Scenario - inference with different shapes with dynamic dummy, both dimensions reshaped to any.
 * No model reload performed between requests.
 *
 * 1. Load model with input shape (-1, -1)
 * 2. Do the inference with (3, 2) shape, expect correct output shape
 * 3. Do the inference with (1, 4) shape, expect correct output shape
 */
TYPED_TEST(TestPredict, PerformInferenceDummyAllDimensionsAny) {
    using namespace ovms;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(-1,-1)"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Do the inference with (3, 2)
    ASSERT_EQ(this->performInferenceWithShape(response, {3, 2}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 2}, DUMMY_MODEL_OUTPUT_NAME);

    // Do the inference with (1, 4)
    ASSERT_EQ(this->performInferenceWithShape(response, {1, 4}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 4}, DUMMY_MODEL_OUTPUT_NAME);
}

/* Scenario - inference with different batch sizes for dynamic batch dummy.
 * No model reload performed between requests.
 *
 * 1. Load model with input shape (-1, 10)
 * 2. Do the X inferences with (x, 10) shape, expect correct output shapes. x=[1, 3, 5, 7, 11, 17, 21, 57, 99]
 */
TYPED_TEST(TestPredict, PerformInferenceDummyBatchSizeAny) {
    using namespace ovms;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("-1");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    for (size_t i : {1, 3, 5, 7, 11, 17, 21, 57, 99}) {
        ASSERT_EQ(this->performInferenceWithShape(response, {i, 10}), ovms::StatusCode::OK);
        this->checkOutputShape(response, {i, 10}, DUMMY_MODEL_OUTPUT_NAME);
    }
}

/* Scenario - inference with dummy precision fp32.
 *
 * 1. Load model with input shape (-1, 10)
 * 2. Do the inferences with (3, 10) shape, expect correct output shapes and precision
 */

static ovms::Precision getPrecisionFromResponse(ovms::InferenceResponse& response, const std::string& name) {
    size_t outputCount = response.getOutputCount();
    EXPECT_GE(1, outputCount);
    size_t outputId = 0;
    while (outputId < outputCount) {
        const std::string* cppName;
        InferenceTensor* tensor;
        auto status = response.getOutput(outputId, &cppName, &tensor);
        EXPECT_EQ(status, StatusCode::OK) << status.string();
        EXPECT_NE(nullptr, tensor);
        EXPECT_NE(nullptr, cppName);
        if (name == *cppName) {
            return ovms::getOVMSDataTypeAsPrecision(tensor->getDataType());
        }
        ++outputId;
    }
    return ovms::getOVMSDataTypeAsPrecision(OVMS_DATATYPE_UNDEFINED);
}

static ovms::Precision getPrecisionFromResponse(KFSResponse& response, const std::string& name) {
    KFSOutputTensorIteratorType it;
    size_t bufferId;
    auto status = getOutput(response, name, it, bufferId);
    EXPECT_TRUE(status.ok());
    return ovms::KFSPrecisionToOvmsPrecision(it->datatype());
}

static ovms::Precision getPrecisionFromResponse(TFSResponseType& response, const std::string& name) {
    TFSOutputTensorIteratorType it;
    size_t bufferId;
    auto status = getOutput(response, name, it, bufferId);
    EXPECT_TRUE(status.ok());
    if (!status.ok())
        return ovms::Precision::UNDEFINED;
    return ovms::TFSPrecisionToOvmsPrecision(it->second.dtype());
}
TYPED_TEST(TestPredict, PerformInferenceDummyFp64) {
    using namespace ovms;

    ModelConfig config = DUMMY_FP64_MODEL_CONFIG;
    config.setBatchingParams("3");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::first_type request;
    typename TypeParam::second_type response;

    Preparer<typename TypeParam::first_type> preparer;
    preparer.preparePredictRequest(request, {{"input:0", std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 10}, ovms::Precision::FP64}}});
    ASSERT_EQ(this->performInferenceWithRequest(request, response, "dummy_fp64"), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 10}, "output:0");
    ASSERT_EQ(getPrecisionFromResponse(response, "output:0"), ovms::Precision::FP64);
}

/* Scenario - inference with different shapes with dynamic dummy, both dimensions reshaped to range.
 * No model reload performed between requests.
 *
 * 1. Load model with input shape (2:4, 1:5)
 * 2. Do the inference with (1, 1) shape, expect not in range
 * 3. Do the inference with (2, 1) shape, expect success and correct output shape
 * 4. Do the inference with (3, 2) shape, expect success and correct output shape
 * 5. Do the inference with (3, 5) shape, expect success and correct output shape
 * 6. Do the inference with (3, 6) shape, expect not in range
 * 7. Do the inference with (5, 5) shape, expect not in range
 */
TYPED_TEST(TestPredict, PerformInferenceDummyAllDimensionsHaveRange) {
    using namespace ovms;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(2:4,1:5)"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    ASSERT_EQ(this->performInferenceWithShape(response, {1, 1}), ovms::StatusCode::INVALID_BATCH_SIZE);

    ASSERT_EQ(this->performInferenceWithShape(response, {2, 1}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {2, 1}, DUMMY_MODEL_OUTPUT_NAME);

    ASSERT_EQ(this->performInferenceWithShape(response, {3, 2}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 2}, DUMMY_MODEL_OUTPUT_NAME);

    ASSERT_EQ(this->performInferenceWithShape(response, {3, 5}), ovms::StatusCode::OK);
    this->checkOutputShape(response, {3, 5}, DUMMY_MODEL_OUTPUT_NAME);

    ASSERT_EQ(this->performInferenceWithShape(response, {3, 6}), ovms::StatusCode::INVALID_SHAPE);
    ASSERT_EQ(this->performInferenceWithShape(response, {5, 5}), ovms::StatusCode::INVALID_BATCH_SIZE);
}

/* Scenario - send binary input request to model accepting dynamic batch size.
 *
 * 1. Load model with input layout=nhwc, batch_size=-1, resolution 1x2, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with batch=5 binary image tensor 1x1 - expect status INVALID_SHAPE, because if any dimension is dynamic, we perform no resize operation.
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputBatchSizeAnyResolutionNotMatching) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(-1,1,2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    const int batchSize = 5;
    // Perform inference with binary input 1x1, expect status BINARY_IMAGES_RESOLUTION_MISMATCH, because if any dimension is dynamic, we perform no resize operation.
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5", batchSize), ovms::StatusCode::INVALID_SHAPE);
}

/* Scenario - send binary input request to model accepting dynamic batch size.
 *
 * 1. Load model with input layout=nhwc, batch_size=-1, resolution 1x1, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with batch=5 binary image tensor 1x1 - expect status OK, and correct results.
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputBatchSizeAnyResolutionMatching) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(-1,1,1,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    const int batchSize = 5;
    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5", batchSize), ovms::StatusCode::OK);
    this->checkOutputShape(response, {5, 3, 1, 1}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/* Scenario - send binary input request to model accepting dynamic resolution.
 *
 * 1. Load model with input layout=nhwc, shape 1,-1,-1,3, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with resolution 1x1 binary image tensor - expect status OK and result in NCHW layout
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputResolutionAny) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,-1,-1,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;

    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5"), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 1}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 28.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

/* Scenario - send binary input request to model accepting range of resolution.
 *
 * 1. Load model with input layout=nhwc, shape 1,1:2,1:2,3, initial internal layout: nchw, batch_size=1
 * 2. Do the inference with resolution 4x4 binary image tensor - expect status OK and reshaped to 2x2
 * 3. Do the inference with resolution 1x1 binary image tensor - expect status OK and result in NCHW layout
 */
TYPED_TEST(TestPredict, PerformInferenceWithBinaryInputResolutionRange) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "Binary inputs not implemented for C-API yet";
    using namespace ovms;

    // Prepare model with changed layout to nhwc (internal layout=nchw)
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,1:2,1:2,3)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc:nchw"), ovms::StatusCode::OK);
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    typename TypeParam::second_type response;
    typename TypeParam::first_type request;
    prepareBinary4x4PredictRequest(request, INCREMENT_1x3x4x5_MODEL_INPUT_NAME);

    ASSERT_EQ(
        this->performInferenceWithRequest(
            request, response, "increment_1x3x4x5"),
        ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 2, 2}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);

    response.Clear();

    // Perform inference with binary input, ensure status OK and correct results
    ASSERT_EQ(this->performInferenceWithBinaryImageInput(response, INCREMENT_1x3x4x5_MODEL_INPUT_NAME, "increment_1x3x4x5"), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 3, 1, 1}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
    this->checkOutputValues(response, {37.0, 28.0, 238.0}, INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME);
}

TYPED_TEST(TestPredict, InferenceWithNegativeShape) {
    typename TypeParam::first_type request;
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int64_t negativeBatch = -5;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{negativeBatch, 10}, ovms::Precision::FP32}}},
        data);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);

    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_NE(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
}

TYPED_TEST(TestPredict, InferenceWithNegativeShapeDynamicParameter) {
    typename TypeParam::first_type request;
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int64_t negativeBatch = -5;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{negativeBatch, 10}, ovms::Precision::FP32}}},
        data);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("auto");

    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_NE(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
}

TYPED_TEST(TestPredict, InferenceWithStringInputs_positive_2D) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "String inputs not supported for C-API";
    typename TypeParam::first_type request;
    std::vector<std::string> inputStrings = {"String_123", "String"};
    prepareInferStringRequest(request, PASSTHROUGH_MODEL_INPUT_NAME, inputStrings);
    ovms::ModelConfig config = PASSTHROUGH_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    this->checkOutputShape(response, {2, 11}, PASSTHROUGH_MODEL_OUTPUT_NAME);
    std::vector<uint8_t> expectedData = {
        'S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,
        'S', 't', 'r', 'i', 'n', 'g', 0, 0, 0, 0, 0};
    bool checkRaw = true;
    this->checkOutputValuesU8(response, expectedData, PASSTHROUGH_MODEL_OUTPUT_NAME, checkRaw);
}

TYPED_TEST(TestPredict, InferenceWithStringInputs_positive_2D_data_in_buffer) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest) || typeid(typename TypeParam::first_type) == typeid(TFSRequestType))
        GTEST_SKIP() << "String inputs in buffer not supported for C-API and TFS api";
    typename TypeParam::first_type request;
    std::vector<std::string> inputStrings = {"String_123", "String"};
    prepareInferStringRequest(request, PASSTHROUGH_MODEL_INPUT_NAME, inputStrings, false);
    ovms::ModelConfig config = PASSTHROUGH_MODEL_CONFIG;
    config.setBatchingParams("auto");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    this->checkOutputShape(response, {2, 11}, PASSTHROUGH_MODEL_OUTPUT_NAME);
    std::vector<uint8_t> expectedData = {
        'S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,
        'S', 't', 'r', 'i', 'n', 'g', 0, 0, 0, 0, 0};
    bool checkRaw = true;
    this->checkOutputValuesU8(response, expectedData, PASSTHROUGH_MODEL_OUTPUT_NAME, checkRaw);
}

TYPED_TEST(TestPredict, InferenceWithStringInputs_positive_1D) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest))
        GTEST_SKIP() << "String inputs not supported for C-API";
    typename TypeParam::first_type request;
    std::vector<std::string> inputStrings = {"ala", "", "ma", "kota"};
    prepareInferStringRequest(request, PASSTHROUGH_MODEL_INPUT_NAME, inputStrings);
    ovms::ModelConfig config = PASSTHROUGH_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("(-1)");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 33}, PASSTHROUGH_MODEL_OUTPUT_NAME);
    std::vector<uint8_t> expectedData = {
        4, 0, 0, 0,  // batch size
        0, 0, 0, 0,  // first string start offset
        3, 0, 0, 0,  // end of "ala" in condensed content
        3, 0, 0, 0,  // end of "" in condensed content
        5, 0, 0, 0,  // end of "ma" in condensed content
        9, 0, 0, 0,  // end of "kota" in condensed content
        'a', 'l', 'a',
        'm', 'a',
        'k', 'o', 't', 'a'};
    bool checkRaw = true;
    this->checkOutputValuesU8(response, expectedData, PASSTHROUGH_MODEL_OUTPUT_NAME, checkRaw);
}

TYPED_TEST(TestPredict, InferenceWithStringInputs_positive_1D_data_in_buffer) {
    if (typeid(typename TypeParam::first_type) == typeid(ovms::InferenceRequest) || typeid(typename TypeParam::first_type) == typeid(TFSRequestType))
        GTEST_SKIP() << "String inputs in buffer not supported for C-API and TFS api";
    typename TypeParam::first_type request;
    std::vector<std::string> inputStrings = {"ala", "", "ma", "kota"};
    prepareInferStringRequest(request, PASSTHROUGH_MODEL_INPUT_NAME, inputStrings, false);
    ovms::ModelConfig config = PASSTHROUGH_MODEL_CONFIG;
    config.setBatchingParams("0");
    config.parseShapeParameter("(-1)");
    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    typename TypeParam::second_type response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    this->checkOutputShape(response, {1, 33}, PASSTHROUGH_MODEL_OUTPUT_NAME);
    std::vector<uint8_t> expectedData = {
        4, 0, 0, 0,  // batch size
        0, 0, 0, 0,  // first string start offset
        3, 0, 0, 0,  // end of "ala" in condensed content
        3, 0, 0, 0,  // end of "" in condensed content
        5, 0, 0, 0,  // end of "ma" in condensed content
        9, 0, 0, 0,  // end of "kota" in condensed content
        'a', 'l', 'a',
        'm', 'a',
        'k', 'o', 't', 'a'};
    bool checkRaw = true;
    this->checkOutputValuesU8(response, expectedData, PASSTHROUGH_MODEL_OUTPUT_NAME, checkRaw);
}

class TestPredictKFS : public TestPredict<KFSInterface> {};

TEST_F(TestPredictKFS, RequestDataInFp32ContentResponseInRaw) {
    KFSRequest request;
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    bool putBufferInInputTensorContent = true;  // put in fp32_content
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}},
        data,
        putBufferInInputTensorContent);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;

    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    KFSResponse response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_FALSE(response.outputs(0).has_contents());
    ASSERT_GT(response.raw_output_contents_size(), 0);
}

TEST_F(TestPredictKFS, RequestDataInRawResponseInRaw) {
    KFSRequest request;
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    bool putBufferInInputTensorContent = false;  // put in raw
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}},
        data,
        putBufferInInputTensorContent);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;

    ASSERT_EQ(this->manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    ASSERT_EQ(this->manager.getModelInstance(config.getName(), config.getVersion(), modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    KFSResponse response;
    ASSERT_EQ(modelInstance->infer(&request, &response, modelInstanceUnloadGuard), ovms::StatusCode::OK);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_FALSE(response.outputs(0).has_contents());
    ASSERT_GT(response.raw_output_contents_size(), 0);
}

#pragma GCC diagnostic pop
