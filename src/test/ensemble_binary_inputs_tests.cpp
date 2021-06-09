//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../binaryutils.hpp"
#include "../dl_node.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "../pipelinedefinition.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

const uint NIREQ = 2;

const std::string identity_model_location = std::filesystem::current_path().u8string() + "/src/test/identity";
const ovms::ModelConfig IDENTITY_MODEL_CONFIG{
    "identity",
    identity_model_location,  // base path
    "CPU",                    // target device
    "1",                      // batchsize
    1,                        // NIREQ
    false,                    // is stateful
    true,                     // idle sequence cleanup enabled
    false,                    // low latency transformation enabled
    500,                      // steteful sequence max number
    1,                        // model_version unused since version are read from path
    identity_model_location,  // local path
};

class EnsembleBinaryInputsTest : public TestWithTempDir {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        config = IDENTITY_MODEL_CONFIG;
    }

    void preparePredictRequest(const std::string& filePath, PredictRequest& predictRequest, const std::string& customPipelineInputName) {
        std::ifstream DataFile;
        DataFile.open(filePath, std::ios::binary);
        DataFile.seekg(0, std::ios::end);
        size_t filesize = DataFile.tellg();
        DataFile.seekg(0);
        std::unique_ptr<char[]> image_bytes(new char[filesize]);
        DataFile.read(image_bytes.get(), filesize);

        tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
        inputProto.set_dtype(tensorflow::DataType::DT_STRING);
        inputProto.add_string_val(image_bytes.get(), filesize);
        inputProto.mutable_tensor_shape()->add_dim()->set_size(1);
    }

    void extendPredictRequest(const std::string& filePath, PredictRequest& predictRequest, const std::string& customPipelineInputName) {
        std::ifstream DataFile;
        DataFile.open(filePath, std::ios::binary);
        DataFile.seekg(0, std::ios::end);
        size_t filesize = DataFile.tellg();
        DataFile.seekg(0);
        std::unique_ptr<char[]> image_bytes(new char[filesize]);
        DataFile.read(image_bytes.get(), filesize);

        tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
        inputProto.add_string_val(image_bytes.get(), filesize);
        int currentBatchSize = inputProto.mutable_tensor_shape()->dim(0).size();
        inputProto.mutable_tensor_shape()->mutable_dim(0)->set_size(currentBatchSize++);
    }

    void checkOutputMeta(const TensorProto& outputProto, std::shared_ptr<ovms::TensorInfo> outputInfo) {
        const shape_t expectedShape = outputInfo->getShape();
        const InferenceEngine::Precision expectedPrecision = outputInfo->getPrecision();

        size_t elementsCount = std::accumulate(expectedShape.begin(), expectedShape.end(), 1, std::multiplies<size_t>());
        size_t bytesCount = elementsCount * expectedPrecision.size();

        ASSERT_EQ(outputProto.tensor_content().size(), bytesCount);
        ASSERT_EQ(outputProto.tensor_shape().dim_size(), expectedShape.size());
        for (size_t i = 0; i < expectedShape.size(); i++) {
            ASSERT_EQ(outputProto.tensor_shape().dim(i).size(), expectedShape[i]);
        }
    }

    ModelConfig config;

    std::string identityModelName = "identity";
    const std::string identityModelInputName = "data";
    const std::string identityModelOutputName = identityModelInputName;

    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";
};

TEST_F(EnsembleBinaryInputsTest, Basic_BS1) {
    // Most basic configuration, just process single identity model request
    // input  identity    output
    //  O-------->O-------->O
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 3, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = 1 * 3 * 1 * 1;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, Basic_BS1_NHWC) {
    // Most basic configuration, just process single identity model request
    // input  identity    output
    //  O-------->O-------->O
    config.setBatchingParams("0");
    config.parseShapeParameter("(1,1,1,3)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = 1 * 1 * 1 * 3;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, Basic_BS1_Greyscale) {
    // Process single identity model request on greyscale image
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,1,1,1)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/grayscale.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = 1 * 1 * 1 * 1;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, Basic_BS5) {
    // Most basic configuration, just process single identity model request with batch size = 5
    // input  identity    output
    //  O-------->O-------->O
    const size_t batchSize = 5;
    ConstructorEnabledModelManager managerWithIdentityModel;
    config.setBatchSize(batchSize);
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);
    for (size_t i = 1; i < batchSize; i++)
        extendPredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{batchSize, 3, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = batchSize * 3 * 1 * 1;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, Resize_BS1) {
    // Process single identity model request with resize (1, 3, 1, 1) -> (1, 3, 4, 4)
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,3,4,4)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 3, 4, 4}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = 1 * 3 * 4 * 4;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, Resize_BS5) {
    // Process single identity model request with resize on batch size = 5: (5, 3, 1, 1) -> (5, 3, 4, 4)
    // input  identity    output
    //  O-------->O-------->O

    const size_t batchSize = 5;
    config.setBatchingParams("0");
    config.parseShapeParameter("(5,3,4,4)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);
    for (size_t i = 1; i < batchSize; i++)
        extendPredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{batchSize, 3, 4, 4}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    ASSERT_TRUE(status.ok());

    // Get expected output data
    InferenceEngine::Blob::Ptr convertedInputBlob;
    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    status = convertStringValToBlob(inputProto, convertedInputBlob, tensorInfo);
    ASSERT_TRUE(status.ok());

    InferenceEngine::MemoryBlob::Ptr memoryInputBlob = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(convertedInputBlob);
    float* expectedOutput = (float*)(memoryInputBlob->buffer());

    // Get actual output data
    ASSERT_EQ(predictResponse.outputs().count(customPipelineOutputName), 1);
    const auto& outputProto = predictResponse.outputs().at(customPipelineOutputName);

    checkOutputMeta(outputProto, tensorInfo);

    float* actualOutput = (float*)outputProto.tensor_content().data();

    // Compare pipeline output data with expected output data

    size_t elementsCount = batchSize * 3 * 4 * 4;
    std::vector<float> expectedOutputVector, actualOutputVector;
    expectedOutputVector.assign(expectedOutput, expectedOutput + elementsCount);
    actualOutputVector.assign(actualOutput, actualOutput + elementsCount);
    EXPECT_EQ(expectedOutputVector, actualOutputVector);
}

TEST_F(EnsembleBinaryInputsTest, NotEnoughColorChannels) {
    // Not enought color channels to perform conversion
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,1,1,1)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(EnsembleBinaryInputsTest, UnsupportedLayout) {
    // Wrong input tensor layout
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,3,1)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 3, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(EnsembleBinaryInputsTest, InvalidPrecision) {
    // Invalid precision on input tensor
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,3,1,1)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    preparePredictRequest("/ovms/src/test/binaryutils/rgb.jpg", predictRequest, customPipelineInputName);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::MIXED, shape_t{1, 3, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleBinaryInputsTest, InvalidData) {
    // Invalid precision on input tensor
    // input  identity    output
    //  O-------->O-------->O

    config.setBatchingParams("0");
    config.parseShapeParameter("(1,3,1,1)");
    ConstructorEnabledModelManager managerWithIdentityModel;
    managerWithIdentityModel.reloadModelWithVersions(config);

    PredictRequest predictRequest;
    PredictResponse predictResponse;

    tensorflow::TensorProto& inputProto = (*predictRequest.mutable_inputs())[customPipelineInputName];
    inputProto.set_dtype(tensorflow::DataType::DT_STRING);
    inputProto.add_string_val("INVALID_IMAGE");
    inputProto.mutable_tensor_shape()->add_dim()->set_size(1);

    // Configure pipeline
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, InferenceEngine::Precision::FP32, shape_t{1, 3, 1, 1}, InferenceEngine::Layout::NCHW);
    const tensor_map_t inputsInfo{{customPipelineInputName, tensorInfo}};
    auto inputNode = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    auto modelNode = std::make_unique<DLNode>("identity_node", identityModelName, requestedModelVersion, managerWithIdentityModel);
    auto outputNode = std::make_unique<ExitNode>(&predictResponse);

    Pipeline pipeline(*inputNode, *outputNode);
    pipeline.connect(*inputNode, *modelNode, {{customPipelineInputName, identityModelInputName}});
    pipeline.connect(*modelNode, *outputNode, {{identityModelOutputName, customPipelineOutputName}});

    pipeline.push(std::move(inputNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(outputNode));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::IMAGE_PARSING_FAILED);
}
