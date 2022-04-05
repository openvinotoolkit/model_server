//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <gtest/gtest.h>

#include "../kfs_grpc_inference_service.hpp"
#include "../pipelinedefinition.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

struct Info {
    ovms::Precision precision;
    ovms::shape_t shape;
};
using tensor_desc_map_t = std::unordered_map<std::string, Info>;

class ModelMetadataResponseBuild : public ::testing::Test {
    class MockModelInstance : public MockModelInstanceChangingStates {
    public:
        MockModelInstance(ov::Core& ieCore) :
            MockModelInstanceChangingStates("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {
            status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_MODEL_VERSION, ovms::ModelVersionState::AVAILABLE);
        }

        // Keeps the model in loading state forever
        ovms::Status loadModel(const ovms::ModelConfig& config) override {
            status.setLoading();
            return ovms::StatusCode::OK;
        }

        MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
        MOCK_METHOD(const ovms::tensor_map_t&, getOutputsInfo, (), (const, override));
        MOCK_METHOD(const std::string&, getName, (), (const, override));
        MOCK_METHOD(ovms::model_version_t, getVersion, (), (const, override));
    };

    tensor_desc_map_t inputTensors;
    tensor_desc_map_t outputTensors;
    ovms::tensor_map_t servableInputs;
    ovms::tensor_map_t servableOutputs;

public:
    void prepare(tensor_desc_map_t inTensors, tensor_desc_map_t outTensors) {
        instance = std::make_shared<NiceMock<MockModelInstance>>(*ieCore);

        inputTensors = inTensors;
        outputTensors = outTensors;

        auto prepare = [](const tensor_desc_map_t& desc,
                           ovms::tensor_map_t& tensors) {
            for (const auto& pair : desc) {
                tensors[pair.first] = std::make_shared<ovms::TensorInfo>(
                    pair.first,
                    pair.second.precision,
                    pair.second.shape);
            }
        };

        prepare(inputTensors, servableInputs);
        prepare(outputTensors, servableOutputs);

        ON_CALL(*instance, getInputsInfo())
            .WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getOutputsInfo())
            .WillByDefault(ReturnRef(servableOutputs));
        ON_CALL(*instance, getName())
            .WillByDefault(ReturnRef(modelName));
        ON_CALL(*instance, getVersion())
            .WillByDefault(Return(modelVersion));
    }

    void prepare() {
        prepare(tensor_desc_map_t({{"Input_FP32_1_3_224_224", {
                                                                  ovms::Precision::FP32,
                                                                  {1, 3, 224, 224},
                                                              }},
                    {"Input_U8_1_3_62_62", {
                                               ovms::Precision::U8,
                                               {1, 3, 62, 62},
                                           }}}),
            tensor_desc_map_t({{"Output_I32_1_2000", {
                                                         ovms::Precision::I32,
                                                         {1, 2000},
                                                     }},
                {"Output_FP32_2_20_3", {
                                           ovms::Precision::FP32,
                                           {2, 20, 3},
                                       }}}));
    }

protected:
    std::string modelName = "resnet";
    ovms::model_version_t modelVersion = 23;

    std::shared_ptr<NiceMock<MockModelInstance>> instance;
    ::inference::ModelMetadataResponse response;
    std::unique_ptr<ov::Core> ieCore;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
    }

    void TearDown() override {
        ieCore.reset();
    }
};

TEST_F(ModelMetadataResponseBuild, BasicResponseMetadata) {
    prepare();

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(instance, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.name(), "resnet");

    EXPECT_EQ(response.versions_size(), 1);
    EXPECT_EQ(response.versions().at(0), "23");

    EXPECT_EQ(response.platform(), "OpenVINO");
}

TEST_F(ModelMetadataResponseBuild, ModelVersionNotLoadedAnymore) {
    prepare();
    instance->retireModel();
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(instance, &response), ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

TEST_F(ModelMetadataResponseBuild, ModelVersionNotLoadedYet) {
    prepare();
    instance->loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(instance, &response), ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);
}

TEST_F(ModelMetadataResponseBuild, SingleInputSingleOutputValidResponse) {
    tensor_desc_map_t inputs = tensor_desc_map_t({{"SingleInput", {ovms::Precision::FP32, {1, 3, 224, 224}}}});
    tensor_desc_map_t outputs = tensor_desc_map_t({{"SingleOutput", {ovms::Precision::I32, {1, 2000}}}});
    prepare(inputs, outputs);

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(instance, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.inputs_size(), 1);
    auto input = response.inputs().at(0);
    EXPECT_EQ(input.name(), "SingleInput");
    EXPECT_EQ(input.datatype(), "FP32");
    EXPECT_EQ(input.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(input.shape(), {1, 3, 224, 224}));

    EXPECT_EQ(response.outputs_size(), 1);
    auto output = response.outputs().at(0);
    EXPECT_EQ(output.name(), "SingleOutput");
    EXPECT_EQ(output.datatype(), "I32");
    EXPECT_EQ(output.shape_size(), 2);
    EXPECT_TRUE(isShapeTheSame(output.shape(), {1, 2000}));
}

TEST_F(ModelMetadataResponseBuild, DoubleInputDoubleOutputValidResponse) {
    tensor_desc_map_t inputs = tensor_desc_map_t({{"FirstInput", {ovms::Precision::FP32, {1, 3, 224, 224}}},
        {"SecondInput", {ovms::Precision::U8, {1, 700, 5}}}});
    tensor_desc_map_t outputs = tensor_desc_map_t({{"FirstOutput", {ovms::Precision::I32, {1, 2000}}},
        {"SecondOutput", {ovms::Precision::FP32, {1, 3, 400, 400}}}});
    prepare(inputs, outputs);

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(instance, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.inputs_size(), 2);
    auto firstInput = response.inputs().at(0);
    EXPECT_EQ(firstInput.name(), "FirstInput");
    EXPECT_EQ(firstInput.datatype(), "FP32");
    EXPECT_EQ(firstInput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(firstInput.shape(), {1, 3, 224, 224}));
    auto secondInput = response.inputs().at(1);
    EXPECT_EQ(secondInput.name(), "SecondInput");
    EXPECT_EQ(secondInput.datatype(), "U8");
    EXPECT_EQ(secondInput.shape_size(), 3);
    EXPECT_TRUE(isShapeTheSame(secondInput.shape(), {1, 700, 5}));

    EXPECT_EQ(response.outputs_size(), 2);
    auto firstOutput = response.outputs().at(0);
    EXPECT_EQ(firstOutput.name(), "FirstOutput");
    EXPECT_EQ(firstOutput.datatype(), "I32");
    EXPECT_EQ(firstOutput.shape_size(), 2);
    EXPECT_TRUE(isShapeTheSame(firstOutput.shape(), {1, 2000}));
    auto secondOutput = response.outputs().at(1);
    EXPECT_EQ(secondOutput.name(), "SecondOutput");
    EXPECT_EQ(secondOutput.datatype(), "FP32");
    EXPECT_EQ(secondOutput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(secondOutput.shape(), {1, 3, 400, 400}));
}

class PipelineMetadataResponseBuild : public ::testing::Test {
protected:
    class MockPipelineDefinitionGetInputsOutputsInfo : public ovms::PipelineDefinition {
        ovms::Status status = ovms::StatusCode::OK;

    public:
        MockPipelineDefinitionGetInputsOutputsInfo() :
            PipelineDefinition("pipeline_name", {}, {}) {
            PipelineDefinition::status.handle(ovms::ValidationPassedEvent());
        }

        void mockMetadata(const ovms::tensor_map_t& inputsInfo, const ovms::tensor_map_t& outputsInfo) {
            this->inputsInfo = inputsInfo;
            this->outputsInfo = outputsInfo;
        }

        void mockStatus(ovms::Status status) {
            this->status = status;
        }

        ovms::PipelineDefinitionStatus& getPipelineDefinitionStatus() {
            return ovms::PipelineDefinition::status;
        }
    };

    MockPipelineDefinitionGetInputsOutputsInfo pipelineDefinition;
    ::inference::ModelMetadataResponse response;
    ConstructorEnabledModelManager manager;

public:
    void prepare(const ovms::tensor_map_t& inputsInfo, const ovms::tensor_map_t& outputsInfo) {
        pipelineDefinition.mockMetadata(inputsInfo, outputsInfo);
    }

    void prepare() {
        ovms::tensor_map_t inputsInfo = ovms::tensor_map_t(
            {{"Input_FP32_1_3_224_224", std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224", ovms::Precision::FP32, ovms::Shape{1, 3, 224, 224})},
                {"Input_U8_1_3_62_62", std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62", ovms::Precision::U8, ovms::Shape{1, 3, 62, 62})},
                {"Input_Unspecified", ovms::TensorInfo::getUnspecifiedTensorInfo()}});
        ovms::tensor_map_t outputsInfo = ovms::tensor_map_t(
            {{"Output_I32_1_2000", std::make_shared<ovms::TensorInfo>("Output_I32_1_2000", ovms::Precision::I32, ovms::Shape{1, 2000})},
                {"Output_FP32_2_20_3", std::make_shared<ovms::TensorInfo>("Output_FP32_2_20_3", ovms::Precision::FP32, ovms::Shape{2, 20, 3})},
                {"Output_Unspecified", ovms::TensorInfo::getUnspecifiedTensorInfo()}});
        prepare(inputsInfo, outputsInfo);
    }
};

TEST_F(PipelineMetadataResponseBuild, BasicResponseMetadata) {
    prepare();

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.name(), "pipeline_name");

    EXPECT_EQ(response.versions_size(), 1);
    EXPECT_EQ(response.versions().at(0), "1");

    EXPECT_EQ(response.platform(), "OpenVINO");
}

TEST_F(PipelineMetadataResponseBuild, PipelineInputOutputResponseMetadata) {
    prepare();

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.inputs_size(), 3);
    auto firstInput = response.inputs().at(0);
    EXPECT_EQ(firstInput.name(), "Input_FP32_1_3_224_224");
    EXPECT_EQ(firstInput.datatype(), "FP32");
    EXPECT_EQ(firstInput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(firstInput.shape(), {1, 3, 224, 224}));
    auto secondInput = response.inputs().at(1);
    EXPECT_EQ(secondInput.name(), "Input_U8_1_3_62_62");
    EXPECT_EQ(secondInput.datatype(), "U8");
    EXPECT_EQ(secondInput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(secondInput.shape(), {1, 3, 62, 62}));
    auto thirdInput = response.inputs().at(2);
    EXPECT_EQ(thirdInput.name(), "Input_Unspecified");
    EXPECT_EQ(thirdInput.datatype(), "UNDEFINED");
    EXPECT_EQ(thirdInput.shape_size(), 0);
    EXPECT_TRUE(isShapeTheSame(thirdInput.shape(), {}));

    EXPECT_EQ(response.outputs_size(), 3);
    auto firstOutput = response.outputs().at(1);
    EXPECT_EQ(firstOutput.name(), "Output_I32_1_2000");
    EXPECT_EQ(firstOutput.datatype(), "I32");
    EXPECT_EQ(firstOutput.shape_size(), 2);
    EXPECT_TRUE(isShapeTheSame(firstOutput.shape(), {1, 2000}));
    auto secondOutput = response.outputs().at(0);
    EXPECT_EQ(secondOutput.name(), "Output_FP32_2_20_3");
    EXPECT_EQ(secondOutput.datatype(), "FP32");
    EXPECT_EQ(secondOutput.shape_size(), 3);
    EXPECT_TRUE(isShapeTheSame(secondOutput.shape(), {2, 20, 3}));
    auto thirdOutput = response.outputs().at(2);
    EXPECT_EQ(thirdOutput.name(), "Output_Unspecified");
    EXPECT_EQ(thirdOutput.datatype(), "UNDEFINED");
    EXPECT_EQ(thirdOutput.shape_size(), 0);
    EXPECT_TRUE(isShapeTheSame(thirdOutput.shape(), {}));
}

TEST_F(PipelineMetadataResponseBuild, ModelVersionNotLoadedAnymoreButPipelineNotReloadedYet) {
    pipelineDefinition.mockStatus(ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);
}

TEST_F(PipelineMetadataResponseBuild, ModelVersionNotLoadedYet) {
    pipelineDefinition.mockStatus(ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);
}

TEST_F(PipelineMetadataResponseBuild, PipelineNotLoadedAnymore) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ovms::RetireEvent());
    auto status = ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE) << status.string();
}

TEST_F(PipelineMetadataResponseBuild, PipelineNotLoadedYet) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ovms::ValidationFailedEvent());
    auto status = ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET) << status.string();
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(PipelineMetadataResponseBuild, PipelineAvailableOrAvailableRequiringRevalidation) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);
}

TEST_F(PipelineMetadataResponseBuild, HandleDynamicAndRangeShapes) {
    ovms::tensor_map_t inputsInfo = ovms::tensor_map_t(
        {{"Input_FP32_1_-1_224_224", std::make_shared<ovms::TensorInfo>("Input_FP32_1_-1_224_224", ovms::Precision::FP32, ovms::Shape{1, ovms::Dimension::any(), 224, 224})},
            {"Input_U8_1_3_62:92_62:92", std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62:92_62:92", ovms::Precision::U8, ovms::Shape{1, 3, {62, 92}, {62, 92}})}});
    ovms::tensor_map_t outputsInfo = ovms::tensor_map_t(
        {{"Output_I32_1_-1", std::make_shared<ovms::TensorInfo>("Output_I32_1_-1", ovms::Precision::I32, ovms::Shape{1, ovms::Dimension::any()})},
            {"Output_FP32_1_224:294_224:294_3", std::make_shared<ovms::TensorInfo>("Output_FP32_1_224:294_224:294_3", ovms::Precision::FP32, ovms::Shape{1, {224, 294}, {224, 294}, 3})}});
    prepare(inputsInfo, outputsInfo);

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(pipelineDefinition, &response), ovms::StatusCode::OK);

    EXPECT_EQ(response.inputs_size(), 2);
    auto firstInput = response.inputs().at(0);
    EXPECT_EQ(firstInput.name(), "Input_FP32_1_-1_224_224");
    EXPECT_EQ(firstInput.datatype(), "FP32");
    EXPECT_EQ(firstInput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(firstInput.shape(), {1, -1, 224, 224}));
    auto secondInput = response.inputs().at(1);
    EXPECT_EQ(secondInput.name(), "Input_U8_1_3_62:92_62:92");
    EXPECT_EQ(secondInput.datatype(), "U8");
    EXPECT_EQ(secondInput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(secondInput.shape(), {1, 3, -1, -1}));

    EXPECT_EQ(response.outputs_size(), 2);
    auto firstOutput = response.outputs().at(1);
    EXPECT_EQ(firstOutput.name(), "Output_I32_1_-1");
    EXPECT_EQ(firstOutput.datatype(), "I32");
    EXPECT_EQ(firstOutput.shape_size(), 2);
    EXPECT_TRUE(isShapeTheSame(firstOutput.shape(), {1, -1}));
    auto secondOutput = response.outputs().at(0);
    EXPECT_EQ(secondOutput.name(), "Output_FP32_1_224:294_224:294_3");
    EXPECT_EQ(secondOutput.datatype(), "FP32");
    EXPECT_EQ(secondOutput.shape_size(), 4);
    EXPECT_TRUE(isShapeTheSame(secondOutput.shape(), {1, -1, -1, 3}));
}
