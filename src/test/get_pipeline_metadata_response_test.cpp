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

#include <chrono>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include "../dags/pipelinedefinition.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../model_metric_reporter.hpp"

#include "constructor_enabled_model_manager.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace rapidjson;

class GetPipelineMetadataResponse : public ::testing::Test {
protected:
    class MockPipelineDefinitionGetInputsOutputsInfo : public PipelineDefinition {
        Status status = StatusCode::OK;

    public:
        MockPipelineDefinitionGetInputsOutputsInfo() :
            PipelineDefinition("pipeline_name", {}, {}) {
            PipelineDefinition::status.handle(ValidationPassedEvent());
        }

        void mockMetadata(const tensor_map_t& inputsInfo, const tensor_map_t& outputsInfo) {
            this->inputsInfo = inputsInfo;
            this->outputsInfo = outputsInfo;
        }

        void mockStatus(Status status) {
            this->status = status;
        }
        PipelineDefinitionStatus& getPipelineDefinitionStatus() {
            return PipelineDefinition::status;
        }
    };

    MockPipelineDefinitionGetInputsOutputsInfo pipelineDefinition;
    tensorflow::serving::GetModelMetadataResponse response;
    ConstructorEnabledModelManager manager;

    virtual void prepare() {
        pipelineDefinition.mockMetadata({{"Input_FP32_1_3_224_224",
                                             std::make_shared<TensorInfo>("Input_FP32_1_3_224_224", ovms::Precision::FP32, ovms::Shape{1, 3, 224, 224})},
                                            {"Input_U8_1_3_62_62",
                                                std::make_shared<TensorInfo>("Input_U8_1_3_62_62", ovms::Precision::U8, ovms::Shape{1, 3, 62, 62})},
                                            {"Input_Unspecified",
                                                TensorInfo::getUnspecifiedTensorInfo()}},
            {{"Output_I32_1_2000",
                 std::make_shared<TensorInfo>("Output_I32_1_2000", ovms::Precision::I32, ovms::Shape{1, 2000})},
                {"Output_FP32_2_20_3",
                    std::make_shared<TensorInfo>("Output_FP32_2_20_3", ovms::Precision::FP32, ovms::Shape{2, 20, 3})},
                {"Output_Unspecified",
                    TensorInfo::getUnspecifiedTensorInfo()}});
    }

    void SetUp() override {
        this->prepare();
    }
};

class GetPipelineMetadataResponseBuild : public GetPipelineMetadataResponse {
protected:
    void prepare() override {
        GetPipelineMetadataResponse::prepare();
        ASSERT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::OK);
    }
};

TEST_F(GetPipelineMetadataResponseBuild, HasModelSpec) {
    EXPECT_TRUE(response.has_model_spec());
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectName) {
    EXPECT_EQ(response.model_spec().name(), "pipeline_name");
}

TEST_F(GetPipelineMetadataResponseBuild, HasVersion) {
    EXPECT_TRUE(response.model_spec().has_version());
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectVersion) {
    EXPECT_EQ(response.model_spec().version().value(), 1);
}

TEST_F(GetPipelineMetadataResponseBuild, HasOneMetadataInfo) {
    EXPECT_EQ(response.metadata_size(), 1);
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectMetadataSignatureName) {
    EXPECT_NE(
        response.metadata().find("signature_def"),
        response.metadata().end());
}

TEST_F(GetPipelineMetadataResponseBuild, HasOneSignatureDef) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_EQ(def.signature_def_size(), 1);
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectSignatureDefName) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_NE(
        def.signature_def().find("serving_default"),
        def.signature_def().end());
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectTensorNames) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_EQ(inputs.size(), 3);
    EXPECT_EQ(outputs.size(), 3);

    EXPECT_EQ(
        inputs.at("Input_FP32_1_3_224_224").name(),
        "Input_FP32_1_3_224_224");
    EXPECT_EQ(
        inputs.at("Input_U8_1_3_62_62").name(),
        "Input_U8_1_3_62_62");
    EXPECT_EQ(
        inputs.at("Input_Unspecified").name(),
        "Input_Unspecified");
    EXPECT_EQ(
        outputs.at("Output_I32_1_2000").name(),
        "Output_I32_1_2000");
    EXPECT_EQ(
        outputs.at("Output_FP32_2_20_3").name(),
        "Output_FP32_2_20_3");
    EXPECT_EQ(
        outputs.at("Output_Unspecified").name(),
        "Output_Unspecified");
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectPrecision) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_EQ(
        inputs.at("Input_FP32_1_3_224_224").dtype(),
        tensorflow::DT_FLOAT);
    EXPECT_EQ(
        inputs.at("Input_U8_1_3_62_62").dtype(),
        tensorflow::DT_UINT8);
    EXPECT_EQ(
        inputs.at("Input_Unspecified").dtype(),
        tensorflow::DT_INVALID);
    EXPECT_EQ(
        outputs.at("Output_I32_1_2000").dtype(),
        tensorflow::DT_INT32);
    EXPECT_EQ(
        outputs.at("Output_FP32_2_20_3").dtype(),
        tensorflow::DT_FLOAT);
    EXPECT_EQ(
        outputs.at("Output_Unspecified").dtype(),
        tensorflow::DT_INVALID);
}

TEST_F(GetPipelineMetadataResponseBuild, HasCorrectShape) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_TRUE(isShapeTheSame(
        inputs.at("Input_FP32_1_3_224_224").tensor_shape(),
        {1, 3, 224, 224}));
    EXPECT_TRUE(isShapeTheSame(
        inputs.at("Input_U8_1_3_62_62").tensor_shape(),
        {1, 3, 62, 62}));
    EXPECT_TRUE(isShapeTheSame(
        inputs.at("Input_Unspecified").tensor_shape(),
        {}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_I32_1_2000").tensor_shape(),
        {1, 2000}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_FP32_2_20_3").tensor_shape(),
        {2, 20, 3}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_Unspecified").tensor_shape(),
        {}));
}

TEST_F(GetPipelineMetadataResponse, ModelVersionNotLoadedAnymoreButPipelineNotReloadedYet) {
    pipelineDefinition.mockStatus(StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::OK);
}

TEST_F(GetPipelineMetadataResponse, ModelVersionNotLoadedYet) {
    pipelineDefinition.mockStatus(StatusCode::MODEL_VERSION_NOT_LOADED_YET);
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::OK);
}

TEST_F(GetPipelineMetadataResponse, PipelineNotLoadedAnymore) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(RetireEvent());
    auto status = ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE) << status.string();
}

TEST_F(GetPipelineMetadataResponse, PipelineNotLoadedYet) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(UsedModelChangedEvent());
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(ValidationFailedEvent());
    auto status = ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET) << status.string();
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(UsedModelChangedEvent());
    status = ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager);
    ASSERT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(GetPipelineMetadataResponse, PipelineAvailableOrAvailableRequiringRevalidation) {
    this->pipelineDefinition.getPipelineDefinitionStatus().handle(UsedModelChangedEvent());
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::OK);
}

TEST_F(GetPipelineMetadataResponseBuild, serialize2Json) {
    std::string json_output;
    const tensorflow::serving::GetModelMetadataResponse* response_p = &response;
    ovms::Status error_status = ovms::GetModelMetadataImpl::serializeResponse2Json(response_p, &json_output);
    const char* json_array = json_output.c_str();
    Document received_doc;
    received_doc.Parse(json_array);
    EXPECT_TRUE(received_doc.IsObject());
    EXPECT_TRUE(received_doc.HasMember("modelSpec"));
    EXPECT_TRUE(received_doc.HasMember("metadata"));
}

class GetPipelineMetadataResponseBuildWithDynamicAndRangeShapes : public GetPipelineMetadataResponseBuild {
protected:
    void prepare() override {
        GetPipelineMetadataResponse::prepare();
        pipelineDefinition.mockMetadata(
            {{"Input_FP32_1_-1_224_224",
                 std::make_shared<TensorInfo>("Input_FP32_1_-1_224_224", ovms::Precision::FP32, ovms::Shape{1, ovms::Dimension::any(), 224, 224})},
                {"Input_U8_1_3_62:92_62:92",
                    std::make_shared<TensorInfo>("Input_U8_1_3_62:92_62:92", ovms::Precision::U8, ovms::Shape{1, 3, {62, 92}, ovms::Dimension::any()})}},
            {{"Output_I32_1_-1",
                 std::make_shared<TensorInfo>("Output_I32_1_-1", ovms::Precision::I32, ovms::Shape{1, ovms::Dimension::any()})},
                {"Output_FP32_1_224:294_224:294_3",
                    std::make_shared<TensorInfo>("Output_FP32_1_224:294_224:294_3", ovms::Precision::FP32, ovms::Shape{1, {224, 294}, {224, 294}, 3})}});
        ASSERT_EQ(ovms::GetModelMetadataImpl::buildResponse(pipelineDefinition, &response, manager), ovms::StatusCode::OK);
    }
};

TEST_F(GetPipelineMetadataResponseBuildWithDynamicAndRangeShapes, HandleDynamicAndRangeShapes) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_TRUE(isShapeTheSame(
        inputs.at("Input_FP32_1_-1_224_224").tensor_shape(),
        {1, -1, 224, 224}));
    EXPECT_TRUE(isShapeTheSame(
        inputs.at("Input_U8_1_3_62:92_62:92").tensor_shape(),
        {1, 3, -1, -1}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_I32_1_-1").tensor_shape(),
        {1, -1}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_FP32_1_224:294_224:294_3").tensor_shape(),
        {1, -1, -1, 3}));
}
