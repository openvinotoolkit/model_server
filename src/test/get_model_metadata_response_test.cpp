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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include "../get_model_metadata_impl.hpp"
#include "../modelmanager.hpp"
#include "../status.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using namespace rapidjson;

class GetModelMetadataResponse : public ::testing::Test {
    struct Info {
        InferenceEngine::Precision precision;
        ovms::shape_t shape;
    };

    using tensor_desc_map_t = std::unordered_map<std::string, Info>;

    class MockModelInstance : public MockModelInstanceChangingStates {
    public:
        MockModelInstance() :
            MockModelInstanceChangingStates("UNUSED_NAME", UNUSED_MODEL_VERSION) {
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
    ovms::tensor_map_t networkInputs;
    ovms::tensor_map_t networkOutputs;

protected:
    std::string modelName = "resnet";
    ovms::model_version_t modelVersion = 23;

    std::shared_ptr<NiceMock<MockModelInstance>> instance;
    tensorflow::serving::GetModelMetadataResponse response;

    virtual void prepare() {
        instance = std::make_shared<NiceMock<MockModelInstance>>();

        inputTensors = tensor_desc_map_t({
            {"Input_FP32_1_3_224_224", {
                                           InferenceEngine::Precision::FP32,
                                           {1, 3, 224, 224},
                                       }},
            {"Input_U8_1_3_62_62", {
                                       InferenceEngine::Precision::U8,
                                       {1, 3, 62, 62},
                                   }},
        });

        outputTensors = tensor_desc_map_t({
            {"Output_I32_1_2000", {
                                      InferenceEngine::Precision::I32,
                                      {1, 2000},
                                  }},
            {"Output_FP32_2_20_3", {
                                       InferenceEngine::Precision::FP32,
                                       {2, 20, 3},
                                   }},
        });

        auto prepare = [](const tensor_desc_map_t& desc,
                           ovms::tensor_map_t& tensors) {
            for (const auto& pair : desc) {
                tensors[pair.first] = std::make_shared<ovms::TensorInfo>(
                    pair.first,
                    pair.second.precision,
                    pair.second.shape);
            }
        };

        prepare(inputTensors, networkInputs);
        prepare(outputTensors, networkOutputs);

        ON_CALL(*instance, getInputsInfo())
            .WillByDefault(ReturnRef(networkInputs));
        ON_CALL(*instance, getOutputsInfo())
            .WillByDefault(ReturnRef(networkOutputs));
        ON_CALL(*instance, getName())
            .WillByDefault(ReturnRef(modelName));
        ON_CALL(*instance, getVersion())
            .WillByDefault(Return(modelVersion));
    }

    void SetUp() override {
        this->prepare();
    }
};

class GetModelMetadataResponseBuild : public GetModelMetadataResponse {
protected:
    void prepare() override {
        GetModelMetadataResponse::prepare();
        ASSERT_EQ(ovms::GetModelMetadataImpl::buildResponse(instance, &response), ovms::StatusCode::OK);
    }
};

TEST_F(GetModelMetadataResponseBuild, HasModelSpec) {
    EXPECT_TRUE(response.has_model_spec());
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectName) {
    EXPECT_EQ(response.model_spec().name(), "resnet");
}

TEST_F(GetModelMetadataResponseBuild, HasVersion) {
    EXPECT_TRUE(response.model_spec().has_version());
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectVersion) {
    EXPECT_EQ(response.model_spec().version().value(), 23);
}

TEST_F(GetModelMetadataResponseBuild, HasOneMetadataInfo) {
    EXPECT_EQ(response.metadata_size(), 1);
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectMetadataSignatureName) {
    EXPECT_NE(
        response.metadata().find("signature_def"),
        response.metadata().end());
}

TEST_F(GetModelMetadataResponseBuild, HasOneSignatureDef) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_EQ(def.signature_def_size(), 1);
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectSignatureDefName) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_NE(
        def.signature_def().find("serving_default"),
        def.signature_def().end());
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectTensorNames) {
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_EQ(inputs.size(), 2);
    EXPECT_EQ(outputs.size(), 2);

    EXPECT_EQ(
        inputs.at("Input_FP32_1_3_224_224").name(),
        "Input_FP32_1_3_224_224");
    EXPECT_EQ(
        inputs.at("Input_U8_1_3_62_62").name(),
        "Input_U8_1_3_62_62");
    EXPECT_EQ(
        outputs.at("Output_I32_1_2000").name(),
        "Output_I32_1_2000");
    EXPECT_EQ(
        outputs.at("Output_FP32_2_20_3").name(),
        "Output_FP32_2_20_3");
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectPrecision) {
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
        outputs.at("Output_I32_1_2000").dtype(),
        tensorflow::DT_INT32);
    EXPECT_EQ(
        outputs.at("Output_FP32_2_20_3").dtype(),
        tensorflow::DT_FLOAT);
}

TEST_F(GetModelMetadataResponseBuild, HasCorrectShape) {
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
        outputs.at("Output_I32_1_2000").tensor_shape(),
        {1, 2000}));
    EXPECT_TRUE(isShapeTheSame(
        outputs.at("Output_FP32_2_20_3").tensor_shape(),
        {2, 20, 3}));
}

TEST_F(GetModelMetadataResponse, ModelVersionNotLoadedAnymore) {
    instance->unloadModel();
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(instance, &response), ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

TEST_F(GetModelMetadataResponse, ModelVersionNotLoadedYet) {
    instance->loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(instance, &response), ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);
}

TEST_F(GetModelMetadataResponseBuild, serialize2Json) {
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

TEST(RESTGetModelMetadataResponse, createGrpcRequestVersionSet) {
    std::string model_name = "dummy";
    std::optional<int64_t> model_version = 1;
    tensorflow::serving::GetModelMetadataRequest request_grpc;
    tensorflow::serving::GetModelMetadataRequest* request_p = &request_grpc;
    ovms::Status status = ovms::GetModelMetadataImpl::createGrpcRequest(model_name, model_version, request_p);
    bool has_requested_version = request_p->model_spec().has_version();
    auto requested_version = request_p->model_spec().version().value();
    std::string metadata_field = request_p->metadata_field(0);
    std::string requested_model_name = request_p->model_spec().name();
    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(has_requested_version, true);
    EXPECT_EQ(requested_version, 1);
    EXPECT_EQ(requested_model_name, "dummy");
    EXPECT_EQ(metadata_field, "signature_def");
}

TEST(RESTGetModelMetadataResponse, createGrpcRequestNoVersion) {
    std::string model_name = "dummy";
    std::optional<int64_t> model_version;
    tensorflow::serving::GetModelMetadataRequest request_grpc;
    tensorflow::serving::GetModelMetadataRequest* request_p = &request_grpc;
    ovms::Status status = ovms::GetModelMetadataImpl::createGrpcRequest(model_name, model_version, request_p);
    bool has_requested_version = request_p->model_spec().has_version();
    std::string metadata_field = request_p->metadata_field(0);
    std::string requested_model_name = request_p->model_spec().name();
    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(has_requested_version, false);
    EXPECT_EQ(requested_model_name, "dummy");
    EXPECT_EQ(metadata_field, "signature_def");
}
