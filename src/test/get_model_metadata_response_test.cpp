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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <rapidjson/document.h>
#include "../get_model_metadata_impl.hpp"
#include "../modelmanager.hpp"
#include "../status.hpp"

using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::NiceMock;
using namespace rapidjson;

class GetModelMetadataResponse : public ::testing::Test {
    struct Info {
        InferenceEngine::Precision precision;
        ovms::shape_t shape;
    };

    using tensor_desc_map_t = std::unordered_map<std::string, Info>;

    class MockModelInstance : public ovms::ModelInstance {
    public:
        MOCK_METHOD(const ovms::tensor_map_t&,      getInputsInfo,  (), (const, override));
        MOCK_METHOD(const ovms::tensor_map_t&,      getOutputsInfo, (), (const, override));
        MOCK_METHOD(const std::string&,             getName,        (), (const, override));
        MOCK_METHOD(ovms::model_version_t,          getVersion,     (), (const, override));
    };

    tensor_desc_map_t  inputTensors;
    tensor_desc_map_t  outputTensors;
    ovms::tensor_map_t networkInputs;
    ovms::tensor_map_t networkOutputs;

    std::string             modelName       = "resnet";
    ovms::model_version_t   modelVersion    = 23;

protected:
    std::shared_ptr<NiceMock<MockModelInstance>> instance;
    tensorflow::serving::GetModelMetadataResponse response;

    void SetUp() override {
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

        prepare(inputTensors,  networkInputs);
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
};

TEST_F(GetModelMetadataResponse, HasModelSpec) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    EXPECT_TRUE(response.has_model_spec());
}

TEST_F(GetModelMetadataResponse, HasVersion) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    EXPECT_TRUE(response.model_spec().has_version());
}

TEST_F(GetModelMetadataResponse, HasCorrectVersion) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    EXPECT_EQ(response.model_spec().version().value(), 23);
}

TEST_F(GetModelMetadataResponse, HasOneMetadataInfo) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    EXPECT_EQ(response.metadata_size(), 1);
}

TEST_F(GetModelMetadataResponse, HasCorrectMetadataSignatureName) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    EXPECT_NE(
        response.metadata().find("signature_def"),
        response.metadata().end());
}

TEST_F(GetModelMetadataResponse, HasOneSignatureDef) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_EQ(def.signature_def_size(), 1);
}

TEST_F(GetModelMetadataResponse, HasCorrectSignatureDefName) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    EXPECT_NE(
        def.signature_def().find("serving_default"),
        def.signature_def().end());
}

TEST_F(GetModelMetadataResponse, HasCorrectTensorNames) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs  = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_EQ(inputs .size(), 2);
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

TEST_F(GetModelMetadataResponse, HasCorrectPrecision) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs  = ((*def.mutable_signature_def())["serving_default"]).inputs();
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

TEST_F(GetModelMetadataResponse, HasCorrectShape) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs  = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    auto isShape = [](
        const tensorflow::TensorShapeProto& actual,
        const std::vector<size_t>&&         expected) -> bool {
        if (actual.dim_size() != expected.size()) {
            return false;
        }
        for (size_t i = 0; i < actual.dim_size(); i++) {
            if (actual.dim(i).size() != expected[i]) {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(isShape(
        inputs.at("Input_FP32_1_3_224_224").tensor_shape(),
        {1, 3, 224, 224}));
    EXPECT_TRUE(isShape(
        inputs.at("Input_U8_1_3_62_62").tensor_shape(),
        {1, 3, 62, 62}));
    EXPECT_TRUE(isShape(
        outputs.at("Output_I32_1_2000").tensor_shape(),
        {1, 2000}));
    EXPECT_TRUE(isShape(
        outputs.at("Output_FP32_2_20_3").tensor_shape(),
        {2, 20, 3}));
}

TEST_F(GetModelMetadataResponse, serialize2Json) {
    ovms::GetModelMetadataImpl::buildResponse(instance, &response);
    std::string json_output;
    const tensorflow::serving::GetModelMetadataResponse * response_p = &response;
    ovms::Status error_status = ovms::GetModelMetadataImpl::serializeResponse2Json(response_p, &json_output);
    const char * json_array = json_output.c_str();
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
    tensorflow::serving::GetModelMetadataRequest * request_p = &request_grpc;
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
    tensorflow::serving::GetModelMetadataRequest * request_p = &request_grpc;
    ovms::Status status = ovms::GetModelMetadataImpl::createGrpcRequest(model_name, model_version, request_p);
    bool has_requested_version = request_p->model_spec().has_version();
    std::string metadata_field = request_p->metadata_field(0);
    std::string requested_model_name = request_p->model_spec().name();
    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(has_requested_version, false);
    EXPECT_EQ(requested_model_name, "dummy");
    EXPECT_EQ(metadata_field, "signature_def");
}
