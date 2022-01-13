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

#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "../deserialization.hpp"
#include "test_utils.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using namespace ovms;

using testing::_;
using testing::NiceMock;
using testing::Throw;

const std::vector<ovms::Precision> SUPPORTED_INPUT_PRECISIONS{
    // ovms::Precision::UNSPECIFIED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // InferenceEngine::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

const std::vector<ovms::Precision> UNSUPPORTED_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

class TensorflowGRPCPredict : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        auto precision = ovms::Precision::FP32;
        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3},
            layout_t{"NC"});
        SetUpTensorProto(TensorInfo::getPrecisionAsDataType(precision));
    }
    void SetUpTensorProto(tensorflow::DataType dataType) {
        tensorProto.set_dtype(dataType);
        auto tensorShape = tensorProto.mutable_tensor_shape();
        tensorShape->Clear();
        tensorShape->add_dim()->set_size(1);
        tensorShape->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);
        *(tensorProto.mutable_tensor_content()) = std::string(1 * DUMMY_MODEL_INPUT_SIZE, '1');
    }
    TensorProto tensorProto;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t tensorMap;
    bool isPipeline = false;
};

class DeserializeTFTensorProto : public TensorflowGRPCPredict {};
class DeserializeTFTensorProtoNegative : public TensorflowGRPCPredict {};

class GRPCPredictRequest : public TensorflowGRPCPredict {
public:
    void SetUp() {
        TensorflowGRPCPredict::SetUp();
        (*request.mutable_inputs())[tensorName] = tensorProto;
    }
    void TearDown() {
        request.mutable_inputs()->clear();
    }

public:
    PredictRequest request;
};

class GRPCPredictRequestNegative : public GRPCPredictRequest {};

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::runtime::InferRequest inferRequest;
    InputSink<ov::runtime::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION)
        << "Unsupported OVMS precision:"
        << toString(testedPrecision)
        << " should return error";
}

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::runtime::InferRequest inferRequest;
    InputSink<ov::runtime::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION) << status.string();
}

class MockTensorProtoDeserializatorThrowingInferenceEngine {
public:
    MOCK_METHOD(ov::runtime::Tensor,
        deserializeTensorProto,
        (const tensorflow::TensorProto&,
            const std::shared_ptr<ovms::TensorInfo>&));
};

// Enables static method mock
class MockTensorProtoDeserializator {
public:
    static MockTensorProtoDeserializatorThrowingInferenceEngine* mock;
    static ov::runtime::Tensor deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<ovms::TensorInfo>& tensorInfo) {
        return mock->deserializeTensorProto(requestInput, tensorInfo);
    }
};

MockTensorProtoDeserializatorThrowingInferenceEngine* MockTensorProtoDeserializator::mock = nullptr;

TEST_F(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException2) {
    ov::runtime::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::runtime::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::runtime::InferRequest inferRequest = compiledModel.create_infer_request();
    MockTensorProtoDeserializatorThrowingInferenceEngine mockTPobject;
    MockTensorProtoDeserializator::mock = &mockTPobject;
    EXPECT_CALL(mockTPobject, deserializeTensorProto(_, _))
        .Times(1)
        .WillRepeatedly(
            Throw(ov::Exception("")));
    InputSink<ov::runtime::InferRequest&> inputSink(inferRequest);
    Status status;
    status = deserializePredictRequest<MockTensorProtoDeserializator>(
        request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR) << status.string();
}

TEST_F(GRPCPredictRequest, ShouldSuccessForSupportedPrecision) {
    ov::runtime::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::runtime::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::runtime::InferRequest inferRequest = compiledModel.create_infer_request();
    InputSink<ov::runtime::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_TRUE(status.ok());
}

TEST_P(DeserializeTFTensorProtoNegative, ShouldReturnNullptrForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::runtime::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_FALSE((bool)tensor) << "Unsupported OVMS precision:"
                               << toString(testedPrecision)
                               << " should return nullptr";
}

TEST_P(DeserializeTFTensorProto, ShouldReturnValidTensor) {
    ovms::Precision testedPrecision = GetParam();
    SetUpTensorProto(TensorInfo::getPrecisionAsDataType(testedPrecision));
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::runtime::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_TRUE((bool)tensor) << "Supported OVMS precision:"
                              << toString(testedPrecision)
                              << " should return valid tensor ptr";
}

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequestNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<GRPCPredictRequestNegative::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequest,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<GRPCPredictRequest::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProtoNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<DeserializeTFTensorProtoNegative::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProto,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<DeserializeTFTensorProto::ParamType>& info) {
        return toString(info.param);
    });
