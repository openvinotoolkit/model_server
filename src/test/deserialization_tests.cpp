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

#
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../deserialization.hpp"
#include "ovtestutils.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using InferenceEngine::IInferRequest;
using InferenceEngine::Precision;
using InferenceEngine::PreProcessInfo;
using InferenceEngine::ResponseDesc;

using namespace ovms;
using namespace InferenceEngine;

using testing::_;
using testing::NiceMock;
using testing::Throw;

const std::vector<Precision> SUPPORTED_INPUT_PRECISIONS{
    // Precision::UNSPECIFIED,
    // Precision::MIXED,
    Precision::FP32,
    Precision::FP16,
    // Precision::Q78,
    Precision::I16,
    Precision::U8,
    Precision::I8,
    Precision::U16,
    Precision::I32,
    // Precision::I64,
    // Precision::BIN,
    // Precision::BOOL
    // //Precision::CUSTOM)
};

const std::vector<Precision> UNSUPPORTED_INPUT_PRECISIONS{
    Precision::UNSPECIFIED,
    Precision::MIXED,
    // Precision::FP32,
    // Precision::FP16,
    Precision::Q78,
    // Precision::I16,
    // Precision::U8,
    // Precision::I8,
    // Precision::U16,
    // Precision::I32,
    Precision::I64,
    Precision::BIN,
    Precision::BOOL
    // Precision::CUSTOM)
};

class TensorflowGRPCPredict : public ::testing::TestWithParam<Precision> {
protected:
    void SetUp() override {
        Precision precision = Precision::FP32;
        InferenceEngine::TensorDesc tensorDesc_prec_1_3_1_1_NHWC = {
            precision,
            {1, 3, 1, 1},
            InferenceEngine::Layout::NHWC};

        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            tensorDesc_prec_1_3_1_1_NHWC.getPrecision(),
            tensorDesc_prec_1_3_1_1_NHWC.getDims(),
            tensorDesc_prec_1_3_1_1_NHWC.getLayout());
        SetUpTensorProto(fromInferenceEnginePrecision(precision));
    }
    void SetUpTensorProto(tensorflow::DataType dataType) {
        tensorProto.set_dtype(dataType);
        auto tensorShape = tensorProto.mutable_tensor_shape();
        tensorShape->Clear();
        tensorShape->add_dim()->set_size(1);
        tensorShape->add_dim()->set_size(3);
        tensorShape->add_dim()->set_size(1);
        tensorShape->add_dim()->set_size(1);
        *(tensorProto.mutable_tensor_content()) = std::string(1 * 3 * 1 * 1, '1');
    }
    TensorProto tensorProto;
    const char* tensorName = "Input_PRECISION_1_3_1_1_NHWC";
    ovms::tensor_map_t tensorMap;
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
    Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    InferenceEngine::InferRequest inferRequest;
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inferRequest);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION)
        << "Unsupported OVMS precision:"
        << testedPrecision
        << " should return error";
}

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetBlobException) {
    Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    std::shared_ptr<MockIInferRequestFailingInSetBlob> mInferRequestPtr =
        std::make_shared<MockIInferRequestFailingInSetBlob>();
    InferenceEngine::InferRequest inferRequest(mInferRequestPtr);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inferRequest);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION);
}

class MockTensorProtoDeserializatorThrowingInferenceEngine {
public:
    MOCK_METHOD(InferenceEngine::Blob::Ptr,
        deserializeTensorProto,
        (const tensorflow::TensorProto&,
            const std::shared_ptr<ovms::TensorInfo>&));
};

// Enables static method mock
class MockTensorProtoDeserializator {
public:
    static MockTensorProtoDeserializatorThrowingInferenceEngine* mock;
    static InferenceEngine::Blob::Ptr deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<ovms::TensorInfo>& tensorInfo) {
        return mock->deserializeTensorProto(requestInput, tensorInfo);
    }
};

MockTensorProtoDeserializatorThrowingInferenceEngine* MockTensorProtoDeserializator::mock = nullptr;

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetBlobException2) {
    Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    std::shared_ptr<MockIInferRequestFailingInSetBlob> mInferRequestPtr =
        std::make_shared<MockIInferRequestFailingInSetBlob>();
    InferenceEngine::InferRequest inferRequest(mInferRequestPtr);
    MockTensorProtoDeserializatorThrowingInferenceEngine mockTPobject;
    MockTensorProtoDeserializator::mock = &mockTPobject;
    EXPECT_CALL(mockTPobject, deserializeTensorProto(_, _))
        .Times(1)
        .WillRepeatedly(
            Throw(InferenceEngine::GeneralError("")));
    auto status =
        deserializePredictRequest<MockTensorProtoDeserializator>(
            request, tensorMap, inferRequest);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR);
}

TEST_P(GRPCPredictRequest, ShouldSuccessForSupportedPrecision) {
    Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    std::shared_ptr<MockIInferRequest> mInferRequestPtr = std::make_shared<MockIInferRequest>();
    InferenceEngine::InferRequest inferRequest(mInferRequestPtr);
    EXPECT_CALL(*mInferRequestPtr, SetBlob(_, _, _));
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inferRequest);
    EXPECT_TRUE(status.ok());
}

TEST_P(DeserializeTFTensorProtoNegative, ShouldReturnNullptrForPrecision) {
    Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    // InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteBlobGenerator>(tensorProto, tensorMap[tensorName]);
    InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_EQ(nullptr, blobPtr) << "Unsupported OVMS precision:"
                                << testedPrecision
                                << " should return nullptr";
}

TEST_P(DeserializeTFTensorProto, ShouldReturnValidBlob) {
    Precision testedPrecision = GetParam();
    SetUpTensorProto(fromInferenceEnginePrecision(testedPrecision));
    tensorMap[tensorName]->setPrecision(testedPrecision);
    // InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteBlobGenerator>(tensorProto, tensorMap[tensorName]);
    InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_NE(nullptr, blobPtr) << "Supported OVMS precision:"
                                << testedPrecision
                                << " should return valid blob ptr";
}

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequestNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequest,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProtoNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProto,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    ::testing::PrintToStringParamName());
#pragma GCC diagnostic pop
