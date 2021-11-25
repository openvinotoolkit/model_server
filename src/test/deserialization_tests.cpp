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
#include "ovtestutils.hpp"
#include "test_utils.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using InferenceEngine::IInferRequest;
using InferenceEngine::PreProcessInfo;
using InferenceEngine::ResponseDesc;

using namespace ovms;
using namespace InferenceEngine;

using testing::_;
using testing::NiceMock;
using testing::Throw;

const std::vector<InferenceEngine::Precision> SUPPORTED_INPUT_PRECISIONS{
    // InferenceEngine::Precision::UNSPECIFIED,
    // InferenceEngine::Precision::MIXED,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::Q78,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U16,
    InferenceEngine::Precision::I32,
    // InferenceEngine::Precision::I64,
    // InferenceEngine::Precision::BIN,
    // InferenceEngine::Precision::BOOL
    // //InferenceEngine::Precision::CUSTOM)
};

const std::vector<InferenceEngine::Precision> UNSUPPORTED_INPUT_PRECISIONS{
    InferenceEngine::Precision::UNSPECIFIED,
    InferenceEngine::Precision::MIXED,
    // InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::Q78,
    // InferenceEngine::Precision::I16,
    // InferenceEngine::Precision::U8,
    // InferenceEngine::Precision::I8,
    // InferenceEngine::Precision::U16,
    // InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::BIN,
    InferenceEngine::Precision::BOOL
    // InferenceEngine::Precision::CUSTOM)
};

class TensorflowGRPCPredict : public ::testing::TestWithParam<InferenceEngine::Precision> {
protected:
    void SetUp() override {
        InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
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
    InferenceEngine::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    InferenceEngine::InferRequest inferRequest;
    InputSink<InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION)
        << "Unsupported OVMS precision:"
        << testedPrecision
        << " should return error";
}

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetBlobException) {
    InferenceEngine::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    InferenceEngine::InferRequest inferRequest;
    InputSink<InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION) << status.string();
}

class MockTensorProtoDeserializatorThrowingInferenceEngine {
public:
    MOCK_METHOD(InferenceEngine::Blob::Ptr,
        deserializeTensorProto,
        (const tensorflow::TensorProto&,
            const std::shared_ptr<ovms::TensorInfo>&, bool));
};

// Enables static method mock
class MockTensorProtoDeserializator {
public:
    static MockTensorProtoDeserializatorThrowingInferenceEngine* mock;
    static InferenceEngine::Blob::Ptr deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<ovms::TensorInfo>& tensorInfo, bool isPipeline) {
        return mock->deserializeTensorProto(requestInput, tensorInfo, isPipeline);
    }
};

MockTensorProtoDeserializatorThrowingInferenceEngine* MockTensorProtoDeserializator::mock = nullptr;

TEST_F(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetBlobException2) {
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32, shape_t{1, 10}, InferenceEngine::Layout::NC);
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_INPUT_NAME,
        tensorDesc.getPrecision(),
        tensorDesc.getDims(),
        tensorDesc.getLayout());
    tensorMap[tensorName] = tensorInfo;
    InferenceEngine::Core ieCore;
    InferenceEngine::CNNNetwork network = ieCore.ReadNetwork(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    InferenceEngine::ExecutableNetwork execNetwork = ieCore.LoadNetwork(network, "CPU");
    InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
    std::shared_ptr<NiceMock<MockBlob>> mockBlobPtr = std::make_shared<NiceMock<MockBlob>>(tensorDesc);
    inferRequest.SetBlob("b", mockBlobPtr);
    MockTensorProtoDeserializatorThrowingInferenceEngine mockTPobject;
    MockTensorProtoDeserializator::mock = &mockTPobject;
    EXPECT_CALL(mockTPobject, deserializeTensorProto(_, _, _))
        .Times(1)
        .WillRepeatedly(
            Throw(InferenceEngine::GeneralError("")));
    InputSink<InferRequest&> inputSink(inferRequest);
    Status status;
    status = deserializePredictRequest<MockTensorProtoDeserializator>(
        request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR) << status.string();
}

TEST_F(GRPCPredictRequest, ShouldSuccessForSupportedPrecision) {
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32, shape_t{1, 10}, InferenceEngine::Layout::NC);
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_INPUT_NAME,
        tensorDesc.getPrecision(),
        tensorDesc.getDims(),
        tensorDesc.getLayout());
    tensorMap[tensorName] = tensorInfo;
    InferenceEngine::Core ieCore;
    InferenceEngine::CNNNetwork network = ieCore.ReadNetwork(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    InferenceEngine::ExecutableNetwork execNetwork = ieCore.LoadNetwork(network, "CPU");
    InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
    std::shared_ptr<NiceMock<MockBlob>> mockBlobPtr = std::make_shared<NiceMock<MockBlob>>(tensorDesc);
    inferRequest.SetBlob("b", mockBlobPtr);
    InputSink<InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_TRUE(status.ok());
}

TEST_P(DeserializeTFTensorProtoNegative, ShouldReturnNullptrForPrecision) {
    InferenceEngine::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName], isPipeline);
    EXPECT_EQ(nullptr, blobPtr) << "Unsupported OVMS precision:"
                                << testedPrecision
                                << " should return nullptr";
}

TEST_P(DeserializeTFTensorProto, ShouldReturnValidBlob) {
    InferenceEngine::Precision testedPrecision = GetParam();
    SetUpTensorProto(fromInferenceEnginePrecision(testedPrecision));
    tensorMap[tensorName]->setPrecision(testedPrecision);
    InferenceEngine::Blob::Ptr blobPtr = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName], isPipeline);
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
