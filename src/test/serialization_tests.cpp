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

#include "../serialization.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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

const std::vector<Precision> SUPPORTED_OUTPUT_PRECISIONS{
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
    Precision::I64,
    // Precision::BIN,
    // Precision::BOOL
    // //Precision::CUSTOM)
};

const std::vector<Precision> UNSUPPORTED_OUTPUT_PRECISIONS{
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
    // Precision::I64,
    Precision::BIN,
    Precision::BOOL
    // Precision::CUSTOM),
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

class SerializeTFTensorProto : public TensorflowGRPCPredict {
public:
    std::tuple<
        std::shared_ptr<ovms::TensorInfo>,
        std::shared_ptr<MockBlob>>
    getInputs(Precision precision) {
        std::shared_ptr<ovms::TensorInfo> networkOutput =
            std::make_shared<ovms::TensorInfo>(
                std::string("2_values_C_layout"),
                precision,
                shape_t{2},
                InferenceEngine::Layout::C);
        std::shared_ptr<MockBlob> mockBlob = std::make_shared<MockBlob>(networkOutput->getTensorDesc());
        return std::make_tuple(networkOutput, mockBlob);
    }
};

TEST_P(SerializeTFTensorProto, SerializeTensorProtoShouldSucceedForPrecision) {
    Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TensorProto responseOutput;
    std::shared_ptr<MockBlob> mockBlob = std::get<1>(inputs);
    EXPECT_CALL(*mockBlob, element_size());
    auto status = serializeBlobToTensorProto(responseOutput,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << testedPrecision
        << "should succeed";
}

class SerializeTFTensorProtoNegative : public SerializeTFTensorProto {};

TEST_P(SerializeTFTensorProtoNegative, SerializeTensorProtoShouldSucceedForPrecision) {
    Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TensorProto responseOutput;
    auto status = serializeBlobToTensorProto(responseOutput,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision"
        << testedPrecision
        << "should fail";
}

class SerializeTFGRPCPredictResponse : public SerializeTFTensorProto {
public:
    std::tuple<InferenceEngine::InferRequest, ovms::tensor_map_t>
    getInputs(Precision testedPrecision) {
        InferenceEngine::TensorDesc tensorDesc(testedPrecision, shape_t{2}, InferenceEngine::Layout::C);
        std::shared_ptr<MockIInferRequestProperGetBlob> mInferRequestPtr =
            std::make_shared<MockIInferRequestProperGetBlob>(tensorDesc);
        InferenceEngine::InferRequest inferRequest(mInferRequestPtr);
        ovms::tensor_map_t tenMap;
        std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
            std::string("First"),
            tensorDesc.getPrecision(),
            tensorDesc.getDims(),
            tensorDesc.getLayout());
        tenMap["First"] = tensorInfo;
        return std::make_tuple(inferRequest, tenMap);
    }
};

TEST_P(SerializeTFGRPCPredictResponse, ShouldSuccessForSupportedPrecision) {
    Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    PredictResponse response;
    auto status = serializePredictResponse(std::get<0>(inputs), std::get<1>(inputs), &response);
    EXPECT_TRUE(status.ok());
}

class SerializeTFGRPCPredictResponseNegative : public SerializeTFGRPCPredictResponse {};

TEST_P(SerializeTFGRPCPredictResponseNegative, DISABLED_ShouldFailForUnsupportedPrecision) {
    Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    PredictResponse response;
    auto status = serializePredictResponse(std::get<0>(inputs), std::get<1>(inputs), &response);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION);
}

TEST_F(SerializeTFGRPCPredictResponseNegative, DISABLED_OutputNameNotCreatedInInferenceShouldFail) {
    auto inputs = getInputs(Precision::FP32);
    std::shared_ptr<MockIInferRequest> mInferRequestPtr =
        std::make_shared<MockIInferRequest>();
    InferenceEngine::InferRequest inferRequest(mInferRequestPtr);
    EXPECT_CALL(*mInferRequestPtr, GetBlob(_, _, _));
    PredictResponse response;
    auto status = serializePredictResponse(inferRequest, std::get<1>(inputs), &response);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_SERIALIZATION_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFTensorProto,
    ::testing::ValuesIn(SUPPORTED_OUTPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFTensorProtoNegative,
    ::testing::ValuesIn(UNSUPPORTED_OUTPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFGRPCPredictResponse,
    ::testing::ValuesIn(SUPPORTED_OUTPUT_PRECISIONS),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFGRPCPredictResponseNegative,
    ::testing::ValuesIn(UNSUPPORTED_OUTPUT_PRECISIONS),
    ::testing::PrintToStringParamName());
#pragma GCC diagnostic pop
