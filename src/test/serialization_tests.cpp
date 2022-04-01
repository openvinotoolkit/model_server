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

#include "../serialization.hpp"
#include "test_utils.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using namespace ovms;

using testing::_;
using testing::NiceMock;
using testing::Throw;

const std::vector<ovms::Precision> SUPPORTED_OUTPUT_PRECISIONS{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    // ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

const std::vector<ovms::Precision> UNSUPPORTED_OUTPUT_PRECISIONS{
    // ovms::Precision::UNDEFINED, // Cannot create tensor with such precision
    // ovms::Precision::MIXED, // Cannot create tensor with such precision
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    // ovms::Precision::Q78, // Cannot create tensor with such precision
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::BIN, // Cannot create tensor with such precision
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM),

    // TODO: There are new API 2.0 precisions we do not support. Add tests for those.
};

class TensorflowGRPCPredict : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        ovms::Precision precision = ovms::Precision::FP32;

        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3, 1, 1},
            Layout{"NHWC"});
        SetUpTensorProto(TensorInfo::getPrecisionAsDataType(precision));
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

class SerializeTFTensorProto : public TensorflowGRPCPredict {
public:
    std::tuple<
        std::shared_ptr<ovms::TensorInfo>,
        ov::Tensor>
    getInputs(ovms::Precision precision) {
        std::shared_ptr<ovms::TensorInfo> servableOutput =
            std::make_shared<ovms::TensorInfo>(
                std::string("2_values_C_layout"),
                precision,
                ovms::Shape{2},
                Layout{"C"});
        ov::Tensor mockTensor = ov::Tensor(
            ovmsPrecisionToIE2Precision(precision), ov::Shape{2});
        return std::make_tuple(servableOutput, mockTensor);
    }
};

TEST(SerializeTFTensorProtoSingle, NegativeMismatchBetweenTensorInfoAndTensorPrecision) {
    ovms::Precision tensorInfoPrecision = ovms::Precision::FP32;
    shape_t tensorInfoShape{1, 3, 224, 224};
    auto layout = Layout{"NCHW"};
    const std::string name = "NOT_IMPORTANT";
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(name, tensorInfoPrecision, tensorInfoShape, layout);
    ov::Tensor tensor(ov::element::i32, tensorInfoShape);
    TensorProto responseOutput;
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorInfo,
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST(SerializeTFTensorProtoSingle, NegativeMismatchBetweenTensorInfoAndTensorShape) {
    ovms::Precision tensorInfoPrecision = ovms::Precision::FP32;
    shape_t tensorInfoShape{1, 3, 224, 224};
    shape_t tensorShape{1, 3, 225, 225};
    auto layout = Layout{"NCHW"};
    const std::string name = "NOT_IMPORTANT";
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(name, tensorInfoPrecision, tensorInfoShape, layout);
    ov::Tensor tensor(tensorInfo->getOvPrecision(), tensorShape);
    TensorProto responseOutput;
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorInfo,
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_P(SerializeTFTensorProto, SerializeTensorProtoShouldSucceedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TensorProto responseOutput;
    ov::Tensor mockTensor = std::get<1>(inputs);
    // EXPECT_CALL(*mockTensor, get_byte_size()); // TODO: Mock it properly with templates
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        mockTensor);
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << toString(testedPrecision)
        << "should succeed";
}

class SerializeTFTensorProtoNegative : public SerializeTFTensorProto {};

TEST_P(SerializeTFTensorProtoNegative, SerializeTensorProtoShouldSucceedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TensorProto responseOutput;
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision"
        << toString(testedPrecision)
        << "should fail";
}

TEST(SerializeTFGRPCPredictResponse, ShouldSuccessForSupportedPrecision) {
    PredictResponse response;
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_INPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(tensorInfo->getOvPrecision(), ov::Shape{1, 10});
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, tenMap, &response, getTensorInfoName);
    EXPECT_TRUE(status.ok());
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFTensorProto,
    ::testing::ValuesIn(SUPPORTED_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeTFTensorProto::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeTFTensorProtoNegative,
    ::testing::ValuesIn(UNSUPPORTED_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeTFTensorProtoNegative::ParamType>& info) {
        return toString(info.param);
    });
