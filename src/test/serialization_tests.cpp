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

#include <array>
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

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../capi_frontend/inferencetensor.hpp"
#include "../serialization.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "test_utils.hpp"

using TFTensorProto = tensorflow::TensorProto;

using TFPredictRequest = tensorflow::serving::PredictRequest;
using TFPredictResponse = tensorflow::serving::PredictResponse;

using namespace ovms;

using testing::_;
using testing::ElementsAre;
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
};

const std::vector<ovms::Precision> SUPPORTED_KFS_OUTPUT_PRECISIONS{
    // ovms::Precision::UNDECIFIED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // InferenceEngine::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    // ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

const std::vector<ovms::Precision> UNSUPPORTED_KFS_OUTPUT_PRECISIONS{
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
    // ovms::Precision::U32,
    // ovms::Precision::U64,
    // ovms::Precision::BIN, // Cannot create tensor with such precision
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

const std::vector<ovms::Precision> SUPPORTED_CAPI_OUTPUT_PRECISIONS{
    // ovms::Precision::BF16,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    ovms::Precision::I64,
    ovms::Precision::I32,
    ovms::Precision::I16,
    ovms::Precision::I8,
    // ovms::Precision::I4,
    ovms::Precision::U64,
    ovms::Precision::U32,
    ovms::Precision::U16,
    ovms::Precision::U8,
    // ovms::Precision::U4,
    // ovms::Precision::U1,
    // ovms::Precision::BOOL,
    // ovms::Precision::UNDEFINED,
};

const std::vector<ovms::Precision> UNSUPPORTED_CAPI_OUTPUT_PRECISIONS{
    ovms::Precision::BF16,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    // ovms::Precision::I64,
    // ovms::Precision::I32,
    // ovms::Precision::I16,
    // ovms::Precision::I8,
    ovms::Precision::I4,
    // ovms::Precision::U64,
    // ovms::Precision::U32,
    // ovms::Precision::U16,
    // ovms::Precision::U8,
    ovms::Precision::U4,
    ovms::Precision::U1,
    ovms::Precision::BOOL,
    // ovms::Precision::UNDEFINED,  // Cannot create ov tensor with such precision
};

namespace {
const std::string UNUSED_NAME{"UNUSED_NAME"};
const model_version_t UNUSED_VERSION{0};
}  // namespace

class TensorflowGRPCPredict : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        ovms::Precision precision = ovms::Precision::FP32;

        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3, 1, 1},
            Layout{"NHWC"});
        SetUpTensorProto(getPrecisionAsDataType(precision));
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
    TFTensorProto tensorProto;
    const char* tensorName = "Input_PRECISION_1_3_1_1_NHWC";
    ovms::tensor_map_t tensorMap;
};

class SerializeTFTensorProto : public TensorflowGRPCPredict {
public:
    std::tuple<
        std::shared_ptr<const ovms::TensorInfo>,
        ov::Tensor>
    getInputs(ovms::Precision precision) {
        std::shared_ptr<const ovms::TensorInfo> servableOutput =
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
    TFTensorProto responseOutput;
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
    TFTensorProto responseOutput;
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorInfo,
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_P(SerializeTFTensorProto, SerializeTensorProtoShouldSucceedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TFTensorProto responseOutput;
    ov::Tensor mockTensor = std::get<1>(inputs);
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        mockTensor);
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << toString(testedPrecision)
        << "should succeed";
}

class SerializeTFTensorProtoNegative : public SerializeTFTensorProto {};

TEST_P(SerializeTFTensorProtoNegative, SerializeTensorProtoShouldFailForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    TFTensorProto responseOutput;
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision"
        << toString(testedPrecision)
        << "should fail";
}

TEST(SerializeTFGRPCPredictResponse, ShouldSuccessForSupportedPrecision) {
    TFPredictResponse response;
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_OUTPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(tensorInfo->getOvPrecision(), ov::Shape{1, 10});
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName);
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

class KFServingGRPCPredict : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        ovms::Precision precision = ovms::Precision::FP32;

        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3, 1, 1},
            Layout{"NHWC"});
    }
    const char* tensorName = "Input_PRECISION_1_3_1_1_NHWC";
    ovms::tensor_map_t tensorMap;
    ::KFSResponse response;
};

TEST_F(KFServingGRPCPredict, ValidSerializationRaw) {
    ov::Tensor tensor(ov::element::f32, shape_t{1, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto* content = protoGetter.createContent(tensorName);
    auto status = serializeTensorToTensorProtoRaw(responseOutput,
        content,
        tensorMap[tensorName],
        tensor);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    EXPECT_EQ(responseOutput.name(), tensorName);
    EXPECT_EQ(responseOutput.datatype(), "FP32");
    EXPECT_EQ(responseOutput.shape(0), 1);
    EXPECT_EQ(responseOutput.shape(1), 3);
    EXPECT_EQ(responseOutput.shape(2), 1);
    EXPECT_EQ(responseOutput.shape(3), 1);
    EXPECT_EQ(response.raw_output_contents()[0].size(), 12);
}

TEST_F(KFServingGRPCPredict, ValidSerialization) {
    ov::Tensor tensor(ov::element::f32, shape_t{1, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorMap[tensorName],
        tensor);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    EXPECT_EQ(responseOutput.name(), tensorName);
    EXPECT_EQ(responseOutput.datatype(), "FP32");
    EXPECT_EQ(responseOutput.shape(0), 1);
    EXPECT_EQ(responseOutput.shape(1), 3);
    EXPECT_EQ(responseOutput.shape(2), 1);
    EXPECT_EQ(responseOutput.shape(3), 1);
    EXPECT_EQ(responseOutput.contents().fp32_contents_size(), 3);
}

TEST_F(KFServingGRPCPredict, NegativeMismatchBetweenTensorInfoAndTensorPrecisionRaw) {
    ov::Tensor tensor(ov::element::i32, shape_t{1, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto* content = protoGetter.createContent(tensorName);
    auto status = serializeTensorToTensorProtoRaw(responseOutput,
        content,
        tensorMap[tensorName],
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(KFServingGRPCPredict, NegativeMismatchBetweenTensorInfoAndTensorPrecision) {
    ov::Tensor tensor(ov::element::i32, shape_t{1, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorMap[tensorName],
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(KFServingGRPCPredict, NegativeMismatchBetweenTensorInfoAndTensorShapeRaw) {
    ov::Tensor tensor(ov::element::i32, shape_t{2, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto* content = protoGetter.createContent(tensorName);
    auto status = serializeTensorToTensorProtoRaw(responseOutput,
        content,
        tensorMap[tensorName],
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(KFServingGRPCPredict, NegativeMismatchBetweenTensorInfoAndTensorShape) {
    ov::Tensor tensor(ov::element::i32, shape_t{2, 3, 1, 1});
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto status = serializeTensorToTensorProto(responseOutput,
        tensorMap[tensorName],
        tensor);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

class SerializeKFSInferOutputTensor : public KFServingGRPCPredict {
public:
    std::tuple<
        std::shared_ptr<const ovms::TensorInfo>,
        ov::Tensor>
    getInputs(ovms::Precision precision) {
        std::shared_ptr<const ovms::TensorInfo> servableOutput =
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

TEST_P(SerializeKFSInferOutputTensor, SerializeTensorProtoShouldSucceedForPrecisionRaw) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto* content = protoGetter.createContent(tensorName);
    ov::Tensor mockTensor = std::get<1>(inputs);
    auto status = serializeTensorToTensorProtoRaw(responseOutput,
        content,
        std::get<0>(inputs),
        mockTensor);
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << toString(testedPrecision)
        << "should succeed";
}

TEST_P(SerializeKFSInferOutputTensor, SerializeTensorProtoShouldSucceedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    ov::Tensor mockTensor = std::get<1>(inputs);
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        mockTensor);
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << toString(testedPrecision)
        << "should succeed";
}

class SerializeKFSInferOutputTensorNegative : public SerializeKFSInferOutputTensor {};

TEST_P(SerializeKFSInferOutputTensorNegative, SerializeTensorProtoShouldFailForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto* content = protoGetter.createContent(tensorName);
    auto status = serializeTensorToTensorProtoRaw(responseOutput,
        content,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision"
        << toString(testedPrecision)
        << "should fail";
}

TEST_P(SerializeKFSInferOutputTensorNegative, SerializeTensorProtoShouldFailedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    auto inputs = getInputs(testedPrecision);
    KFSResponse response;
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(&response);
    auto& responseOutput = protoGetter.createOutput(tensorName);
    auto status = serializeTensorToTensorProto(responseOutput,
        std::get<0>(inputs),
        std::get<1>(inputs));
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision"
        << toString(testedPrecision)
        << "should fail";
}

TEST(SerializeKFSGRPCPredictResponse, ShouldSuccessForSupportedPrecision) {
    KFSResponse response;
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_OUTPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(tensorInfo->getOvPrecision(), ov::Shape{1, 10});
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(DUMMY_MODEL_OUTPUT_NAME, response.outputs(0).name());
    EXPECT_EQ("FP32", response.outputs(0).datatype());
    EXPECT_EQ(1, response.outputs(0).shape(0));
    EXPECT_EQ(10, response.outputs(0).shape(1));
    EXPECT_EQ(40, response.raw_output_contents(0).size());
}

TEST(SerializeKFSGRPCPredictResponse, ShouldSuccessForSupportedPrecisionWithuseSharedOutputContent) {
    KFSResponse response;
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_INPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(tensorInfo->getOvPrecision(), ov::Shape{1, 10});
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName, true);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(DUMMY_MODEL_INPUT_NAME, response.outputs(0).name());
    EXPECT_EQ("FP32", response.outputs(0).datatype());
    EXPECT_EQ(1, response.outputs(0).shape(0));
    EXPECT_EQ(10, response.outputs(0).shape(1));
    EXPECT_EQ(0, response.outputs(0).contents().fp32_contents_size());
    EXPECT_EQ(40, response.raw_output_contents(0).size());
}

TEST(SerializeKFSGRPCPredictResponse, ShouldSuccessForSupportedPrecisionWithsharedInputContentsNotUsed) {
    KFSResponse response;
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_INPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(tensorInfo->getOvPrecision(), ov::Shape{1, 10});
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName, false);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(DUMMY_MODEL_INPUT_NAME, response.outputs(0).name());
    EXPECT_EQ("FP32", response.outputs(0).datatype());
    EXPECT_EQ(1, response.outputs(0).shape(0));
    EXPECT_EQ(10, response.outputs(0).shape(1));
    EXPECT_EQ(10, response.outputs(0).contents().fp32_contents_size());
    EXPECT_EQ(0, response.raw_output_contents_size());
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeKFSInferOutputTensor,
    ::testing::ValuesIn(SUPPORTED_KFS_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeKFSInferOutputTensor::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeKFSInferOutputTensorNegative,
    ::testing::ValuesIn(UNSUPPORTED_KFS_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeKFSInferOutputTensorNegative::ParamType>& info) {
        return toString(info.param);
    });

// C-API

class CAPISerialization : public ::testing::TestWithParam<ovms::Precision> {
protected:
    tensor_map_t prepareInputs(ovms::Precision precision, ovms::Shape shape = ovms::Shape{1, 10}) {
        tensor_map_t ret;
        std::shared_ptr<const ovms::TensorInfo> servableOutput =
            std::make_shared<ovms::TensorInfo>(std::string(DUMMY_MODEL_OUTPUT_NAME), precision, shape, Layout{"NC"});
        ret[DUMMY_MODEL_OUTPUT_NAME] = servableOutput;
        return ret;
    }
    InferenceResponse response{"dummy", 1};
};

TEST(SerializeCAPITensorSingle, NegativeMismatchBetweenTensorInfoAndTensorPrecision) {
    InferenceResponse response{"dummy", 1};
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_OUTPUT_NAME,
        ovms::Precision::I32,  // wrong precision
        ovms::Shape{1, 10},
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(ov::element::Type_t::f32, ov::Shape{1, 10});
    float data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10};
    std::memcpy(tensor.data(), data, tensor.get_byte_size());
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST(SerializeCAPITensorSingle, NegativeMismatchBetweenTensorInfoAndTensorShape) {
    InferenceResponse response{"dummy", 1};
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    ovms::tensor_map_t tenMap;
    std::shared_ptr<const ovms::TensorInfo> tensorInfo = std::make_shared<ovms::TensorInfo>(
        DUMMY_MODEL_OUTPUT_NAME,
        ovms::Precision::FP32,
        ovms::Shape{1, 8},  // wrong shape
        Layout{"NC"});
    tenMap[DUMMY_MODEL_OUTPUT_NAME] = tensorInfo;
    ov::Tensor tensor(ov::element::Type_t::f32, ov::Shape{1, 10});
    float data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10};
    std::memcpy(tensor.data(), data, tensor.get_byte_size());
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, tensor);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    auto status = serializePredictResponse(outputGetter, UNUSED_NAME, UNUSED_VERSION, tenMap, &response, getTensorInfoName);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR);
}

class SerializeCAPITensorPositive : public CAPISerialization {};

struct MockedTensorProvider {
    ov::Tensor& tensor;
    MockedTensorProvider(ov::Tensor& tensor) :
        tensor(tensor) {}
};
template <>
Status OutputGetter<MockedTensorProvider&>::get(const std::string& name, ov::Tensor& tensor) {
    tensor = outputSource.tensor;
    return StatusCode::OK;
}

TEST_P(SerializeCAPITensorPositive, SerializeTensorShouldSucceedForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    ov::Tensor tensor(ovmsPrecisionToIE2Precision(testedPrecision), ov::Shape{1, 10});
    MockedTensorProvider provider(tensor);
    OutputGetter<MockedTensorProvider&> outputGetter(provider);

    auto inputs = prepareInputs(testedPrecision);
    auto status = serializePredictResponse(outputGetter,
        UNUSED_NAME,
        UNUSED_VERSION,
        inputs,
        &response,
        getTensorInfoName);
    EXPECT_TRUE(status.ok())
        << "Supported OV serialization precision"
        << toString(testedPrecision)
        << "should succeed";
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeCAPITensorPositive,
    ::testing::ValuesIn(SUPPORTED_CAPI_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeCAPITensorPositive::ParamType>& info) {
        return toString(info.param);
    });

class SerializeCAPITensorNegative : public CAPISerialization {};

TEST_P(SerializeCAPITensorNegative, SerializeTensorShouldFailForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    ov::Tensor tensor(ovmsPrecisionToIE2Precision(testedPrecision), ov::Shape{1, 10});
    MockedTensorProvider provider(tensor);
    OutputGetter<MockedTensorProvider&> outputGetter(provider);

    auto inputs = prepareInputs(testedPrecision);
    auto status = serializePredictResponse(outputGetter,
        UNUSED_NAME,
        UNUSED_VERSION,
        inputs,
        &response,
        getTensorInfoName);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION)
        << "Unsupported OV serialization precision "
        << toString(testedPrecision)
        << " should fail";
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    SerializeCAPITensorNegative,
    ::testing::ValuesIn(UNSUPPORTED_CAPI_OUTPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<SerializeCAPITensorNegative::ParamType>& info) {
        return toString(info.param);
    });

TEST_F(CAPISerialization, ValidSerialization) {
    constexpr size_t NUMBER_OF_ELEMENTS = 3;
    std::array<float, NUMBER_OF_ELEMENTS> data = {3.0, 2.0, 1.0};
    shape_t shape{1, NUMBER_OF_ELEMENTS, 1, 1};
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data(), data.data(), sizeof(float) * NUMBER_OF_ELEMENTS);
    MockedTensorProvider provider(tensor);
    OutputGetter<MockedTensorProvider&> outputGetter(provider);

    auto inputs = prepareInputs(ovms::Precision::FP32, shape);
    auto status = serializePredictResponse(outputGetter,
        UNUSED_NAME,
        UNUSED_VERSION,
        inputs,
        &response,
        getTensorInfoName);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    InferenceTensor* responseOutput{nullptr};
    uint32_t outputCount = response.getOutputCount();
    ASSERT_EQ(1, outputCount);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    const std::string* outputName{nullptr};
    status = response.getOutput(0, &outputName, &responseOutput);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_NE(outputName, nullptr);
    ASSERT_EQ(*outputName, DUMMY_MODEL_OUTPUT_NAME);
    ASSERT_NE(responseOutput, nullptr);
    EXPECT_EQ(responseOutput->getDataType(), OVMS_DATATYPE_FP32);
    EXPECT_THAT(responseOutput->getShape(), ElementsAre(1, NUMBER_OF_ELEMENTS, 1, 1));
    const auto* buffer = responseOutput->getBuffer();
    ASSERT_NE(buffer, nullptr);
    ASSERT_NE(buffer->data(), nullptr);
    EXPECT_EQ(buffer->getByteSize(), tensor.get_byte_size());
    EXPECT_EQ(std::memcmp(tensor.data(), buffer->data(), sizeof(float) * NUMBER_OF_ELEMENTS), 0);
}

template <typename T>
class SerializeString : public ::testing::Test {
public:
    T response;
};

using MyTypes = ::testing::Types<TFPredictResponse, ::KFSResponse>;
TYPED_TEST_SUITE(SerializeString, MyTypes);

// Serialization to string due to suffix _string in mapping
TYPED_TEST(SerializeString, Valid_2D_U8_String) {
    std::vector<uint8_t> data = {
        'S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,
        'z', 'e', 'b', 'r', 'a', 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ov::Tensor tensor(ov::element::u8, ov::Shape{3, 11}, data.data());
    MockedTensorProvider provider(tensor);
    OutputGetter<MockedTensorProvider&> outputGetter(provider);

    ovms::tensor_map_t infos;
    infos["out_string"] = std::make_shared<ovms::TensorInfo>("out", "out_string", ovms::Precision::U8, ovms::Shape{-1, -1}, Layout{"N..."});

    bool useSharedOutputContent = true;
    ASSERT_EQ(serializePredictResponse(outputGetter,
                  UNUSED_NAME,
                  UNUSED_VERSION,
                  infos,
                  &this->response,
                  getTensorInfoName,
                  useSharedOutputContent),
        ovms::StatusCode::OK);
    assertStringResponse(this->response, {"String_123", "zebra", ""}, "out_string");
}

// Serialization to U8 due to missing suffix _string in mapping
TYPED_TEST(SerializeString, Valid_2D_U8_NonString) {
    std::vector<uint8_t> data = {
        'S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,
        'z', 'e', 'b', 'r', 'a', 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ov::Tensor tensor(ov::element::u8, ov::Shape{3, 11}, data.data());
    MockedTensorProvider provider(tensor);
    OutputGetter<MockedTensorProvider&> outputGetter(provider);

    ovms::tensor_map_t infos;
    infos["out_string"] = std::make_shared<ovms::TensorInfo>("out", "out", ovms::Precision::U8, ovms::Shape{-1, -1}, Layout{"N..."});

    bool useSharedOutputContent = false;  // TODO: support raw field
    ASSERT_EQ(serializePredictResponse(outputGetter,
                  UNUSED_NAME,
                  UNUSED_VERSION,
                  infos,
                  &this->response,
                  getTensorInfoName,
                  useSharedOutputContent),
        ovms::StatusCode::OK);
    bool checkRaw = false;  // raw not supported
    checkIncrement4DimResponse("out", data, this->response, std::vector<size_t>{3, 11}, checkRaw);
}
