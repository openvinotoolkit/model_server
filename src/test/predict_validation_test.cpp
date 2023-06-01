//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../predict_request_validation_utils.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

class TfsPredictValidation : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    tensorflow::serving::PredictRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_1_224_224_3_NHWC",
                std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, ovms::Layout{"NCHW"})},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Input_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, ovms::Layout{"NCDHW"})},
            {"Input_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, ovms::Layout{"NCHW"})},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
        ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));

        preparePredictRequest(request,
            {
                {"Input_FP32_1_224_224_3_NHWC",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
                {"Input_U8_1_3_62_62_NCHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
                {"Input_I64_1_6_128_128_16_NCDHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            });

        // U16 uses int_val instead of tensor_content so it needs separate test
        auto& inputD = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
        inputD.set_dtype(tensorflow::DataType::DT_UINT16);
        inputD.mutable_int_val()->Resize(1 * 2 * 8 * 4, 1);
        inputD.mutable_tensor_shape()->add_dim()->set_size(1);
        inputD.mutable_tensor_shape()->add_dim()->set_size(2);
        inputD.mutable_tensor_shape()->add_dim()->set_size(8);
        inputD.mutable_tensor_shape()->add_dim()->set_size(4);
    }

    static void prepareTensorContent(tensorflow::TensorProto& proto) {
        if (proto.tensor_shape().dim_size() == 0) {
            *proto.mutable_tensor_content() = "";
            return;
        }
        size_t numberOfElements = 1;
        for (int i = 0; i < proto.tensor_shape().dim_size(); i++) {
            numberOfElements *= proto.tensor_shape().dim(i).size();
        }
        *proto.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(proto.dtype()), '1');
    }
};

TEST_F(TfsPredictValidation, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(TfsPredictValidation, RequestNotEnoughInputs) {
    request.mutable_inputs()->erase("Input_U8_1_3_62_62_NCHW");

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(TfsPredictValidation, RequestTooManyInputs) {
    auto& inputD = (*request.mutable_inputs())["input_d"];

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(TfsPredictValidation, RequestWrongInputName) {
    auto input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    request.mutable_inputs()->erase("Input_I64_1_6_128_128_16_NCDHW");
    (*request.mutable_inputs())["Some_Input"] = input;

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(TfsPredictValidation, RequestTooManyShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->add_dim()->set_size(16);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(TfsPredictValidation, RequestNotEnoughShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->clear_dim();

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(TfsPredictValidation, RequestWrongBatchSize) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);  // dim(0) is batch size

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidation, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);  // dim(0) is batch size
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(TfsPredictValidation, ValidRequestBinaryInputs) {
    modelConfig.setBatchingParams("auto");
    std::string inputName = "Binary_Input";
    tensorflow::serving::PredictRequest binaryInputRequest;

    auto& input = (*binaryInputRequest.mutable_inputs())[inputName];
    input.set_dtype(tensorflow::DataType::DT_STRING);
    const int requestBatchSize = 1;
    for (int i = 0; i < requestBatchSize; i++) {
        input.add_string_val("val");
    }
    input.mutable_tensor_shape()->add_dim()->set_size(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_TRUE(status.ok());
}

TEST_F(TfsPredictValidation, RequestWrongBatchSizeBinaryInputs) {
    std::string inputName = "Binary_Input";
    tensorflow::serving::PredictRequest binaryInputRequest;

    auto& input = (*binaryInputRequest.mutable_inputs())[inputName];
    input.set_dtype(tensorflow::DataType::DT_STRING);
    const int requestBatchSize = 2;
    for (int i = 0; i < requestBatchSize; i++) {
        input.add_string_val("val");
    }
    input.mutable_tensor_shape()->add_dim()->set_size(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidation, RequestWrongBatchSizeAutoBinaryInputs) {
    modelConfig.setBatchingParams("auto");
    std::string inputName = "Binary_Input";
    tensorflow::serving::PredictRequest binaryInputRequest;

    auto& input = (*binaryInputRequest.mutable_inputs())[inputName];
    input.set_dtype(tensorflow::DataType::DT_STRING);
    const int requestBatchSize = 2;
    for (int i = 0; i < requestBatchSize; i++) {
        input.add_string_val("val");
    }
    input.mutable_tensor_shape()->add_dim()->set_size(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    preparePredictRequest(request, {{"im_data", {{3, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);

    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{3, 3}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestWrongAndCorrectShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    preparePredictRequest(request, {{"im_data", {{1, 3, 900, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    // First is incorrect, second is correct
    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);

    // First is correct, second is incorrect
    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 6}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestValidBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesTwoInputsOneWrong) {  // one input fails validation, request denied
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto& input2 = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input2.mutable_tensor_shape()->mutable_dim(0)->set_size(2);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesAutoTwoInputs) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\", \"Input_U16_1_2_8_4_NCHW\": \"auto\"}");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);
    prepareTensorContent(input);

    auto& input2 = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(2);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(16);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(8);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesAutoNoNamedInput) {
    modelConfig.parseShapeParameter("auto");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);
    prepareTensorContent(input);

    auto& input2 = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(2);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(16);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(8);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesAutoFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(2);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(3);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(62);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(62);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(TfsPredictValidation, RequestValidShapeValuesTwoInputsFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\", \"Input_U16_1_2_8_4_NCHW\": \"(1,2,8,4)\"}");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(TfsPredictValidation, RequestWrongShapeValuesFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}
TEST_F(TfsPredictValidation, RequestWrongShapeValuesFixedFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(2);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(3);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(62);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(62);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidation, RequestIncorrectContentSize) {
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    *input.mutable_tensor_content() = std::string(1 * 6, '1');

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(TfsPredictValidation, RequestIncorrectContentSizeBatchAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(3);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(TfsPredictValidation, RequestIncorrectContentSizeShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(8);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(TfsPredictValidation, RequestIncorrectValueCount) {
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_int_val()->Clear();
    input.mutable_int_val()->Resize(2, 1);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(TfsPredictValidation, RequestIncorrectValueCountBatchAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(3);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(TfsPredictValidation, RequestIncorrectValueCountShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(10);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(TfsPredictValidation, RequestWrongPrecision) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.set_dtype(tensorflow::DataType::DT_UINT8);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(TfsPredictValidation, RequestNegativeValueInShape) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(-4);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

class TfsPredictValidationArbitraryBatchPosition : public TfsPredictValidation {
protected:
    void SetUp() override {
        TfsPredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_224_224_3_1_HWCN",
                std::make_shared<ovms::TensorInfo>("Input_FP32_224_224_3_1_HWCN", ovms::Precision::FP32, ovms::shape_t{224, 224, 3, 1}, ovms::Layout{"HWCN"})},
            {"Input_U8_3_1_128_CNH",
                std::make_shared<ovms::TensorInfo>("Input_U8_3_1_128_CNH", ovms::Precision::U8, ovms::shape_t{3, 1, 128}, ovms::Layout{"CNH"})},
        });

        preparePredictRequest(request,
            {
                {"Input_FP32_224_224_3_1_HWCN",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{224, 224, 3, 1}, ovms::Precision::FP32}},
                {"Input_U8_3_1_128_CNH",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
            });
    }
};

TEST_F(TfsPredictValidationArbitraryBatchPosition, Valid) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(TfsPredictValidationArbitraryBatchPosition, RequestWrongBatchSize) {
    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit fourth dimension (N), expect validator to report wrong batch size instead of wrong shape.
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidationArbitraryBatchPosition, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit fourth dimension (N), expect validator to report batch size change request instead of reshape request.
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(TfsPredictValidationArbitraryBatchPosition, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit first dimension (H), expect validator to report wrong shape instead of wrong batch size.
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(TfsPredictValidationArbitraryBatchPosition, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("auto");

    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit first dimension (H), expect validator to report reshape request instead of requesting batch size change.
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

class TfsPredictValidationDynamicModel : public TfsPredictValidation {
protected:
    void SetUp() override {
        TfsPredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({{"Input_FP32_any_224:512_224:512_3_NHWC",
                                                 std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_100:200_any_CN",
                std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, ovms::Dimension::any()}, ovms::Layout{"CN"})}});

        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(ovms::Dimension::any()));

        const ovms::dimension_value_t requestBatchSize = 16;
        preparePredictRequest(request,
            {
                {"Input_FP32_any_224:512_224:512_3_NHWC",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{requestBatchSize, 300, 320, 3}, ovms::Precision::FP32}},
                {"Input_U8_100:200_any_CN",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{101, requestBatchSize}, ovms::Precision::U8}},
            });
    }
};

TEST_F(TfsPredictValidationDynamicModel, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(TfsPredictValidationDynamicModel, RequestBatchNotInRangeFirstPosition) {
    auto& input = (*request.mutable_inputs())["Input_FP32_any_224:512_224:512_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(98);  // Should be in 1-5 range

    servableInputs["Input_FP32_any_224:512_224:512_3_NHWC"] = std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{{1, 5}, {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidationDynamicModel, RequestDimensionNotInRangeFirstPosition) {
    auto& input = (*request.mutable_inputs())["Input_U8_100:200_any_CN"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(98);  // Should be in 100-200 range

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(TfsPredictValidationDynamicModel, RequestBatchNotInRangeSecondPosition) {
    auto& input = (*request.mutable_inputs())["Input_U8_100:200_any_CN"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(98);  // Should be in 1-5 range

    servableInputs["Input_U8_100:200_any_CN"] = std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, {1, 5}}, ovms::Layout{"CN"});

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(TfsPredictValidationDynamicModel, RequestDimensionNotInRangeSecondPosition) {
    auto& input = (*request.mutable_inputs())["Input_FP32_any_224:512_224:512_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(223);  // Should be in 224-512 range

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(TfsPredictValidationDynamicModel, RequestDimensionInRangeWrongTensorContent) {
    auto& input = (*request.mutable_inputs())["Input_U8_100:200_any_CN"];

    size_t numberOfElements = 1;
    for (int i = 0; i < input.tensor_shape().dim_size(); i++) {
        numberOfElements *= input.tensor_shape().dim(i).size();
    }
    numberOfElements -= 1;
    *input.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(input.dtype()), '1');

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

class TfsPredictValidationPrecision : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        auto precision = ovms::Precision::FP32;
        mockedInputsInfo[tensorName] = std::make_shared<ovms::TensorInfo>(tensorName, precision, ovms::shape_t{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Layout{"NC"});
    }
    tensorflow::serving::PredictRequest request;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t mockedInputsInfo;
};

TEST_P(TfsPredictValidationPrecision, ValidPrecisions) {
    ovms::Precision testedPrecision = GetParam();
    mockedInputsInfo[tensorName] = createTensorInfoCopyWithPrecision(mockedInputsInfo[tensorName], testedPrecision);
    preparePredictRequest(request,
        {
            {tensorName,
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, testedPrecision}},
        });
    auto status = ovms::request_validation_utils::validate(request, mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK) << "Precision validation failed:"
                                            << toString(testedPrecision)
                                            << " should pass validation";
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    TfsPredictValidationPrecision,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<TfsPredictValidationPrecision::ParamType>& info) {
        return toString(info.param);
    });

class KFSPredictValidation : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    ::KFSRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_1_224_224_3_NHWC",
                std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, ovms::Layout{"NCHW"})},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Input_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, ovms::Layout{"NCDHW"})},
            {"Input_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, ovms::Layout{"NCHW"})},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
        ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));

        preparePredictRequest(request,
            {{"Input_FP32_1_224_224_3_NHWC",
                 std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
                {"Input_U8_1_3_62_62_NCHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
                {"Input_I64_1_6_128_128_16_NCDHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
                {"Input_U16_1_2_8_4_NCHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}});
    }
};

TEST_F(KFSPredictValidation, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(KFSPredictValidation, RequestNotEnoughInputs) {
    request.mutable_inputs()->RemoveLast();
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(KFSPredictValidation, RequestTooManyInputs) {
    auto inputWrongName = request.add_inputs();
    inputWrongName->set_name("Some_Input");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongInputName) {
    request.mutable_inputs()->RemoveLast();  // remove redundant input
    auto inputWrongName = request.add_inputs();
    inputWrongName->set_name("Some_Input");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT) << status.string();
}

TEST_F(KFSPredictValidation, RequestTooManyShapeDimensions) {
    auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    someInput->mutable_shape()->Add(16);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(KFSPredictValidation, RequestNotEnoughShapeDimensions) {
    auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    someInput->mutable_shape()->Clear();
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongBatchSize) {
    auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    someInput->mutable_shape()->Set(0, 10);                                                    // dim(0) is batch size

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    someInput->mutable_shape()->Set(0, 10);                                                    // dim(0) is batch size. Change from 1
    auto bufferId = request.mutable_inputs()->size() - 1;
    auto previousSize = request.raw_input_contents()[bufferId].size();
    request.mutable_raw_input_contents(bufferId)->assign(size_t(previousSize * 10), '1');
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(KFSPredictValidation, ValidRequestBinaryInputs) {
    std::string inputName = "Binary_Input";
    ::KFSRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 1;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_TRUE(status.ok());
}

TEST_F(KFSPredictValidation, RequestWrongBatchSizeBinaryInputs) {
    std::string inputName = "Binary_Input";
    ::KFSRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 2;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(KFSPredictValidation, RequestWrongBatchSizeAutoBinaryInputs) {
    modelConfig.setBatchingParams("auto");
    std::string inputName = "Binary_Input";
    ::KFSRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 2;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(KFSPredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    preparePredictRequest(request, {{"im_data", {{3, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);

    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{3, 3}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongAndCorrectShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    preparePredictRequest(request, {{"im_data", {{1, 3, 900, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    // First is incorrect, second is correct
    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();

    // First is correct, second is incorrect
    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 6}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}
TEST_F(KFSPredictValidation, RequestValidBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs(request.inputs().size() - 1));
    input.mutable_shape()->RemoveLast();
    input.mutable_shape()->Add(12345);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesTwoInputsOneWrong) {  // one input fails validation, request denied
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    auto& input = (*request.mutable_inputs(request.inputs().size() - 1));
    input.mutable_shape()->RemoveLast();
    input.mutable_shape()->Add(123);
    auto& input2 = (*request.mutable_inputs(request.inputs().size() - 2));
    input2.mutable_shape()->RemoveLast();
    input2.mutable_shape()->Add(123);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 64, 64}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesAutoTwoInputs) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\", \"Input_U16_1_2_8_4_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    prepareKFSInferInputTensor(request, "Input_U16_1_2_8_4_NCHW", {{1, 2, 16, 8}, "UINT16"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesAutoNoNamedInput) {
    modelConfig.parseShapeParameter("auto");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    prepareKFSInferInputTensor(request, "Input_U16_1_2_8_4_NCHW", {{1, 2, 16, 8}, "UINT16"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesAutoFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{2, 3, 62, 62}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

TEST_F(KFSPredictValidation, RequestValidShapeValuesTwoInputsFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\", \"Input_U16_1_2_8_4_NCHW\": \"(1,2,8,4)\"}");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongShapeValuesFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}
TEST_F(KFSPredictValidation, RequestWrongShapeValuesFixedFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{2, 3, 62, 62}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(KFSPredictValidation, RequestIncorrectContentSize) {
    findKFSInferInputTensorContentInRawInputs(request, "Input_I64_1_6_128_128_16_NCDHW")->assign('c', 2);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(KFSPredictValidation, RequestIncorrectContentSizeBatchAuto) {
    modelConfig.setBatchingParams("auto");
    prepareKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW", {{1, 6, 128, 128, 16}, "INT64"});
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(0)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(KFSPredictValidation, RequestIncorrectContentSizeShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    prepareKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW", {{1, 6, 128, 128, 16}, "INT64"});
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(1)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

class KFSPredictValidationInputTensorContent : public ::testing::TestWithParam<ovms::Precision> {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    ::KFSRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);
    }
};

TEST_F(KFSPredictValidationInputTensorContent, RequestInputTensorContentAndRawInputContents) {
    ovms::Precision testedPrecision = ovms::Precision::FP32;
    const std::string inputName = "someName";
    servableInputs = ovms::tensor_map_t({
        {inputName,
            std::make_shared<ovms::TensorInfo>(inputName, testedPrecision, ovms::shape_t{1, 2}, ovms::Layout{"NC"})},
    });
    ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
    ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
    ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));
    preparePredictRequest(request,
        {{inputName,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2}, testedPrecision}}},
        {},      // data,
        false);  // put buffer in InputTensorContent
    auto buf = findKFSInferInputTensor(request, inputName)->mutable_contents()->mutable_fp32_contents();
    buf->Clear();
    buf->Add(1);
    buf->Add(1);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MESSAGE_STRUCTURE) << status.string();
}

TEST_P(KFSPredictValidationInputTensorContent, RequestCorrectContentSizeInputTensorContent) {
    ovms::Precision testedPrecision = GetParam();
    const std::string inputName = "someName";
    servableInputs = ovms::tensor_map_t({
        {inputName,
            std::make_shared<ovms::TensorInfo>(inputName, testedPrecision, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
    });
    ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
    ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
    ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));
    preparePredictRequest(request,
        {{inputName,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, testedPrecision}}},
        {},     // data,
        true);  // put buffer in InputTensorContent
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

class KFSPredictValidationInputTensorContentNegative : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    ::KFSRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_1_224_224_3_NHWC",
                std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, ovms::Layout{"NCHW"})},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Input_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, ovms::Layout{"NCDHW"})},
            {"Input_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, ovms::Layout{"NCHW"})},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
        ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));

        preparePredictRequest(request,
            {{"Input_FP32_1_224_224_3_NHWC",
                 std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
                {"Input_U8_1_3_62_62_NCHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
                {"Input_I64_1_6_128_128_16_NCDHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
                {"Input_U16_1_2_8_4_NCHW",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
            {},     // data,
            true);  // put buffer in InputTensorContent
    }
};

TEST_F(KFSPredictValidationInputTensorContentNegative, RequestIncorrectContentSizeInputTensorContent) {
    auto buf = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW")->mutable_contents()->mutable_int64_contents();
    buf->Clear();
    buf->Add(1);  // There should be 1*6*128*128*16 values
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT) << status.string();
}

TEST_F(KFSPredictValidationInputTensorContentNegative, RequestIncorrectContentSizeBatchAutoInputTensorContent) {
    modelConfig.setBatchingParams("auto");
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(0)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT) << status.string();
}
TEST_F(KFSPredictValidationInputTensorContentNegative, RequestIncorrectContentSizeShapeAutoInputTensorContent) {
    modelConfig.parseShapeParameter("auto");
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(1)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT) << status.string();
}

TEST_F(KFSPredictValidation, RequestWrongPrecision) {
    prepareKFSInferInputTensor(request, "Input_FP32_1_224_224_3_NHWC", {{1, 224, 224, 3}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION) << status.string();
}

TEST_F(KFSPredictValidation, RequestNegativeValueInShape) {
    auto input = findKFSInferInputTensor(request, "Input_FP32_1_224_224_3_NHWC");
    (*input->mutable_shape()->Mutable(1)) = -4;

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

class KFSPredictValidationRawInputContents : public KFSPredictValidation {
protected:
    std::string stringData = "BYTES_CONTENTS";
    uint8_t stringDataSize[4] = {0x0E, 0x00, 0x00, 0x00};
    ::KFSRequest binaryInputRequest;
    ::KFSRequest::InferInputTensor* input;
    std::string inputName = "Binary_Input";
    std::string* content;

    void SetUp() override {
        KFSPredictValidation::SetUp();

        input = binaryInputRequest.add_inputs();
        input->set_name(inputName);
        input->set_datatype("BYTES");
        const int requestBatchSize = 1;
        std::string buffer;
        buffer.append((char*)stringDataSize, 4);
        buffer.append(stringData);
        content = binaryInputRequest.add_raw_input_contents();
        *content = buffer;
        input->mutable_shape()->Add(requestBatchSize);

        servableInputs.clear();
    }
};

TEST_F(KFSPredictValidationRawInputContents, ValidRequest) {
    ovms::shape_t shape = {1, 15};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_TRUE(status.ok());
}

TEST_F(KFSPredictValidationRawInputContents, ValidRequest_BatchSizeBiggerThan1) {
    content->append((char*)stringDataSize, 4);
    content->append(stringData);
    input->mutable_shape()->Clear();
    input->mutable_shape()->Add(2);

    servableInputs.clear();
    ovms::shape_t shape = {2, 15};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(KFSPredictValidationRawInputContents, BatchSizeDoesNotMatchNumberOfStringInBuffer) {
    content->append((char*)stringDataSize, 4);
    content->append(stringData);
    content->append((char*)stringDataSize, 4);
    content->append(stringData);
    input->mutable_shape()->Clear();
    input->mutable_shape()->Add(2);

    servableInputs.clear();
    ovms::shape_t shape = {1, 15};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(KFSPredictValidationRawInputContents, InvalidBatchSize) {
    content->append((char*)stringDataSize, 4);
    content->append(stringData);
    input->mutable_shape()->Clear();
    input->mutable_shape()->Add(2);

    servableInputs.clear();
    ovms::shape_t shape = {1, 15};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(KFSPredictValidationRawInputContents, InvalidWidth) {
    servableInputs.clear();
    ovms::shape_t shape = {1, 10};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(KFSPredictValidationRawInputContents, InvalidBatchSize_BatchSizeChangeRequired) {
    modelConfig.setBatchingParams("auto");
    content->append((char*)stringDataSize, 4);
    content->append(stringData);
    input->mutable_shape()->Clear();
    input->mutable_shape()->Add(2);

    servableInputs.clear();
    ovms::shape_t shape = {1, 15};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(KFSPredictValidationRawInputContents, InvalidWidth_ReshapeRequired) {
    modelConfig.parseShapeParameter("auto");

    servableInputs.clear();
    ovms::shape_t shape = {1, 10};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(KFSPredictValidationRawInputContents, InputTooSmall) {
    uint8_t invalidBuffer[] = {0x0E, 0x00, 0x00};
    binaryInputRequest.mutable_raw_input_contents()->Clear();
    auto invalidContent = binaryInputRequest.add_raw_input_contents();
    invalidContent->append((char*)invalidBuffer, 3);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_INPUT_FORMAT);
}

TEST_F(KFSPredictValidationRawInputContents, InvalidFormat) {
    uint8_t invalidBuffer[] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    binaryInputRequest.mutable_raw_input_contents()->Clear();
    auto invalidContent = binaryInputRequest.add_raw_input_contents();
    invalidContent->append((char*)invalidBuffer, 7);

    servableInputs.clear();
    ovms::shape_t shape = {1};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::U8,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_INPUT_FORMAT);
}

class KFSPredictValidationArbitraryBatchPosition : public KFSPredictValidation {
protected:
    void SetUp() override {
        KFSPredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_224_224_3_1_HWCN",
                std::make_shared<ovms::TensorInfo>("Input_FP32_224_224_3_1_HWCN", ovms::Precision::FP32, ovms::shape_t{224, 224, 3, 1}, ovms::Layout{"HWCN"})},
            {"Input_U8_3_1_128_CNH",
                std::make_shared<ovms::TensorInfo>("Input_U8_3_1_128_CNH", ovms::Precision::U8, ovms::shape_t{3, 1, 128}, ovms::Layout{"CNH"})},
        });

        preparePredictRequest(request,
            {
                {"Input_FP32_224_224_3_1_HWCN",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{224, 224, 3, 1}, ovms::Precision::FP32}},
                {"Input_U8_3_1_128_CNH",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
            });
    }
};

TEST_F(KFSPredictValidationArbitraryBatchPosition, Valid) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(KFSPredictValidationArbitraryBatchPosition, RequestWrongBatchSize) {
    // Edit fourth dimension (N), expect validator to report wrong batch size instead of wrong shape.
    prepareKFSInferInputTensor(request, "Input_FP32_224_224_3_1_HWCN", {{224, 224, 3, 10}, "FP32"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(KFSPredictValidationArbitraryBatchPosition, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    // Edit fourth dimension (N), expect validator to report batch size change request instead of reshape request.
    prepareKFSInferInputTensor(request, "Input_FP32_224_224_3_1_HWCN", {{224, 224, 3, 10}, "FP32"});

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(KFSPredictValidationArbitraryBatchPosition, RequestWrongShapeValues) {
    // Edit first dimension (H), expect validator to report wrong shape instead of wrong batch size.
    prepareKFSInferInputTensor(request, "Input_FP32_224_224_3_1_HWCN", {{10, 224, 3, 1}, "FP32"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(KFSPredictValidationArbitraryBatchPosition, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("auto");
    // Edit first dimension (H), expect validator to report reshape request instead of requesting batch size change.
    prepareKFSInferInputTensor(request, "Input_FP32_224_224_3_1_HWCN", {{10, 224, 3, 1}, "FP32"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

class KFSPredictValidationDynamicModel : public KFSPredictValidation {
protected:
    void SetUp() override {
        KFSPredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({{"Input_FP32_any_224:512_224:512_3_NHWC",
                                                 std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_100:200_any_CN",
                std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, ovms::Dimension::any()}, ovms::Layout{"CN"})}});

        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(ovms::Dimension::any()));

        const ovms::dimension_value_t requestBatchSize = 16;
        preparePredictRequest(request,
            {
                {"Input_FP32_any_224:512_224:512_3_NHWC",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{requestBatchSize, 300, 320, 3}, ovms::Precision::FP32}},
                {"Input_U8_100:200_any_CN",
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{101, requestBatchSize}, ovms::Precision::U8}},
            });
    }
};

TEST_F(KFSPredictValidationDynamicModel, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(KFSPredictValidationDynamicModel, RequestBatchNotInRangeFirstPosition) {
    prepareKFSInferInputTensor(request, "Input_FP32_any_224:512_224:512_3_NHWC", {{16, 300, 320, 3}, "FP32"});

    servableInputs["Input_FP32_any_224:512_224:512_3_NHWC"] = std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{{1, 5}, {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(KFSPredictValidationDynamicModel, RequestDimensionNotInRangeFirstPosition) {
    prepareKFSInferInputTensor(request, "Input_U8_100:200_any_CN", {{98, 1}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(KFSPredictValidationDynamicModel, RequestBatchNotInRangeSecondPosition) {
    prepareKFSInferInputTensor(request, "Input_U8_100:200_any_CN", {{100, 98}, "UINT8"});
    servableInputs["Input_U8_100:200_any_CN"] = std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, {1, 5}}, ovms::Layout{"CN"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(KFSPredictValidationDynamicModel, RequestDimensionNotInRangeSecondPosition) {
    prepareKFSInferInputTensor(request, "Input_FP32_any_224:512_224:512_3_NHWC", {{1, 223, 224, 3}, "FP32"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(KFSPredictValidationDynamicModel, RequestDimensionInRangeWrongTensorContent) {
    findKFSInferInputTensorContentInRawInputs(request, "Input_U8_100:200_any_CN")->clear();
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

class KFSPredictValidationPrecision : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        auto precision = ovms::Precision::FP32;
        mockedInputsInfo[tensorName] = std::make_shared<ovms::TensorInfo>(tensorName, precision, ovms::shape_t{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Layout{"NC"});
    }
    ::KFSRequest request;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t mockedInputsInfo;
};

TEST_P(KFSPredictValidationPrecision, ValidPrecisions) {
    ovms::Precision testedPrecision = GetParam();
    mockedInputsInfo[tensorName] = createTensorInfoCopyWithPrecision(mockedInputsInfo[tensorName], testedPrecision);
    preparePredictRequest(request,
        {
            {tensorName,
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, testedPrecision}},
        });
    auto status = ovms::request_validation_utils::validate(request, mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK) << "Precision validation failed:"
                                            << toString(testedPrecision)
                                            << " should pass validation";
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    KFSPredictValidationPrecision,
    ::testing::ValuesIn(SUPPORTED_KFS_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<KFSPredictValidationPrecision::ParamType>& info) {
        return toString(info.param);
    });
INSTANTIATE_TEST_SUITE_P(
    Test,
    KFSPredictValidationInputTensorContent,
    ::testing::ValuesIn(SUPPORTED_KFS_INPUT_PRECISIONS_TENSORINPUTCONTENT),
    [](const ::testing::TestParamInfo<KFSPredictValidationPrecision::ParamType>& info) {
        return toString(info.param);
    });

static void prepareInferStringInputWithTwoDimensionShapeTensor(::KFSRequest& request, const std::string& name) {
    KFSTensorInputProto* tensor = request.add_inputs();
    tensor->set_name(name);
    tensor->set_datatype("BYTES");
    tensor->mutable_shape()->Clear();
    tensor->add_shape(1);
    tensor->add_shape(1);
}

static void prepareInferStringInputWithTwoDimensionShapeTensor(tensorflow::serving::PredictRequest& request, const std::string& name) {
    request.mutable_inputs()->clear();
    auto& input = (*request.mutable_inputs())[name];
    input.set_dtype(tensorflow::DataType::DT_STRING);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
}

static void prepareInferStringInputWithNegativeShape(::KFSRequest& request, const std::string& name) {
    KFSTensorInputProto* tensor = request.add_inputs();
    tensor->set_name(name);
    tensor->set_datatype("BYTES");
    tensor->mutable_shape()->Clear();
    tensor->add_shape(-5);
}

static void prepareInferStringInputWithNegativeShape(tensorflow::serving::PredictRequest& request, const std::string& name) {
    request.mutable_inputs()->clear();
    auto& input = (*request.mutable_inputs())[name];
    input.set_dtype(tensorflow::DataType::DT_STRING);
    input.mutable_tensor_shape()->add_dim()->set_size(-5);
}

template <typename TensorType>
class PredictValidationString2DTest : public ::testing::Test {
protected:
    TensorType request;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t mockedInputsInfo;
    void SetUp() override {
        auto shape2d = ovms::Shape{-1, -1};
        mockedInputsInfo[tensorName] = std::make_shared<ovms::TensorInfo>(tensorName, ovms::Precision::U8, shape2d, ovms::Layout{"NC"});
    }
};

using MyTypes = ::testing::Types<tensorflow::serving::PredictRequest, ::KFSRequest>;
TYPED_TEST_SUITE(PredictValidationString2DTest, MyTypes);

TYPED_TEST(PredictValidationString2DTest, positive) {
    // bs=1
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
    this->request.Clear();
    // bs=2
    inputStrings = {"String_123", "other"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TYPED_TEST(PredictValidationString2DTest, positive_data_in_buffer) {
    if (typeid(TypeParam) == typeid(TFSRequestType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    // bs=1
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings, false);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
    this->request.Clear();
    // bs=2
    inputStrings = {"String_123", "other"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings, false);
    status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TYPED_TEST(PredictValidationString2DTest, negative_no_string) {
    std::vector<std::string> inputStrings = {};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(PredictValidationString2DTest, negative_over_1gb_after_expansion) {
    std::string longString(1024 * 1024 * 512 * 1, 'a');            // 512mb
    std::vector<std::string> inputStrings = {longString, "", ""};  // sum=1.5gb
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_STRING_MAX_SIZE_EXCEEDED);
}

TYPED_TEST(PredictValidationString2DTest, negative_no_string_in_buffer) {
    if (typeid(TypeParam) == typeid(TFSRequestType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> inputStrings = {};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings, false);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(PredictValidationString2DTest, negative_shape_has_more_dimensions_than_1) {
    prepareInferStringInputWithTwoDimensionShapeTensor(this->request, this->tensorName);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TYPED_TEST(PredictValidationString2DTest, negative_shape_has_negative_shape_value) {
    prepareInferStringInputWithNegativeShape(this->request, this->tensorName);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(PredictValidationString2DTest, batchsize_change_required) {
    this->mockedInputsInfo[this->tensorName] = std::make_shared<ovms::TensorInfo>(this->tensorName, ovms::Precision::U8, ovms::Shape{3, -1}, ovms::Layout{"NC"});
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1}, {}, ovms::Mode::AUTO);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TYPED_TEST(PredictValidationString2DTest, shape_change_required) {
    this->mockedInputsInfo[this->tensorName] = std::make_shared<ovms::TensorInfo>(this->tensorName, ovms::Precision::U8, ovms::Shape{-1, 4}, ovms::Layout{"NC"});
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    ovms::ShapeInfo inputShape{ovms::AUTO, {-1, 4}};
    ovms::shapes_info_map_t shapeMap;
    shapeMap[this->tensorName] = inputShape;
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1}, {}, ovms::Mode::FIXED, shapeMap);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TYPED_TEST(PredictValidationString2DTest, string_not_allowed_with_demultiplexer) {
    this->mockedInputsInfo[this->tensorName] = this->mockedInputsInfo[this->tensorName]->createCopyWithDemultiplexerDimensionPrefix(ovms::Dimension::any());
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::NOT_IMPLEMENTED);
}

template <typename TensorType>
class PredictValidationString1DTest : public ::testing::Test {
protected:
    TensorType request;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t mockedInputsInfo;
    void SetUp() override {
        auto shape1d = ovms::Shape{-1};
        mockedInputsInfo[tensorName] = std::make_shared<ovms::TensorInfo>(tensorName, ovms::Precision::U8, shape1d, ovms::Layout{"NC"});
    }
};

using MyTypes = ::testing::Types<tensorflow::serving::PredictRequest, ::KFSRequest>;
TYPED_TEST_SUITE(PredictValidationString1DTest, MyTypes);

TYPED_TEST(PredictValidationString1DTest, positive) {
    // bs=1
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
    // bs=2
    inputStrings = {"String_123", "other"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TYPED_TEST(PredictValidationString1DTest, negative_wrong_request_shape) {
    prepareInferStringInputWithTwoDimensionShapeTensor(this->request, this->tensorName);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TYPED_TEST(PredictValidationString1DTest, positive_over_1gb) {
    std::string longString(1024 * 1024 * 512 * 1, 'a');            // 512mb
    std::vector<std::string> inputStrings = {longString, "", ""};  // sum=1.5gb
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TYPED_TEST(PredictValidationString1DTest, negative_negative_shape) {
    prepareInferStringInputWithNegativeShape(this->request, this->tensorName);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(PredictValidationString1DTest, string_not_allowed_with_demultiplexer) {
    this->mockedInputsInfo[this->tensorName] = this->mockedInputsInfo[this->tensorName]->createCopyWithDemultiplexerDimensionPrefix(ovms::Dimension::any());
    std::vector<std::string> inputStrings = {"String_123"};
    prepareInferStringRequest(this->request, this->tensorName, inputStrings);
    auto status = ovms::request_validation_utils::validate(this->request, this->mockedInputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::NOT_IMPLEMENTED);
}

#pragma GCC diagnostic pop
