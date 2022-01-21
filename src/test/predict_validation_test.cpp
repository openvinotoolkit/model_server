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

#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::runtime::Core& ieCore) :
        ModelInstance("UNUSED_NAME", 42, ieCore) {}
    MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
    MOCK_METHOD(ovms::Dimension, getBatchSize, (), (const, override));
    MOCK_METHOD(const ovms::ModelConfig&, getModelConfig, (), (const, override));
    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
        return validate(request);
    }
};

class PredictValidation : public ::testing::Test {
protected:
    std::unique_ptr<ov::runtime::Core> ieCore;
    std::unique_ptr<NiceMock<MockModelInstance>> instance;
    tensorflow::serving::PredictRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::runtime::Core>();
        instance = std::make_unique<NiceMock<MockModelInstance>>(*ieCore);

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

        request = preparePredictRequest(
            {
                {"Input_FP32_1_224_224_3_NHWC",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 224, 224, 3}, tensorflow::DataType::DT_FLOAT}},
                {"Input_U8_1_3_62_62_NCHW",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 3, 62, 62}, tensorflow::DataType::DT_UINT8}},
                {"Input_I64_1_6_128_128_16_NCDHW",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 6, 128, 128, 16}, tensorflow::DataType::DT_INT64}},
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

TEST_F(PredictValidation, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(PredictValidation, RequestNotEnoughInputs) {
    request.mutable_inputs()->erase("Input_U8_1_3_62_62_NCHW");

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(PredictValidation, RequestTooManyInputs) {
    auto& inputD = (*request.mutable_inputs())["input_d"];

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(PredictValidation, RequestWrongInputName) {
    auto input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    request.mutable_inputs()->erase("Input_I64_1_6_128_128_16_NCDHW");
    (*request.mutable_inputs())["Some_Input"] = input;

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(PredictValidation, RequestTooManyShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->add_dim()->set_size(16);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(PredictValidation, RequestNotEnoughShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->clear_dim();

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(PredictValidation, RequestWrongBatchSize) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);  // dim(0) is batch size

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidation, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);  // dim(0) is batch size
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(PredictValidation, ValidRequestBinaryInputs) {
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

TEST_F(PredictValidation, RequestWrongBatchSizeBinaryInputs) {
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

TEST_F(PredictValidation, RequestWrongBatchSizeAutoBinaryInputs) {
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

TEST_F(PredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    request = preparePredictRequest({{"im_data", {{3, 3, 800, 1344}, tensorflow::DataType::DT_FLOAT}},
        {"im_info", {{1, 3}, tensorflow::DataType::DT_FLOAT}}});

    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);

    request = preparePredictRequest({{"im_data", {{1, 3, 800, 1344}, tensorflow::DataType::DT_FLOAT}},
        {"im_info", {{3, 3}, tensorflow::DataType::DT_FLOAT}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(PredictValidation, RequestWrongAndCorrectShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    request = preparePredictRequest({{"im_data", {{1, 3, 900, 1344}, tensorflow::DataType::DT_FLOAT}},
        {"im_info", {{1, 3}, tensorflow::DataType::DT_FLOAT}}});

    // First is incorrect, second is correct
    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);

    // First is correct, second is incorrect
    request = preparePredictRequest({{"im_data", {{1, 3, 800, 1344}, tensorflow::DataType::DT_FLOAT}},
        {"im_info", {{1, 6}, tensorflow::DataType::DT_FLOAT}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(PredictValidation, RequestValidBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(PredictValidation, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(PredictValidation, RequestWrongShapeValuesTwoInputsOneWrong) {  // one input fails validation, request denied
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

TEST_F(PredictValidation, RequestWrongShapeValuesAuto) {
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

TEST_F(PredictValidation, RequestWrongShapeValuesAutoTwoInputs) {
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

TEST_F(PredictValidation, RequestWrongShapeValuesAutoNoNamedInput) {
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

TEST_F(PredictValidation, RequestWrongShapeValuesAutoFirstDim) {
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

TEST_F(PredictValidation, RequestValidShapeValuesTwoInputsFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\", \"Input_U16_1_2_8_4_NCHW\": \"(1,2,8,4)\"}");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(PredictValidation, RequestWrongShapeValuesFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}
TEST_F(PredictValidation, RequestWrongShapeValuesFixedFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");

    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(2);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(3);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(62);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(62);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidation, RequestIncorrectContentSize) {
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    *input.mutable_tensor_content() = std::string(1 * 6, '1');

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PredictValidation, RequestIncorrectContentSizeBatchAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(3);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PredictValidation, RequestIncorrectContentSizeShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(8);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PredictValidation, RequestIncorrectValueCount) {
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_int_val()->Clear();
    input.mutable_int_val()->Resize(2, 1);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(PredictValidation, RequestIncorrectValueCountBatchAuto) {
    modelConfig.setBatchingParams("auto");
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(3);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(PredictValidation, RequestIncorrectValueCountShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(10);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(PredictValidation, RequestWrongPrecision) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.set_dtype(tensorflow::DataType::DT_UINT8);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(PredictValidation, RequestNegativeValueInShape) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_224_224_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(-4);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

class PredictValidationArbitraryBatchPosition : public PredictValidation {
protected:
    void SetUp() override {
        PredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_224_224_3_1_HWCN",
                std::make_shared<ovms::TensorInfo>("Input_FP32_224_224_3_1_HWCN", ovms::Precision::FP32, ovms::shape_t{224, 224, 3, 1}, ovms::Layout{"HWCN"})},
            {"Input_U8_3_1_128_CNH",
                std::make_shared<ovms::TensorInfo>("Input_U8_3_1_128_CNH", ovms::Precision::U8, ovms::shape_t{3, 1, 128}, ovms::Layout{"CNH"})},
        });

        request = preparePredictRequest(
            {
                {"Input_FP32_224_224_3_1_HWCN",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{224, 224, 3, 1}, tensorflow::DataType::DT_FLOAT}},
                {"Input_U8_3_1_128_CNH",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{3, 1, 128}, tensorflow::DataType::DT_UINT8}},
            });
    }
};

TEST_F(PredictValidationArbitraryBatchPosition, Valid) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(PredictValidationArbitraryBatchPosition, RequestWrongBatchSize) {
    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit fourth dimension (N), expect validator to report wrong batch size instead of wrong shape.
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidationArbitraryBatchPosition, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit fourth dimension (N), expect validator to report batch size change request instead of reshape request.
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(PredictValidationArbitraryBatchPosition, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit first dimension (H), expect validator to report wrong shape instead of wrong batch size.
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(PredictValidationArbitraryBatchPosition, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("auto");

    auto& input = (*request.mutable_inputs())["Input_FP32_224_224_3_1_HWCN"];

    // Edit first dimension (H), expect validator to report reshape request instead of requesting batch size change.
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);
    prepareTensorContent(input);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

class PredictValidationDynamicModel : public PredictValidation {
protected:
    void SetUp() override {
        PredictValidation::SetUp();

        servableInputs = ovms::tensor_map_t({{"Input_FP32_any_224:512_224:512_3_NHWC",
                                                 std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_100:200_any_CN",
                std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, ovms::Dimension::any()}, ovms::Layout{"CN"})}});

        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(ovms::Dimension::any()));

        const ovms::dimension_value_t requestBatchSize = 16;
        request = preparePredictRequest(
            {
                {"Input_FP32_any_224:512_224:512_3_NHWC",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{requestBatchSize, 300, 320, 3}, tensorflow::DataType::DT_FLOAT}},
                {"Input_U8_100:200_any_CN",
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{101, requestBatchSize}, tensorflow::DataType::DT_UINT8}},
            });
    }
};

TEST_F(PredictValidationDynamicModel, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(PredictValidationDynamicModel, RequestBatchNotInRangeFirstPosition) {
    auto& input = (*request.mutable_inputs())["Input_FP32_any_224:512_224:512_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(98);  // Should be in 1-5 range

    servableInputs["Input_FP32_any_224:512_224:512_3_NHWC"] = std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{{1, 5}, {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidationDynamicModel, RequestDimensionNotInRangeFirstPosition) {
    auto& input = (*request.mutable_inputs())["Input_U8_100:200_any_CN"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(98);  // Should be in 100-200 range

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(PredictValidationDynamicModel, RequestBatchNotInRangeSecondPosition) {
    auto& input = (*request.mutable_inputs())["Input_U8_100:200_any_CN"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(98);  // Should be in 1-5 range

    servableInputs["Input_U8_100:200_any_CN"] = std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, {1, 5}}, ovms::Layout{"CN"});

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidationDynamicModel, RequestDimensionNotInRangeSecondPosition) {
    auto& input = (*request.mutable_inputs())["Input_FP32_any_224:512_224:512_3_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(223);  // Should be in 224-512 range

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(PredictValidationDynamicModel, RequestDimensionInRangeWrongTensorContent) {
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

#pragma GCC diagnostic pop
