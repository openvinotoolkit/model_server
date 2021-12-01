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
class PredictValidation : public ::testing::Test {
    class MockModelInstance : public ovms::ModelInstance {
    public:
        MockModelInstance(InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) :
            ModelInstance("UNUSED_NAME", 42, ieCore, ieCore_2) {}
        MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
        MOCK_METHOD(size_t, getBatchSize, (), (const, override));
        MOCK_METHOD(const ovms::ModelConfig&, getModelConfig, (), (const, override));
        const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
            return validate(request);
        }
    };

protected:
    std::unique_ptr<InferenceEngine::Core> ieCore;
    std::unique_ptr<ov::runtime::Core> ieCore_2;
    std::unique_ptr<NiceMock<MockModelInstance>> instance;
    tensorflow::serving::PredictRequest request;
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t networkInputs;

    void SetUp() override {
        ieCore = std::make_unique<InferenceEngine::Core>();
        ieCore_2 = std::make_unique<ov::runtime::Core>();
        instance = std::make_unique<NiceMock<MockModelInstance>>(*ieCore, *ieCore_2);

        networkInputs = ovms::tensor_map_t({
            {"Input_FP32_1_3_224_224_NHWC",
                std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 3, 224, 224}, InferenceEngine::Layout::NHWC)},
            {"Input_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, InferenceEngine::Layout::NCHW)},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Input_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, InferenceEngine::Layout::NCDHW)},
            {"Input_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, InferenceEngine::Layout::NCHW)},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(networkInputs));
        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
        ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));

        request = preparePredictRequest(
            {
                {"Input_FP32_1_3_224_224_NHWC",
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
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.mutable_tensor_shape()->add_dim()->set_size(16);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(PredictValidation, RequestNotEnoughShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
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

    networkInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    networkInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        InferenceEngine::Layout::NHWC);

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

    networkInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    networkInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        InferenceEngine::Layout::NHWC);

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

    networkInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    networkInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        InferenceEngine::Layout::NHWC);

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(PredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    request = preparePredictRequest({{"im_data", {{3, 3, 800, 1344}, tensorflow::DataType::DT_FLOAT}},
        {"im_info", {{1, 3}, tensorflow::DataType::DT_FLOAT}}});

    networkInputs.clear();
    networkInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, InferenceEngine::Layout::NCHW)},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, InferenceEngine::Layout::NC)},
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
    networkInputs.clear();
    networkInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, InferenceEngine::Layout::NCHW)},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, InferenceEngine::Layout::NC)},
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
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.set_dtype(tensorflow::DataType::DT_UINT8);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(PredictValidation, RequestNegativeValueInShape) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(-4);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

#pragma GCC diagnostic pop
