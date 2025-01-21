//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/inferencerequest.hpp"
#include "../modelconfig.hpp"
#include "../predict_request_validation_utils.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

using ovms::InferenceRequest;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

class CAPIPredictValidation : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    ovms::InferenceRequest request{"model_name", 1};
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;
    ovms::tensor_map_t servableOutputs;
    bool createCopy{false};
    uint32_t decrementBufferSize{0};
    std::vector<float> requestData{10000000};
    std::vector<float> outputBuffer{10000000};

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);
        std::iota(requestData.begin(), requestData.end(), 1.0);

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

        servableOutputs = ovms::tensor_map_t({
            {"Output_FP32_1_224_224_3_NHWC",
                std::make_shared<ovms::TensorInfo>("Output_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
            {"Output_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Output_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, ovms::Layout{"NCHW"})},
            {"Output_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Output_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, ovms::Layout{"NCDHW"})},
            {"Output_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Output_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, ovms::Layout{"NCHW"})},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getOutputsInfo()).WillByDefault(ReturnRef(servableOutputs));
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
            requestData);
    }
};

TEST_F(CAPIPredictValidation, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(CAPIPredictValidation, AllowScalar) {
    servableInputs = ovms::tensor_map_t({{"Input_FP32_Scalar",
        std::make_shared<ovms::TensorInfo>("Input_FP32_Scalar", ovms::Precision::FP32, ovms::shape_t{}, ovms::Layout{"..."})}});
    requestData = std::vector<float>{2.5f};
    preparePredictRequest(request,
        {{"Input_FP32_Scalar",
            std::tuple<ovms::signed_shape_t, ovms::Precision>{std::vector<int64_t>{}, ovms::Precision::FP32}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok()) << status.string();
}

// Requesting 0 batch via C-API
// C-API mocked endpoints tested: dynamic batch (-1), range (0-100) and static 0.
TEST_F(CAPIPredictValidation, Allow0DimInBatch) {
    std::vector<ovms::Shape> shapes{
        ovms::Shape{ovms::Dimension::any(), 400, 99},   // dynamic
        ovms::Shape{ovms::Dimension{0, 100}, 400, 99},  // range
        ovms::Shape{0, 400, 99}                         // static
    };

    for (const auto& shape : shapes) {
        servableInputs = ovms::tensor_map_t({{"Input",
            std::make_shared<ovms::TensorInfo>("Input", ovms::Precision::FP32, shape, ovms::Layout{"N..."})}});
        requestData = std::vector<float>{};
        preparePredictRequest(request,
            {{"Input",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{0, 400, 99}, ovms::Precision::FP32}}},
            requestData);
        auto status = instance->mockValidate(&request);
        EXPECT_TRUE(status.ok()) << status.string();
    }
}

// Requesting 0 dimension in position other than batch via C-API
// C-API mocked endpoints tested: dynamic shape (-1), range (0-100) and static 0.
TEST_F(CAPIPredictValidation, Allow0DimInShape) {
    std::vector<ovms::Shape> shapes{
        ovms::Shape{20, ovms::Dimension::any(), 400, 99},   // dynamic
        ovms::Shape{20, ovms::Dimension{0, 100}, 400, 99},  // range
        ovms::Shape{20, 0, 400, 99}                         // static
    };

    for (const auto& shape : shapes) {
        servableInputs = ovms::tensor_map_t({{"Input",
            std::make_shared<ovms::TensorInfo>("Input", ovms::Precision::FP32, shape, ovms::Layout{"N..."})}});
        requestData = std::vector<float>{};
        preparePredictRequest(request,
            {{"Input",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{20, 0, 400, 99}, ovms::Precision::FP32}}},
            requestData);
        auto status = instance->mockValidate(&request);
        EXPECT_TRUE(status.ok()) << status.string();
    }
}

TEST_F(CAPIPredictValidation, InvalidPrecision) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, static_cast<ovms::Precision>(99)}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNotEnoughInputs) {
    request.removeInput("Input_U16_1_2_8_4_NCHW");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestTooManyInputs) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestMissingInputName) {
    preparePredictRequest(request,
        {{"BadInput_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(CAPIPredictValidation, RequestWrongInputName) {
    request.removeInput("Input_U16_1_2_8_4_NCHW");
    preparePredictRequest(request,
        {{"BADInput_FP32_1_224_224_3_NHWC",
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestTooManyShapeDimensions) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNotEnoughShapeDimensions) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62, 5}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16, 6}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4, 5}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSize) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);  // dim(0) is batch size

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    preparePredictRequest(request, {{"im_data", {{3, 3, 800, 1344}, ovms::Precision::FP32}}, {"im_info", {{1, 3}, ovms::Precision::FP32}}}, requestData);

    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);

    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}}, {"im_info", {{3, 3}, ovms::Precision::FP32}}}, requestData);

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongAndCorrectShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    preparePredictRequest(request, {{"im_data", {{1, 3, 900, 1344}, ovms::Precision::FP32}}, {"im_info", {{1, 3}, ovms::Precision::FP32}}}, requestData);

    // First is incorrect, second is correct
    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();

    // First is correct, second is incorrect
    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}}, {"im_info", {{1, 6}, ovms::Precision::FP32}}}, requestData);

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}
TEST_F(CAPIPredictValidation, RequestValidBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValues) {
    modelConfig.setBatchingParams("auto");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 17}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNegativeBatchValue) {
    modelConfig.setBatchingParams("auto");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{-1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNegativeShapeValues) {
    modelConfig.setBatchingParams("auto");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, -62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesTwoInputsOneWrong) {  // one input fails validation, request denied
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 17}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 61, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoTwoInputs) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\", \"Input_U16_1_2_8_4_NCHW\": \"auto\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 61, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 2, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoNoNamedInput) {
    modelConfig.parseShapeParameter("auto");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 214, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 61, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 1, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 2, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestValidShapeValuesTwoInputsFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\", \"Input_U16_1_2_8_4_NCHW\": \"(1,2,8,4)\"}");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 4, 63, 63}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesFixedFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{2, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSize) {
    decrementBufferSize = 1;
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectInputWithNoBuffer) {
    servableInputs = ovms::tensor_map_t({{"Input_FP32_1_1_1_1_NHWC",
        std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 1, 1, 1}, ovms::Layout{"NHWC"})}});
    ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));

    InferenceRequest request("NOT_USED", 42);
    std::array<int64_t, 4> shape{1, 1, 1, 1};
    request.addInput("Input_FP32_1_1_1_1_NHWC", OVMS_DATATYPE_FP32, shape.data(), shape.size());
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::NONEXISTENT_BUFFER) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSizeZero) {
    decrementBufferSize = 602112;

    servableInputs = ovms::tensor_map_t({{"Input_FP32_1_224_224_3_NHWC",
        std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})}});
    ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));

    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}}},
        requestData, decrementBufferSize);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectBufferType) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, static_cast<OVMS_BufferType>(999));
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BUFFER_TYPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNegativeBufferType) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, static_cast<OVMS_BufferType>(-22));
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BUFFER_TYPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorectDeviceId) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, OVMS_BUFFERTYPE_CPU, 1);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_DEVICE_ID) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorectBufferType) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, OVMS_BUFFERTYPE_GPU);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BUFFER_TYPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestCorectDeviceId) {
    GTEST_SKIP() << "Enable when Other buffer types are supported";
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, OVMS_BUFFERTYPE_GPU, 1);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNotNullDeviceId) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize, OVMS_BUFFERTYPE_CPU, 1);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_DEVICE_ID) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSizeBatchAuto) {
    modelConfig.setBatchingParams("auto");
    decrementBufferSize = 1;
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSizeShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    decrementBufferSize = 1;
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData, decrementBufferSize);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, ValidRequestWithOutputs) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Output_FP32_1_224_224_3_NHWC", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_FP32), OVMS_BUFFERTYPE_CPU, std::nullopt);
    shape = {1, 3, 62, 62};
    request.addOutput("Output_U8_1_3_62_62_NCHW", OVMS_DATATYPE_U8, shape.data(), 4);
    request.setOutputBuffer("Output_U8_1_3_62_62_NCHW", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_U8), OVMS_BUFFERTYPE_CPU, std::nullopt);
    shape = {1, 6, 128, 128, 16};
    request.addOutput("Output_U8_1_3_62_62_NCHW", OVMS_DATATYPE_I64, shape.data(), 5);
    request.setOutputBuffer("Output_U8_1_3_62_62_NCHW", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_I64), OVMS_BUFFERTYPE_CPU, std::nullopt);
    shape = {1, 2, 8, 4};
    request.addOutput("Output_U8_1_3_62_62_NCHW", OVMS_DATATYPE_U16, shape.data(), 4);
    request.setOutputBuffer("Output_U8_1_3_62_62_NCHW", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_U16), OVMS_BUFFERTYPE_CPU, std::nullopt);

    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(CAPIPredictValidation, OutputWithNoBuffer) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::NONEXISTENT_BUFFER);
}

TEST_F(CAPIPredictValidation, InvalidOutputBufferName) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Invalid", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_FP32), OVMS_BUFFERTYPE_CPU, std::nullopt);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::NONEXISTENT_BUFFER);
}

TEST_F(CAPIPredictValidation, InvalidOutputSize) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Output_FP32_1_224_224_3_NHWC", outputBuffer.data(), 1, OVMS_BUFFERTYPE_CPU, std::nullopt);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(CAPIPredictValidation, InvalidOutputBufferType) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Output_FP32_1_224_224_3_NHWC", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_FP32), (OVMS_BufferType)199, std::nullopt);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BUFFER_TYPE);
}

TEST_F(CAPIPredictValidation, InvalidShape) {
    ovms::signed_shape_t shape = {1, 1, 1, 1};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Output_FP32_1_224_224_3_NHWC", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_FP32), OVMS_BUFFERTYPE_CPU, std::nullopt);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(CAPIPredictValidation, InvalidDeviceId) {
    ovms::signed_shape_t shape = {1, 224, 224, 3};
    request.addOutput("Output_FP32_1_224_224_3_NHWC", OVMS_DATATYPE_FP32, shape.data(), 4);
    request.setOutputBuffer("Output_FP32_1_224_224_3_NHWC", outputBuffer.data(), std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>()) * ovms::DataTypeToByteSize(OVMS_DATATYPE_FP32), OVMS_BUFFERTYPE_CPU, 1);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_DEVICE_ID);
}

class CAPIPredictValidationInputTensorContent : public ::testing::TestWithParam<ovms::Precision> {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockedMetadataModelIns>> instance;
    ovms::InferenceRequest request{"model_name", 1};

    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;
    ovms::tensor_map_t servableOutputs = ovms::tensor_map_t({});
    std::vector<float> requestData{10000000};

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockedMetadataModelIns>>(*ieCore);
        ON_CALL(*instance, getOutputsInfo()).WillByDefault(ReturnRef(servableOutputs));
        std::iota(requestData.begin(), requestData.end(), 1.0);
    }
};

TEST_P(CAPIPredictValidationInputTensorContent, RequestCorrectContentSizeInputTensorContent) {
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
        requestData,  // data,
        false);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongPrecision) {
    preparePredictRequest(request,
        {{"Input_FP32_1_224_224_3_NHWC",
             std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_1_3_62_62_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::Q78}},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
            {"Input_U16_1_2_8_4_NCHW",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}},
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION) << status.string();
}

class CAPIPredictValidationArbitraryBatchPosition : public CAPIPredictValidation {
protected:
    void SetUp() override {
        CAPIPredictValidation::SetUp();

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
            },
            requestData);
    }
};

TEST_F(CAPIPredictValidationArbitraryBatchPosition, Valid) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(CAPIPredictValidationArbitraryBatchPosition, RequestWrongBatchSize) {
    // Edit fourth dimension (N), expect validator to report wrong batch size instead of wrong shape.
    preparePredictRequest(request,
        {
            {"Input_FP32_224_224_3_1_HWCN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{224, 224, 3, 10}, ovms::Precision::FP32}},
            {"Input_U8_3_1_128_CNH",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
        },
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidationArbitraryBatchPosition, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    // Edit fourth dimension (N), expect validator to report batch size change request instead of reshape request.
    preparePredictRequest(request,
        {
            {"Input_FP32_224_224_3_1_HWCN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{224, 224, 3, 10}, ovms::Precision::FP32}},
            {"Input_U8_3_1_128_CNH",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
        },
        requestData);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidationArbitraryBatchPosition, RequestWrongShapeValues) {
    // Edit first dimension (H), expect validator to report wrong shape instead of wrong batch size.
    preparePredictRequest(request,
        {
            {"Input_FP32_224_224_3_1_HWCN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{221, 224, 3, 1}, ovms::Precision::FP32}},
            {"Input_U8_3_1_128_CNH",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
        },
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidationArbitraryBatchPosition, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("auto");
    // Edit first dimension (H), expect validator to report reshape request instead of requesting batch size change.
    preparePredictRequest(request,
        {
            {"Input_FP32_224_224_3_1_HWCN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{10, 224, 3, 1}, ovms::Precision::FP32}},
            {"Input_U8_3_1_128_CNH",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{3, 1, 128}, ovms::Precision::U8}},
        },
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

class CAPIPredictValidationDynamicModel : public CAPIPredictValidation {
protected:
    void SetUp() override {
        CAPIPredictValidation::SetUp();

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
            },
            requestData);
    }
};

TEST_F(CAPIPredictValidationDynamicModel, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(CAPIPredictValidationDynamicModel, RequestBatchNotInRangeFirstPosition) {
    preparePredictRequest(request,
        {
            {"Input_FP32_any_224:512_224:512_3_NHWC",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{16, 300, 320, 3}, ovms::Precision::FP32}},
            {"Input_U8_100:200_any_CN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{101, 16}, ovms::Precision::U8}},
        },
        requestData);

    servableInputs["Input_FP32_any_224:512_224:512_3_NHWC"] = std::make_shared<ovms::TensorInfo>("Input_FP32_any_224:512_224:512_3_NHWC", ovms::Precision::FP32, ovms::Shape{{1, 5}, {224, 512}, {224, 512}, 3}, ovms::Layout{"NHWC"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(CAPIPredictValidationDynamicModel, RequestDimensionNotInRangeFirstPosition) {
    preparePredictRequest(request,
        {
            {"Input_FP32_any_224:512_224:512_3_NHWC",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{16, 300, 320, 3}, ovms::Precision::FP32}},
            {"Input_U8_100:200_any_CN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{98, 1}, ovms::Precision::U8}},
        },
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidationDynamicModel, RequestBatchNotInRangeSecondPosition) {
    preparePredictRequest(request,
        {
            {"Input_FP32_any_224:512_224:512_3_NHWC",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{16, 300, 320, 3}, ovms::Precision::FP32}},
            {"Input_U8_100:200_any_CN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{100, 98}, ovms::Precision::U8}},
        },
        requestData);
    servableInputs["Input_U8_100:200_any_CN"] = std::make_shared<ovms::TensorInfo>("Input_U8_100:200_any_CN", ovms::Precision::U8, ovms::Shape{{100, 200}, {1, 5}}, ovms::Layout{"CN"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidationDynamicModel, RequestDimensionNotInRangeSecondPosition) {
    preparePredictRequest(request,
        {
            {"Input_FP32_any_224:512_224:512_3_NHWC",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 223, 224, 3}, ovms::Precision::FP32}},
            {"Input_U8_100:200_any_CN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{101, 16}, ovms::Precision::U8}},
        },
        requestData);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidationDynamicModel, RequestDimensionInRangeWrongTensorContent) {
    decrementBufferSize = 1;
    preparePredictRequest(request,
        {
            {"Input_FP32_any_224:512_224:512_3_NHWC",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{16, 300, 320, 3}, ovms::Precision::FP32}},
            {"Input_U8_100:200_any_CN",
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{101, 16}, ovms::Precision::U8}},
        },
        requestData, decrementBufferSize);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

class CAPIPredictValidationPrecision : public ::testing::TestWithParam<ovms::Precision> {
protected:
    std::vector<float> requestData{10000000};

    void SetUp() override {
        std::iota(requestData.begin(), requestData.end(), 1.0);
        auto precision = ovms::Precision::FP32;
        mockedInputsInfo[tensorName] = std::make_shared<ovms::TensorInfo>(tensorName, precision, ovms::shape_t{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Layout{"NC"});
    }
    ovms::InferenceRequest request{"model_name", 1};

    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t mockedInputsInfo;
    ovms::tensor_map_t mockedOutputsInfo;
};

TEST_P(CAPIPredictValidationPrecision, ValidPrecisions) {
    ovms::Precision testedPrecision = GetParam();
    mockedInputsInfo[tensorName] = createTensorInfoCopyWithPrecision(mockedInputsInfo[tensorName], testedPrecision);
    preparePredictRequest(request,
        {
            {tensorName,
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, testedPrecision}},
        },
        requestData);
    auto status = ovms::request_validation_utils::validate(request, mockedInputsInfo, mockedOutputsInfo, "dummy", ovms::model_version_t{1});
    EXPECT_EQ(status, ovms::StatusCode::OK) << "Precision validation failed:"
                                            << toString(testedPrecision)
                                            << " should pass validation";
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    CAPIPredictValidationPrecision,
    ::testing::ValuesIn(SUPPORTED_CAPI_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<CAPIPredictValidationPrecision::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    CAPIPredictValidationInputTensorContent,
    ::testing::ValuesIn(SUPPORTED_CAPI_INPUT_PRECISIONS_TENSORINPUTCONTENT),
    [](const ::testing::TestParamInfo<CAPIPredictValidationPrecision::ParamType>& info) {
        return toString(info.param);
    });

#pragma GCC diagnostic pop
