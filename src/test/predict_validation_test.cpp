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

#include <string>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../modelinstance.hpp"

using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::NiceMock;

class PredictValidation : public ::testing::Test {
    using tensor_desc_map_t = std::unordered_map<std::string, InferenceEngine::TensorDesc>;

    class MockModelInstance : public ovms::ModelInstance {
    public:
        MOCK_METHOD(const ovms::tensor_map_t&,  getInputsInfo,  ());
        MOCK_METHOD(size_t,                     getBatchSize,   ());
    };

    std::unordered_map<std::string, InferenceEngine::TensorDesc> tensors;
    ovms::tensor_map_t networkInputs;

protected:
    NiceMock<MockModelInstance> instance;
    tensorflow::serving::PredictRequest request;

    void SetUp() override {
        tensors = tensor_desc_map_t({
            {"Input_FP32_1_3_224_224_NHWC", {
                InferenceEngine::Precision::FP32,
                {1, 3, 224, 224},
                InferenceEngine::Layout::NHWC
            }},
            {"Input_U8_1_3_62_62_NCHW", {
                InferenceEngine::Precision::U8,
                {1, 3, 62, 62},
                InferenceEngine::Layout::NCHW
            }},
            {"Input_I64_1_6_128_128_16_NCDHW", {
                InferenceEngine::Precision::I64,
                {1, 6, 128, 128, 16},
                InferenceEngine::Layout::NCDHW
            }},
            {"Input_U16_1_2_8_4_NCHW", {
                InferenceEngine::Precision::U16,
                {1, 2, 8, 4},
                InferenceEngine::Layout::NCHW
            }},
        });

        for (const auto& pair : tensors) {
            networkInputs[pair.first] = std::make_shared<ovms::TensorInfo>(
                pair.first,
                pair.second.getPrecision(),
                pair.second.getDims(),
                pair.second.getLayout());
        }

        ON_CALL(instance, getInputsInfo()).WillByDefault(ReturnRef(networkInputs));
        ON_CALL(instance, getBatchSize()).WillByDefault(Return(1));

        auto& inputA = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
        inputA.set_dtype(tensorflow::DataType::DT_FLOAT);
        *inputA.mutable_tensor_content() = std::string(1 * 3 * 224 * 224 * 4, '1');
        inputA.mutable_tensor_shape()->add_dim()->set_size(1);
        inputA.mutable_tensor_shape()->add_dim()->set_size(3);
        inputA.mutable_tensor_shape()->add_dim()->set_size(224);
        inputA.mutable_tensor_shape()->add_dim()->set_size(224);

        auto& inputB = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
        inputB.set_dtype(tensorflow::DataType::DT_UINT8);
        *inputB.mutable_tensor_content() = std::string(1 * 3 * 62 * 62 * 1, '1');
        inputB.mutable_tensor_shape()->add_dim()->set_size(1);
        inputB.mutable_tensor_shape()->add_dim()->set_size(3);
        inputB.mutable_tensor_shape()->add_dim()->set_size(62);
        inputB.mutable_tensor_shape()->add_dim()->set_size(62);

        auto& inputC = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
        inputC.set_dtype(tensorflow::DataType::DT_INT64);
        *inputC.mutable_tensor_content() = std::string(1 * 6 * 128 * 128 * 16 * 8, '1');
        inputC.mutable_tensor_shape()->add_dim()->set_size(1);
        inputC.mutable_tensor_shape()->add_dim()->set_size(6);
        inputC.mutable_tensor_shape()->add_dim()->set_size(128);
        inputC.mutable_tensor_shape()->add_dim()->set_size(128);
        inputC.mutable_tensor_shape()->add_dim()->set_size(16);

        // U16 uses int_val instead of tensor_content so it needs separate test
        auto& inputD = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
        inputD.set_dtype(tensorflow::DataType::DT_UINT16);
        inputD.mutable_int_val()->Resize(1 * 2 * 8 * 4, 1);
        inputD.mutable_tensor_shape()->add_dim()->set_size(1);
        inputD.mutable_tensor_shape()->add_dim()->set_size(2);
        inputD.mutable_tensor_shape()->add_dim()->set_size(8);
        inputD.mutable_tensor_shape()->add_dim()->set_size(4);
    }
};

TEST_F(PredictValidation, ValidRequest) {
    auto status = instance.validate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(PredictValidation, RequestNotEnoughInputs) {
    request.mutable_inputs()->erase("Input_U8_1_3_62_62_NCHW");

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(PredictValidation, RequestTooManyInputs) {
    auto& inputD = (*request.mutable_inputs())["input_d"];

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS);
}

TEST_F(PredictValidation, RequestWrongInputName) {
    auto input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    request.mutable_inputs()->erase("Input_I64_1_6_128_128_16_NCDHW");
    (*request.mutable_inputs())["Some_Input"] = input;

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(PredictValidation, RequestTooManyShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.mutable_tensor_shape()->add_dim()->set_size(16);

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(PredictValidation, RequestNotEnoughShapeDimensions) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.mutable_tensor_shape()->clear_dim();

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(PredictValidation, RequestWrongBatchSize) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(10);  // dim(0) is batch size

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(PredictValidation, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs())["Input_U8_1_3_62_62_NCHW"];
    input.mutable_tensor_shape()->mutable_dim(0)->set_size(1);
    input.mutable_tensor_shape()->mutable_dim(1)->set_size(4);
    input.mutable_tensor_shape()->mutable_dim(2)->set_size(63);
    input.mutable_tensor_shape()->mutable_dim(3)->set_size(63);

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(PredictValidation, RequestIncorrectContentSize) {
    auto& input = (*request.mutable_inputs())["Input_I64_1_6_128_128_16_NCDHW"];
    *input.mutable_tensor_content() = std::string(1 * 6, '1');

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PredictValidation, RequestIncorrectValueCount) {
    auto& input = (*request.mutable_inputs())["Input_U16_1_2_8_4_NCHW"];
    input.mutable_int_val()->Clear();
    input.mutable_int_val()->Resize(2, 1);

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(PredictValidation, RequestWrongPrecision) {
    auto& input = (*request.mutable_inputs())["Input_FP32_1_3_224_224_NHWC"];
    input.set_dtype(tensorflow::DataType::DT_UINT8);

    auto status = instance.validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_PRECISION);
}
