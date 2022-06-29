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

#include "../get_model_metadata_impl.hpp"
#include "test_utils.hpp"

class GetModelMetadataSignature : public ::testing::Test {
    struct Info {
        ovms::Precision precision;
        ovms::shape_t shape;
    };

    using tensor_desc_map_t = std::unordered_map<std::string, Info>;

    tensor_desc_map_t tensors;

protected:
    ovms::tensor_map_t inputs;
    ovms::proto_signature_map_t signature;

    void SetUp() override {
        tensors = tensor_desc_map_t{
            {"Input_FP32_1_3_224_224", {
                                           ovms::Precision::FP32,
                                           {1, 3, 224, 224},
                                       }},
            {"Input_I64_1_6_128_128_16", {
                                             ovms::Precision::I64,
                                             {1, 6, 128, 128, 16},
                                         }},
        };

        for (const auto& pair : tensors) {
            inputs[pair.first] = std::make_shared<ovms::TensorInfo>(
                pair.first,
                pair.second.precision,
                pair.second.shape);
        }
    }
};

TEST_F(GetModelMetadataSignature, ConvertCorrectNumberOfInputs) {
    ovms::GetModelMetadataImpl::convert(inputs, &signature);
    EXPECT_EQ(signature.size(), 2);
}

TEST_F(GetModelMetadataSignature, ConvertInputsExist) {
    ovms::GetModelMetadataImpl::convert(inputs, &signature);
    EXPECT_NE(signature.find("Input_FP32_1_3_224_224"), signature.end());
    EXPECT_NE(signature.find("Input_I64_1_6_128_128_16"), signature.end());
}

TEST_F(GetModelMetadataSignature, ConvertCorrectInputNames) {
    ovms::GetModelMetadataImpl::convert(inputs, &signature);
    EXPECT_EQ(signature["Input_FP32_1_3_224_224"].name(), "Input_FP32_1_3_224_224");
    EXPECT_EQ(signature["Input_I64_1_6_128_128_16"].name(), "Input_I64_1_6_128_128_16");
}

TEST_F(GetModelMetadataSignature, ConvertCorrectPrecision) {
    ovms::GetModelMetadataImpl::convert(inputs, &signature);
    EXPECT_EQ(signature["Input_FP32_1_3_224_224"].dtype(), tensorflow::DT_FLOAT);
    EXPECT_EQ(signature["Input_I64_1_6_128_128_16"].dtype(), tensorflow::DT_INT64);
}

TEST_F(GetModelMetadataSignature, ConvertCorrectTensorShape) {
    ovms::GetModelMetadataImpl::convert(inputs, &signature);

    EXPECT_TRUE(isShapeTheSame(
        signature["Input_FP32_1_3_224_224"].tensor_shape(),
        {1, 3, 224, 224}));

    EXPECT_TRUE(isShapeTheSame(
        signature["Input_I64_1_6_128_128_16"].tensor_shape(),
        {1, 6, 128, 128, 16}));
}
