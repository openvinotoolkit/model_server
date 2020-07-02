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
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../rest_predict_request.hpp"

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

TEST(RestPredictRequestNoNamedJson, RowOrder_2x1x3x1x5) {
    shape_t shape = {2, 1, 3, 1, 5};
    tensor_map_t tensors = {
        {"my_input", std::make_shared<TensorInfo>("my_input", InferenceEngine::Precision::FP32, shape)},
    };
    RestPredictRequest<float> request(tensors);

    ASSERT_EQ(request.parse(R"({"signature_name":"","instances":[
        [
            [
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]]
            ]
        ],
        [
            [
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]]
            ]
        ]
    ]})"), StatusCode::OK);
    EXPECT_EQ(request.getOrder(), Order::ROW);
    EXPECT_EQ(request.getFormat(), Format::NONAMED);
    ASSERT_EQ(request.getInputs().count("my_input"), 1);
    EXPECT_THAT(request.getInputs().at("my_input").shape.get(), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(request.getInputs().at("my_input").data, ElementsAre(
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5));
}

TEST(RestPredictRequestNoNamedJson, ColumnOrder_2x1x3x1x5) {
    shape_t shape = {2, 1, 3, 1, 5};
    tensor_map_t tensors = {
        {"my_input", std::make_shared<TensorInfo>("my_input", InferenceEngine::Precision::FP32, shape)},
    };
    RestPredictRequest<float> request(tensors);

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":[
        [
            [
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]]
            ]
        ],
        [
            [
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5]]
            ]
        ]
    ]})"), StatusCode::OK);
    EXPECT_EQ(request.getOrder(), Order::COLUMN);
    EXPECT_EQ(request.getFormat(), Format::NONAMED);
    ASSERT_EQ(request.getInputs().count("my_input"), 1);
    EXPECT_THAT(request.getInputs().at("my_input").shape.get(), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(request.getInputs().at("my_input").data, ElementsAre(
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5));
}
