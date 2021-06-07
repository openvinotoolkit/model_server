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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rest_parser.hpp"
#include "absl/strings/escaping.h"
#include "test_utils.hpp"

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

TEST(RestParserNoNamed, RowOrder_2x1x3x1x5) {
    RestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
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
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5));
}

TEST(RestParserNoNamed, RowOrder_5) {
    RestParser parser(prepareTensors({{"my_input", {5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[1,2,3,4,5]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(1, 2, 3, 4, 5));
}

TEST(RestParserNoNamed, ColumnOrder_2x1x3x1x5) {
    RestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[
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
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5,
                                                                1, 2, 3, 4, 5));
}

TEST(RestParserNoNamed, ColumnOrder_5) {
    RestParser parser(prepareTensors({{"my_input", {5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[1,2,3,4,5]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(1, 2, 3, 4, 5));
}
