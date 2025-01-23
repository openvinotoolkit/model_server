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
#pragma warning(push)
#pragma warning(disable : 6001)
#include "absl/strings/escaping.h"
#pragma warning(pop)
#include "test_utils.hpp"

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

TEST(TFSRestParserNoNamed, RowOrder_2x1x3x1x5) {
    TFSRestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}));

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

TEST(TFSRestParserNoNamed, RowOrderBinary_2) {
    TFSRestParser parser(prepareTensors({{"my_input", {2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"b64": "ORw0"},
        {"b64": "ORw0"}
        ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2));
    EXPECT_EQ(my_input.dtype(), tensorflow::DataType::DT_STRING);

    char expectedBinary[] = {57, 28, 52};
    EXPECT_EQ(my_input.string_val()[0], std::string(expectedBinary, expectedBinary + 3));
    EXPECT_EQ(my_input.string_val()[1], std::string(expectedBinary, expectedBinary + 3));
}

TEST(TFSRestParserNoNamed, RowOrder_2x1x3x1x0) {
    TFSRestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}, ovms::Precision::I32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        [
            [
                [[ ]],
                [[ ]],
                [[ ]]
            ]
        ],
        [
            [
                [[ ]],
                [[ ]],
                [[ ]]
            ]
        ]
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    EXPECT_EQ(parser.getProto().inputs().at("my_input").dtype(), tensorflow::DT_INT32);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2, 1, 3, 1, 0));
    EXPECT_EQ(my_input.tensor_content().size(), 0);
}

TEST(TFSRestParserNoNamed, Parse2InputsRow) {
    TFSRestParser parser(prepareTensors({{"first", {2}}, {"second", {3}}}));
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
        StatusCode::INVALID_INPUT_FORMAT);
}

TEST(TFSRestParserNoNamed, Parse0InputsRow) {
    TFSRestParser parser(prepareTensors({}));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[1]})"), StatusCode::REST_INPUT_NOT_PREALLOCATED);
}
TEST(TFSRestParserNoNamed, Parse0InputsColumn) {
    TFSRestParser parser(prepareTensors({}));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[1]})"), StatusCode::REST_INPUT_NOT_PREALLOCATED);
}
TEST(TFSRestParserNoNamed, Parse2InputsColumn) {
    TFSRestParser parser(prepareTensors({{"first", {2}}, {"second", {3}}}));
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
        StatusCode::INVALID_INPUT_FORMAT);
}
TEST(TFSRestParserNoNamed, RowOrder_5) {
    TFSRestParser parser(prepareTensors({{"my_input", {5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[1,2,3,4,5]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(1, 2, 3, 4, 5));
}

TEST(TFSRestParserNoNamed, ColumnOrder_2x1x3x1x5) {
    TFSRestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}));

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

TEST(TFSRestParserNoNamed, ColumnOrderBinary_2) {
    TFSRestParser parser(prepareTensors({{"my_input", {2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[
        {"b64": "ORw0"},
        {"b64": "ORw0"}
        ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2));
    EXPECT_EQ(my_input.dtype(), tensorflow::DataType::DT_STRING);

    char expectedBinary[] = {57, 28, 52};
    EXPECT_EQ(my_input.string_val()[0], std::string(expectedBinary, expectedBinary + 3));
    EXPECT_EQ(my_input.string_val()[1], std::string(expectedBinary, expectedBinary + 3));
}

TEST(TFSRestParserNoNamed, ColumnOrder_2x1x3x1x0) {
    TFSRestParser parser(prepareTensors({{"my_input", {2, 1, 3, 1, 5}}}, ovms::Precision::FP32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[
        [
            [
                [[ ]],
                [[ ]],
                [[ ]]
            ]
        ],
        [
            [
                [[ ]],
                [[ ]],
                [[ ]]
            ]
        ]
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_EQ(my_input.dtype(), tensorflow::DT_FLOAT);
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(2, 1, 3, 1, 0));
    EXPECT_EQ(my_input.tensor_content().size(), 0);
}

TEST(TFSRestParserNoNamed, ColumnOrder_1d_5elements) {
    TFSRestParser parser(prepareTensors({{"my_input", {5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":[1,2,3,4,5]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(my_input.dtype(), tensorflow::DT_FLOAT);
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre(5));
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(1, 2, 3, 4, 5));
}

TEST(TFSRestParserNoNamed, ColumnOrder_Scalar) {
    TFSRestParser parser(prepareTensors({{"my_input", {}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":5})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NONAMED);
    ASSERT_EQ(parser.getProto().inputs().count("my_input"), 1);
    const auto& my_input = parser.getProto().inputs().at("my_input");
    EXPECT_THAT(my_input.dtype(), tensorflow::DT_FLOAT);
    EXPECT_THAT(asVector(my_input.tensor_shape()), ElementsAre());
    EXPECT_THAT(asVector<float>(my_input.tensor_content()), ElementsAre(5));
}
