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

#pragma warning(push)
#pragma warning(disable : 4624)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../rest_parser.hpp"
#include "test_utils.hpp"

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

using tensorflow::DataTypeSize;

const char* predictRequestColumnNamedJson = R"({
    "inputs": {
        "inputA": [
            [
                [[1.0, 2.0],
                 [3.0, 4.0],
                 [5.0, 6.0]],
                [[7.0, 8.0],
                 [9.0, 10.0],
                 [11.0, 12.0]]
            ],
            [
                [[101.0, 102.0],
                 [103.0, 104.0],
                 [105.0, 106.0]],
                [[107.0, 108.0],
                 [109.0, 110.0],
                 [111.0, 112.0]]
            ]
        ],
        "inputB": [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ],
            [
                [11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0]
            ]
        ],
        "inputC": [
            {"b64": "ORw0"},
            {"b64": "ORw0"}
        ]
    },
    "signature_name": "serving_default"
})";

TEST(TFSRestParserColumn, ParseValid2Inputs) {
    TFSRestParser parser(prepareTensors({{"inputA", {2, 2, 3, 2}},
        {"inputB", {2, 2, 3}}, {"inputC", {2}}}));

    auto status = parser.parse(predictRequestColumnNamedJson);

    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs_size(), 3);
    ASSERT_EQ(parser.getProto().inputs().count("inputA"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("inputB"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("inputC"), 1);
    const auto& inputA = parser.getProto().inputs().at("inputA");
    const auto& inputB = parser.getProto().inputs().at("inputB");
    const auto& inputC = parser.getProto().inputs().at("inputC");
    EXPECT_EQ(inputA.dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_EQ(inputB.dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_EQ(inputC.dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_THAT(asVector(inputA.tensor_shape()), ElementsAre(2, 2, 3, 2));
    EXPECT_THAT(asVector(inputB.tensor_shape()), ElementsAre(2, 2, 3));
    EXPECT_THAT(asVector(inputC.tensor_shape()), ElementsAre(2));
    ASSERT_EQ(inputA.tensor_content().size(), 2 * 2 * 3 * 2 * DataTypeSize(tensorflow::DataType::DT_FLOAT));
    ASSERT_EQ(inputB.tensor_content().size(), 2 * 2 * 3 * DataTypeSize(tensorflow::DataType::DT_FLOAT));
    ASSERT_EQ(inputC.string_val().size(), 2);
    EXPECT_THAT(asVector<float>(inputA.tensor_content()), ElementsAre(
                                                              1.0, 2.0,
                                                              3.0, 4.0,
                                                              5.0, 6.0,
                                                              //-------
                                                              7.0, 8.0,
                                                              9.0, 10.0,
                                                              11.0, 12.0,
                                                              //=========
                                                              101.0, 102.0,
                                                              103.0, 104.0,
                                                              105.0, 106.0,
                                                              //---------
                                                              107.0, 108.0,
                                                              109.0, 110.0,
                                                              111.0, 112.0));
    EXPECT_THAT(asVector<float>(inputB.tensor_content()), ElementsAre(
                                                              1.0, 2.0, 3.0,
                                                              4.0, 5.0, 6.0,
                                                              //============
                                                              11.0, 12, 13.0,
                                                              14.0, 15.0, 16.0));

    char expectedBinary[] = {57, 28, 52};
    EXPECT_EQ(inputC.string_val()[0], std::string(expectedBinary, expectedBinary + 3));
    EXPECT_EQ(inputC.string_val()[1], std::string(expectedBinary, expectedBinary + 3));
}

TEST(TFSRestParserColumn, ValidShape_1d_vector_1elem) {
    TFSRestParser parser(prepareTensors({{"i", {1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[155.0]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0));
}

TEST(TFSRestParserColumn, ValidShape_1x1) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[[155.0]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0));
}

TEST(TFSRestParserColumn, ValidShape_1x2) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[[155.0, 56.0]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1, 2));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 56.0));
}

TEST(TFSRestParserColumn, ValidShape_0) {
    TFSRestParser parser(prepareTensors({{"i", {0}}}, ovms::Precision::FP32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(0));
    EXPECT_EQ(parser.getProto().inputs().at("i").tensor_content().size(), 0);
}

TEST(TFSRestParserColumn, ValidShape_2x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[[155.0],[513.0]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 513.0));
}

TEST(TFSRestParserColumn, ValidShape_2x2) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[[155.0, 9.0], [513.0, -5.0]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 2));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 9.0, 513.0, -5.0));
}

TEST(TFSRestParserColumn, ValidShape_2x0) {
    TFSRestParser parser(prepareTensors({{"i", {2, 0}}}, ovms::Precision::I64));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":[[],[]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_INT64);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 0));
    EXPECT_EQ(parser.getProto().inputs().at("i").tensor_content().size(), 0);
}

TEST(TFSRestParserColumn, ValidShape_2x1x3) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 3}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[5.0,9.0,2.0]],
            [[-5.0,-2.0,-10.0]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 3));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 9.0, 2.0, -5.0, -2.0, -10.0));
}

TEST(TFSRestParserColumn, ValidShape_2x3x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 3, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[5.0], [9.0], [1.0]],
            [[-1.0], [-9.0], [25.0]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 3, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 9.0, 1.0, -1.0, -9.0, 25.0));
}

TEST(TFSRestParserColumn, ValidShape_2x1x2x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 2, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[5.0], [2.0]]],
            [[[6.0], [18.0]]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 2, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 2.0, 6.0, 18.0));
}

TEST(TFSRestParserColumn, ValidShape_2x1x3x1x5) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 3, 1, 5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[[1.9, 2.9, 3.9, 4.9, 5.9]],
            [[1.9, 2.9, 3.9, 4.9, 5.9]],
            [[1.9, 2.9, 3.9, 4.9, 5.9]]]],
            [[[[1.9, 2.9, 3.9, 4.9, 5.9]],
            [[1.9, 2.9, 3.9, 4.9, 5.9]],
            [[1.9, 2.9, 3.9, 4.9, 5.9]]]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9));
}

TEST(TFSRestParserColumn, ValidShape_2x1x3x1x0) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 3, 1, 0}}}, ovms::Precision::FP32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[[ ]],
            [[ ]],
            [[ ]]]],
            [[[[ ]],
            [[ ]],
            [[ ]]]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 3, 1, 0));
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_EQ(parser.getProto().inputs().at("i").tensor_content().size(), 0);
}

TEST(TFSRestParserColumn, ValidScalar) {
    TFSRestParser parser(prepareTensors({{"i", {}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":155.0
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre());
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DT_FLOAT);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0));
}

TEST(TFSRestParserColumn, ValidScalarNoMetadataInt32) {
    TFSRestParser parser(prepareTensors({}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":155
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre());
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DT_INT32);
    EXPECT_THAT(asVector<int>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155));
}

TEST(TFSRestParserColumn, ValidScalarNoMetadataFloat) {
    TFSRestParser parser(prepareTensors({}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i":155.2
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre());
    EXPECT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DT_FLOAT);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.2));
}

TEST(TFSRestParserColumn, AllowsDifferent0thDimension) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 2, 2}},
        {"j", {1, 1, 2, 2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[5.0, 2.0], [10.0, 7.0]]],
            [[[5.0, 2.0], [10.0, 7.0]]]
        ],
        "j": [
            [[[5.0, 2.0], [10.0, 7.0]]]
        ]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 2, 2));
    EXPECT_THAT(asVector(parser.getProto().inputs().at("j").tensor_shape()), ElementsAre(1, 1, 2, 2));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 2.0, 10.0, 7.0, 5.0, 2.0, 10.0, 7.0));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("j").tensor_content()), ElementsAre(5.0, 2.0, 10.0, 7.0));
}

TEST(TFSRestParserColumn, ParseUint8) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U8))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,15,255]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<uint8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 15, 255));
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U8))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,15.0,255.0]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<uint8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 15, 255));
    }
}

TEST(TFSRestParserColumn, ParseInt8) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I8))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,127,-128]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<int8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 127, -128));
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I8))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,127.0,-128.0]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<int8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 127, -128));
    }
}

TEST(TFSRestParserColumn, ParseUint16) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,128,65535]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector(parser.getProto().mutable_inputs()->at("i").mutable_int_val()), ElementsAre(0, 5, 128, 65535));
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,128.0,65535.0]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector(parser.getProto().mutable_inputs()->at("i").mutable_int_val()), ElementsAre(0, 5, 128, 65535));
    }
}

TEST(TFSRestParserColumn, ParseInt16) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,32768,-32767]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<int16_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 32768, -32767));
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,32768.0,-32767.0]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<int16_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 32768, -32767));
    }
}

TEST(TFSRestParserColumn, ParseInt32) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,2147483648,-2147483647]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<int32_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 2147483648, -2147483647));

    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,2147483648,-2147483647]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<int32_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 2147483648, -2147483647));
}

TEST(TFSRestParserColumn, ParseUint64) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U64))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,128,18446744073709551615]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<uint64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 128, 18446744073709551615U));
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U64))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,128.0,555222.0]]]}})"), StatusCode::OK);
        EXPECT_THAT(asVector<uint64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 128, 555222));  // Can't looselessly cast large double to int64
    }
}

TEST(TFSRestParserColumn, ParseInt64) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I64));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,5522,-9223372036854775807]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<int64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 5522, -9223372036854775807));

    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I64));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,5522.0,-55333.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<int64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 5522, -55333));  // Can't looselessly cast double to int64
}

TEST(TFSRestParserColumn, ParseFloat) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[-5.0, 0.0, -4.0, 155234.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(-5.0, 0.0, -4.0, 155234.0));

    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP32));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[-5.12, 0.4344, -4.521, 155234.221]]]}})"), StatusCode::OK);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(-5.12, 0.4344, -4.521, 155234.221));
}

TEST(TFSRestParserColumn, ParseHalf) {
    std::vector<TFSRestParser> parsers{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[-5, 0, -4, 155234]]]}})"), StatusCode::OK);
    }

    parsers = std::vector<TFSRestParser>{TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP16))};
    for (TFSRestParser& parser : parsers) {
        ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[[-5.1222, 0.434422, -4.52122, 155234.22122]]]}})"), StatusCode::OK);
    }
}

TEST(TFSRestParserColumn, InputsNotAnObject) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":"string"})"), StatusCode::REST_INPUTS_NOT_AN_OBJECT);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":5})"), StatusCode::REST_INPUTS_NOT_AN_OBJECT);
}

TEST(TFSRestParserColumn, NoInputsFound) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{}})"), StatusCode::REST_NO_INPUTS_FOUND);
}

TEST(TFSRestParserColumn, CannotParseInput) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":null}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[1,null]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[[1,2],[3,"str"]]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_1) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 4, 5] size is 3 instead of 2 to be valid
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
        [1, 3],
        [1, 4, 5]],
        [[5, 8],
        [9, 3],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_2) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 3}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2, [8]],
        [1, 3, [3]],
        [1, 4, [5]]],
        [[5, 8, [-1]],
        [9, 3, [-5]],
        [1, 4, [-4]]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_3) {
    TFSRestParser parser(prepareTensors({{"i", {1, 4, 3, 2}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
        [1, 3],
        [1, 4]],

        [[1, 2]],

        [[5, 8],
        [9, 3],
        [1, 4]],

        [[5, 8]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_4) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [5, 6] is not a number but array
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
        [1, 3],
        [1, 4, [5, 6]]],
        [[5, 8],
        [9, 3],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_5) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1] is of wrong shape
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1],
        [1, 2],
        [1, 3],
        [1, 4]],
        [[5, 8],
        [9, 3],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_6) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 2, 2}}}));

    // [1, 1] missing - 2x2, 2x3
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
        [1, 3]],
        [[5, 8],
        [9, 3],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_7) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 5] numbers are on wrong level
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[1, 5],
        [[1, 1],
        [1, 2],
        [1, 3]],
        [[5, 8],
        [9, 3],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InputNotNdArray_8) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 2], [9, 3] numbers are on wrong level
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 1],
        [[1, 2]],
        [1, 3]],
        [[5, 8],
        [[9, 3]],
        [1, 4]]]
    ]}})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InstancesShapeDiffer_1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 2x2x2
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [
                [[1, 1],
                [1, 2],
                [1, 3]],
                [[5, 8],
                [9, 3],
                [1, 4]]
            ],
            [
                [[1, 1],
                [1, 2]],
                [[5, 8],
                [9, 3]]
            ]
        ]
    }})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InstancesShapeDiffer_2) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 2x3x3
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [
                [[1, 1],
                [1, 2],
                [1, 3]],
                [[5, 8],
                [9, 3],
                [1, 4]]
            ],
            [
                [[1, 1, 3],
                [1, 2, 2],
                [1, 3, 9]],
                [[5, 8, 8],
                [9, 3, 3],
                [1, 4, 10]]
            ]
        ]
    }})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, InstancesShapeDiffer_3) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 1x2x3x2
    EXPECT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "i": [
            [
                [[1, 1],
                [1, 2],
                [1, 3]],
                [[5, 8],
                [9, 3],
                [1, 4]]
            ],
            [[
                [[1, 1],
                [1, 2],
                [1, 3]],
                [[5, 8],
                [9, 3],
                [1, 4]]
            ]]
        ]
    }})"),
        StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(TFSRestParserColumn, RemoveUnnecessaryInputs) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}, {"j", {1, 1}}, {"k", {1, 1}}, {"l", {1, 1}}, {"m", {}}}, ovms::Precision::FP16));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "k":[[155.0]], "l": [[1.0]]
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("j"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);  // exists in request, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("l"), 1);  // exists in request, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("m"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().size(), 2);
}

TEST(TFSRestParserColumn, RemoveUnnecessaryInputs_ExpectedScalarInRequest) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}, {"j", {1, 1}}, {"k", {1, 1}}, {"l", {1, 1}}, {"m", {}}}, ovms::Precision::FP16));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "k":[[155.0]], "l": [[1.0]], "m": 3
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("j"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);  // exists in request and endpoint metadata, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("l"), 1);  // exists in request and endpoint metadata, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("m"), 1);  // exists in request and endpoint metadata, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().size(), 3);
}

TEST(TFSRestParserColumn, RemoveUnnecessaryInputs_UnexpectedScalarInRequest) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}, {"j", {1, 1}}, /*{"k", {1, 1}},*/ {"l", {1, 1}} /*,{"m", {}}*/}, ovms::Precision::FP16));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","inputs":{
        "k":[[155.0]], "l": [[1.0]], "m": 4
    }})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::COLUMN);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("j"), 0);  // missing in request, expect missing after conversion
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);  // missing in endpoint metadata but exists in request, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("l"), 1);  // exists in request and endpoint metadata, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().count("m"), 1);  // missing in endpoint metadata but exists in request, expect exists after conversion
    ASSERT_EQ(parser.getProto().inputs().size(), 3);
}
