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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

#include "../rest_parser.hpp"
#include "absl/strings/escaping.h"
#include "test_utils.hpp"

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

using tensorflow::DataTypeSize;

const char* predictRequestRowNamedJson = R"({
    "instances": [
        {
            "inputA": [
                [[1.0, 2.0],
                 [3.0, 4.0],
                 [5.0, 6.0]],
                [[7.0, 8.0],
                 [9.0, 10.0],
                 [11.0, 12.0]]
            ],
            "inputB": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ]
        },
        {
            "inputA": [
                [[101.0, 102.0],
                 [103.0, 104.0],
                 [105.0, 106.0]],
                [[107.0, 108.0],
                 [109.0, 110.0],
                 [111.0, 112.0]]
            ],
            "inputB": [
                [11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0]
            ]
        }
    ],
    "signature_name": "serving_default"
})";

TEST(TFSRestParserRow, ParseValid2Inputs) {
    TFSRestParser parser(prepareTensors({{"inputA", {2, 2, 3, 2}},
        {"inputB", {2, 2, 3}}}));

    auto status = parser.parse(predictRequestRowNamedJson);

    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs_size(), 2);
    ASSERT_EQ(parser.getProto().inputs().count("inputA"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("inputB"), 1);
    const auto& inputA = parser.getProto().inputs().at("inputA");
    const auto& inputB = parser.getProto().inputs().at("inputB");
    EXPECT_EQ(inputA.dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_EQ(inputB.dtype(), tensorflow::DataType::DT_FLOAT);
    EXPECT_THAT(asVector(inputA.tensor_shape()), ElementsAre(2, 2, 3, 2));
    EXPECT_THAT(asVector(inputB.tensor_shape()), ElementsAre(2, 2, 3));
    ASSERT_EQ(inputA.tensor_content().size(), 2 * 2 * 3 * 2 * DataTypeSize(tensorflow::DataType::DT_FLOAT));
    ASSERT_EQ(inputB.tensor_content().size(), 2 * 2 * 3 * DataTypeSize(tensorflow::DataType::DT_FLOAT));
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
}

TEST(TFSRestParserRow, ValidShape_1x1) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[155.0]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0));
}

TEST(TFSRestParserRow, ValidShape_1x2) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[155.0, 56.0]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1, 2));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 56.0));
}

TEST(TFSRestParserRow, ValidShape_2x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[155.0]}, {"i":[513.0]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 513.0));
}

TEST(TFSRestParserRow, ValidShape_2x2) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[155.0, 9.0]}, {"i":[513.0, -5.0]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 2));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(155.0, 9.0, 513.0, -5.0));
}

TEST(TFSRestParserRow, ValidShape_2x1x3) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 3}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [5.0, 9.0, 2.0]
        ]},
        {"i":[
            [-5.0, -2.0, -10.0]
        ]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 3));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 9.0, 2.0, -5.0, -2.0, -10.0));
}

TEST(TFSRestParserRow, ValidShape_2x3x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 3, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [5.9],
            [9.9],
            [1.9]
        ]},
        {"i":[
            [-1.9],
            [-9.9],
            [25.9]
        ]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 3, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.9, 9.9, 1.9, -1.9, -9.9, 25.9));
}

TEST(TFSRestParserRow, ValidShape_2x1x2x1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 2, 1}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [
                [5.0],
                [2.0]
            ]
        ]},
        {"i":[
            [
                [6.0],
                [18.0]
            ]
        ]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 2, 1));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(5.0, 2.0, 6.0, 18.0));
}

TEST(TFSRestParserRow, ValidShape_2x1x3x1x5) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 3, 1, 5}}}));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [
                [[1.0, 2.0, 3.0, 4.0, 5.0]],
                [[1.0, 2.0, 3.0, 4.0, 5.0]],
                [[1.0, 2.0, 3.0, 4.0, 5.0]]
            ]
        ]},
        {"i":[
            [
                [[1.9, 2.9, 3.9, 4.9, 5.9]],
                [[1.9, 2.9, 3.9, 4.9, 5.9]],
                [[1.9, 2.9, 3.9, 4.9, 5.9]]
            ]
        ]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(
                                                                                          1.0, 2.0, 3.0, 4.0, 5.0,
                                                                                          1.0, 2.0, 3.0, 4.0, 5.0,
                                                                                          1.0, 2.0, 3.0, 4.0, 5.0,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9,
                                                                                          1.9, 2.9, 3.9, 4.9, 5.9));
}

TEST(TFSRestParserRow, MissingInputInBatch) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1, 2, 2}},
        {"j", {1, 1, 2, 2}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[
    {
        "i":[[[5, 2], [10, 7]]],
        "j":[[[5, 2], [10, 7]]]
    },
    {
        "i":[[[5, 2], [10, 7]]]
    }
]})"),
        StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER);
}

TEST(TFSRestParserRow, ParseUint8) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U8));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,5,15,255]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<uint8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 15, 255));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U8));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,5.0,15.0,255.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<uint8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 15, 255));
}

TEST(TFSRestParserRow, ParseInt8) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I8));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,-5,127,-128]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 127, -128));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I8));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,-5.0,127.0,-128.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int8_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 127, -128));
}

TEST(TFSRestParserRow, ParseUint16) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,5,128,65535]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector(parser.getProto().mutable_inputs()->at("i").mutable_int_val()), ElementsAre(0, 5, 128, 65535));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,5.0,128.0,65535.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector(parser.getProto().mutable_inputs()->at("i").mutable_int_val()), ElementsAre(0, 5, 128, 65535));
}

TEST(TFSRestParserRow, ParseInt16) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,-5,32768,-32767]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int16_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 32768, -32767));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,-5.0,32768.0,-32767.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int16_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 32768, -32767));
}

TEST(TFSRestParserRow, ParseInt32) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I32));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,-5,2147483648,-2147483647]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int32_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 2147483648, -2147483647));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I32));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,-5.0,2147483648.0,-2147483647.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int32_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 2147483648, -2147483647));
}

TEST(TFSRestParserRow, ParseUint64) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U64));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,5,128,18446744073709551615]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<uint64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 128, 18446744073709551615UL));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::U64));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,5.0,128.0,555222.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<uint64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, 5, 128, 555222));  // Can't looselessly cast large double to int64
}

TEST(TFSRestParserRow, ParseInt64) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I64));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0,-5,5522,-9223372036854775807]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 5522, -9223372036854775807));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::I64));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[0.0,-5.0,5522.0,-55333.0]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<int64_t>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(0, -5, 5522, -55333));  // Can't looselessly cast double to int64
}

TEST(TFSRestParserRow, ParseFloat) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP32));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[-5, 0, -4, 155234]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(-5, 0, -4, 155234));
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP32));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[-5.12, 0.4344, -4.521, 155234.221]]}]})"), StatusCode::OK);
    EXPECT_THAT(asVector<float>(parser.getProto().inputs().at("i").tensor_content()), ElementsAre(-5.12, 0.4344, -4.521, 155234.221));
}

TEST(TFSRestParserRow, ParseHalf) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[-5, 0, -4, 155234]]}]})"), StatusCode::OK);
    parser = TFSRestParser(prepareTensors({{"i", {1, 1, 4}}}, ovms::Precision::FP16));
    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[-5.1222, 0.434422, -4.52122, 155234.22122]]}]})"), StatusCode::OK);
}

TEST(TFSRestParserRow, InvalidJson) {
    TFSRestParser parser(prepareTensors({{"i", {1, 3, 2}}}));

    EXPECT_EQ(parser.parse(""),
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse("{{}"),
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name:"","instances":[{"i":[1]}]})"),  // missing "
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{i":[1]}]})"),  // missing "
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[1}]})"),  // missing ]
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[1]}])"),  // missing }
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"(["signature_name":"","instances":[{"i":[1]}]})"),  // missing {
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":{[{"i":[1]}]})"),  // too many {
        StatusCode::JSON_INVALID);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[1.0,5.0],[3.0,0.0] [9.0,5.0]]}]})"),  // missing ,
        StatusCode::JSON_INVALID);
}

TEST(TFSRestParserRow, BodyNotAnObject) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse("[]"), StatusCode::REST_BODY_IS_NOT_AN_OBJECT);
    EXPECT_EQ(parser.parse("\"string\""), StatusCode::REST_BODY_IS_NOT_AN_OBJECT);
    EXPECT_EQ(parser.parse("1"), StatusCode::REST_BODY_IS_NOT_AN_OBJECT);
    EXPECT_EQ(parser.parse("null"), StatusCode::REST_BODY_IS_NOT_AN_OBJECT);
}

TEST(TFSRestParserRow, CouldNotDetectOrder) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":""})"), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","bad":[{"i":[1]}]})"), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[1]}],"inputs":{"i":[[1]]}})"), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
}

TEST(TFSRestParserRow, InstancesNotAnArray) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":{}})"), StatusCode::REST_INSTANCES_NOT_AN_ARRAY);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":"string"})"), StatusCode::REST_INSTANCES_NOT_AN_ARRAY);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":5})"), StatusCode::REST_INSTANCES_NOT_AN_ARRAY);
}

TEST(TFSRestParserRow, NamedInstanceNotAnObject) {
    TFSRestParser parser(prepareTensors({{"i", {2, 1}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[5]},2,3]})"), StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[5]},null]})"), StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT);
}

TEST(TFSRestParserRow, CouldNotDetectNamedOrNoNamed) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[null, null]})"), StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED);
}

TEST(TFSRestParserRow, NoInstancesFound) {
    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[]})"), StatusCode::REST_NO_INSTANCES_FOUND);
}

TEST(TFSRestParserRow, CannotParseInstance) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{}]})"), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":2}]})"), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":null}]})"), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[1,null]}]})"), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[[1,2],[3,"str"]]}]})"), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_1) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 4, 5] size is 3 instead of 2 to be valid
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 2],
        [1, 3],
        [1, 4, 5]],
        [[5, 8],
        [9, 3],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_2) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 3}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 2, [8]],
        [1, 3, [3]],
        [1, 4, [5]]],
        [[5, 8, [-1]],
        [9, 3, [-5]],
        [1, 4, [-4]]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_3) {
    TFSRestParser parser(prepareTensors({{"i", {1, 4, 3, 2}}}));

    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 2],
        [1, 3],
        [1, 4]],

        [[1, 2]],

        [[5, 8],
        [9, 3],
        [1, 4]],

        [[5, 8]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_4) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [5, 6] is not a number but array
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 2],
        [1, 3],
        [1, 4, [5, 6]]],
        [[5, 8],
        [9, 3],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_5) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1] is of wrong shape
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1],
        [1, 2],
        [1, 3],
        [1, 4]],
        [[5, 8],
        [9, 3],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_6) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 2, 2}}}));

    // [1, 1] missing - 2x2, 2x3
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 2],
        [1, 3]],
        [[5, 8],
        [9, 3],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_7) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 5] numbers are on wrong level
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [1, 5],
        [[1, 1],
        [1, 2],
        [1, 3]],
        [[5, 8],
        [9, 3],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InputNotNdArray_8) {
    TFSRestParser parser(prepareTensors({{"i", {1, 2, 3, 2}}}));

    // [1, 2], [9, 3] numbers are on wrong level
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[{"i":[
        [[1, 1],
        [[1, 2]],
        [1, 3]],
        [[5, 8],
        [[9, 3]],
        [1, 4]]
    ]}]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InstancesShapeDiffer_1) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 2x2x2
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [[1, 1],
            [1, 2],
            [1, 3]],
            [[5, 8],
            [9, 3],
            [1, 4]]
        ]},
        {"i":[
            [[1, 1],
            [1, 2]],
            [[5, 8],
            [9, 3]]
        ]}
    ]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InstancesShapeDiffer_2) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 2x3x3
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [[1, 1],
            [1, 2],
            [1, 3]],
            [[5, 8],
            [9, 3],
            [1, 4]]
        ]},
        {"i":[
            [[1, 1, 3],
            [1, 2, 2],
            [1, 3, 9]],
            [[5, 8, 8],
            [9, 3, 3],
            [1, 4, 10]]
        ]}
    ]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, InstancesShapeDiffer_3) {
    TFSRestParser parser(prepareTensors({{"i", {2, 2, 3, 2}}}));

    // 2x3x2 vs 1x2x3x2
    EXPECT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"i":[
            [[1, 1],
            [1, 2],
            [1, 3]],
            [[5, 8],
            [9, 3],
            [1, 4]]
        ]},
        {"i":[[
            [[1, 1],
            [1, 2],
            [1, 3]],
            [[5, 8],
            [9, 3],
            [1, 4]]
        ]]}
    ]})"),
        StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST(TFSRestParserRow, RemoveUnnecessaryInputs) {
    TFSRestParser parser(prepareTensors({{"i", {1, 1}}, {"j", {1, 1}}, {"k", {1, 1}}, {"l", {1, 1}}}, ovms::Precision::FP16));

    ASSERT_EQ(parser.parse(R"({"signature_name":"","instances":[
        {"k":[155.0]}, {"l":[1.0]}
    ]})"),
        StatusCode::OK);
    EXPECT_EQ(parser.getOrder(), Order::ROW);
    EXPECT_EQ(parser.getFormat(), Format::NAMED);
    ASSERT_EQ(parser.getProto().inputs().size(), 2);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 0);
    ASSERT_EQ(parser.getProto().inputs().count("j"), 0);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("l"), 1);
}
