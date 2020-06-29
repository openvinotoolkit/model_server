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

#include <rapidjson/document.h>

using namespace ovms;

using namespace testing;
using ::testing::ElementsAre;

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
        ]
    },
    "signature_name": "serving_default"
})";

TEST(RestPredictRequestColumnJson, ParseValid2Inputs) {
    RestPredictRequest<float> request;
    auto status = request.parse(predictRequestColumnNamedJson);

    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(request.getOrder(), Order::COLUMN);
    EXPECT_EQ(request.getFormat(), Format::NAMED);
    ASSERT_EQ(request.getInputs().size(), 2);
    ASSERT_EQ(request.getInputs().count("inputA"), 1);
    ASSERT_EQ(request.getInputs().count("inputB"), 1);
    EXPECT_THAT(request.getInputs().at("inputA").shape.get(), ElementsAre(2, 2, 3, 2));
    EXPECT_THAT(request.getInputs().at("inputB").shape.get(), ElementsAre(2, 2, 3));
    EXPECT_EQ(request.getInputs().at("inputA").data.size(), 2 * 2 * 3 * 2);
    EXPECT_EQ(request.getInputs().at("inputB").data.size(), 2 * 2 * 3);
    EXPECT_THAT(request.getInputs().at("inputA").data, ElementsAre(
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
    EXPECT_THAT(request.getInputs().at("inputB").data, ElementsAre(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        //============
        11.0, 12, 13.0,
        14.0, 15.0, 16.0));
}

TEST(RestPredictRequestColumnJson, ParseValidWithPreallocation) {
    shape_t shapes[] = {
        {2, 2, 3, 2},
        {2, 2, 3}
    };
    tensor_map_t tensors = {
        {"inputA", std::make_shared<TensorInfo>("inputA", InferenceEngine::Precision::FP32, shapes[0])},
        {"inputB", std::make_shared<TensorInfo>("inputB", InferenceEngine::Precision::FP32, shapes[1])}
    };
    RestPredictRequest<float> request(tensors);
    auto status = request.parse(predictRequestColumnNamedJson);

    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(request.getOrder(), Order::COLUMN);
    EXPECT_EQ(request.getFormat(), Format::NAMED);
    ASSERT_EQ(request.getInputs().size(), 2);
    ASSERT_EQ(request.getInputs().count("inputA"), 1);
    ASSERT_EQ(request.getInputs().count("inputB"), 1);
    EXPECT_THAT(request.getInputs().at("inputA").shape.get(), ElementsAre(2, 2, 3, 2));
    EXPECT_THAT(request.getInputs().at("inputB").shape.get(), ElementsAre(2, 2, 3));
    EXPECT_EQ(request.getInputs().at("inputA").data.size(), 2 * 2 * 3 * 2);
    EXPECT_EQ(request.getInputs().at("inputB").data.size(), 2 * 2 * 3);
    EXPECT_THAT(request.getInputs().at("inputA").data, ElementsAre(
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
    EXPECT_THAT(request.getInputs().at("inputB").data, ElementsAre(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        //============
        11.0, 12, 13.0,
        14.0, 15.0, 16.0));
}

TEST(RestPredictRequestColumnJson, ValidShape_1x1) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i":[[155]]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(1, 1));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(155));
}

TEST(RestPredictRequestColumnJson, ValidShape_1x2) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i":[[155, 56]]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(1, 2));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(155, 56));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x1) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i":[[155],[513]]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 1));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(155, 513));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x2) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i":[[155, 9], [513, -5]]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 2));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(155, 9, 513, -5));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x1x3) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[5,9,2]],
            [[-5,-2,-10]]
        ]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 1, 3));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(5, 9, 2, -5, -2, -10));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x3x1) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[5], [9], [1]],
            [[-1], [-9], [25]]
        ]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 3, 1));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(5, 9, 1, -1, -9, 25));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x1x2x1) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[5], [2]]],
            [[[6], [18]]]
        ]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 1, 2, 1));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(5, 2, 6, 18));
}

TEST(RestPredictRequestColumnJson, ValidShape_2x1x3x1x5) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[[1, 2, 3, 4, 5]],
              [[1, 2, 3, 4, 5]],
              [[1, 2, 3, 4, 5]]]],
            [[[[1, 2, 3, 4, 5]],
              [[1, 2, 3, 4, 5]],
              [[1, 2, 3, 4, 5]]]]
        ]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 1, 3, 1, 5));
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5));
}

TEST(RestPredictRequestColumnJson, AllowsDifferent0thDimension) {
    RestPredictRequest<float> request;

    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{
        "i": [
            [[[5, 2], [10, 7]]],
            [[[5, 2], [10, 7]]]
        ],
        "j": [
            [[[5, 2], [10, 7]]]
        ]
    }})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").shape.get(), ElementsAre(2, 1, 2, 2));
    EXPECT_THAT(request.getInputs().at("j").shape.get(), ElementsAre(1, 1, 2, 2));
}

TEST(RestPredictRequestColumnJson, ParseUint8) {
    RestPredictRequest<uint8_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,15,255]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 15, 255));
    request = RestPredictRequest<uint8_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,15.0,255.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 15, 255));
}

TEST(RestPredictRequestColumnJson, ParseInt8) {
    RestPredictRequest<int8_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,127,-128]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 127, -128));
    request = RestPredictRequest<int8_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,127.0,-128.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 127, -128));
}

TEST(RestPredictRequestColumnJson, ParseUint16) {
    RestPredictRequest<uint16_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,128,65535]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 65535));
    request = RestPredictRequest<uint16_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,128.0,65535.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 65535));
}

TEST(RestPredictRequestColumnJson, ParseInt16) {
    RestPredictRequest<uint16_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,32768,-32767]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 32768, -32767));
    request = RestPredictRequest<uint16_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,32768.0,-32767.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 32768, -32767));
}

TEST(RestPredictRequestColumnJson, ParseUint32) {
    RestPredictRequest<uint32_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,128,4294967295]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 4294967295));
    request = RestPredictRequest<uint32_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,128.0,4294967295.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 4294967295));
}

TEST(RestPredictRequestColumnJson, ParseInt32) {
    RestPredictRequest<uint32_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,2147483648,-2147483647]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 2147483648, -2147483647));
    request = RestPredictRequest<uint32_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,2147483648.0,-2147483647.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 2147483648, -2147483647));
}

TEST(RestPredictRequestColumnJson, ParseUint64) {
    RestPredictRequest<uint64_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,5,128,18446744073709551615]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 18446744073709551615));
    request = RestPredictRequest<uint64_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,5.0,128.0,555222.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, 5, 128, 555222));  // Can't looselessly cast large double to int64
}

TEST(RestPredictRequestColumnJson, ParseInt64) {
    RestPredictRequest<int64_t> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0,-5,5522,-9223372036854775807]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 5522, -9223372036854775807));
    request = RestPredictRequest<int64_t>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[0.0,-5.0,5522.0,-55333.0]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(0, -5, 5522, -55333));  // Can't looselessly cast double to int64
}

TEST(RestPredictRequestColumnJson, ParseFloat) {
    RestPredictRequest<float> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[-5, 0, -4, 155234]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(-5, 0, -4, 155234));
    request = RestPredictRequest<float>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[-5.12, 0.4344, -4.521, 155234.221]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(-5.12, 0.4344, -4.521, 155234.221));
}

TEST(RestPredictRequestColumnJson, ParseDouble) {
    RestPredictRequest<double> request;
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[-5, 0, -4, 155234]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(-5, 0, -4, 155234));
    request = RestPredictRequest<double>();
    ASSERT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[[-5.1222, 0.434422, -4.52122, 155234.22122]]]}})"), StatusCode::OK);
    EXPECT_THAT(request.getInputs().at("i").data, ElementsAre(-5.1222, 0.434422, -4.52122, 155234.22122));
}

TEST(RestPredictRequestColumnJson, InputsNotAnObject) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":"string"})"), StatusCode::REST_INPUTS_NOT_AN_OBJECT);
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":5})"), StatusCode::REST_INPUTS_NOT_AN_OBJECT);
}

TEST(RestPredictRequestColumnJson, NoNamedInputNotPreallocated) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":[[1, 2]]})"), StatusCode::REST_INPUT_NOT_PREALLOCATED);
}

TEST(RestPredictRequestColumnJson, NoInputsFound) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{}})"), StatusCode::REST_NO_INPUTS_FOUND);
}

TEST(RestPredictRequestColumnJson, CannotParseInput) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":2}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":null}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[1,null]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[[1,2],[3,"str"]]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_1) {
    RestPredictRequest<float> request;

    // [1, 4, 5] size is 3 instead of 2 to be valid
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
          [1, 3],
          [1, 4, 5]],
         [[5, 8],
          [9, 3],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_2) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2, [8]],
          [1, 3, [3]],
          [1, 4, [5]]],
         [[5, 8, [-1]],
          [9, 3, [-5]],
          [1, 4, [-4]]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_3) {
    RestPredictRequest<float> request;

    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
          [1, 3],
          [1, 4]],
 
         [[1, 2]],
 
         [[5, 8],
          [9, 3],
          [1, 4]],
 
         [[5, 8]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_4) {
    RestPredictRequest<float> request;

    // [5, 6] is not a number but array
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
          [1, 3],
          [1, 4, [5, 6]]],
         [[5, 8],
          [9, 3],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_5) {
    RestPredictRequest<float> request;

    // [1] is of wrong shape
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1],
          [1, 2],
          [1, 3],
          [1, 4]],
         [[5, 8],
          [9, 3],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_6) {
    RestPredictRequest<float> request;

    // [1, 1] missing - 2x2, 2x3
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 2],
          [1, 3]],
         [[5, 8],
          [9, 3],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_7) {
    RestPredictRequest<float> request;

    // [1, 5] numbers are on wrong level
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[1, 5],
         [[1, 1],
          [1, 2],
          [1, 3]],
         [[5, 8],
          [9, 3],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InputNotNdArray_8) {
    RestPredictRequest<float> request;

    // [1, 2], [9, 3] numbers are on wrong level
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{"i":[
        [[[1, 1],
          [[1, 2]],
          [1, 3]],
         [[5, 8],
          [[9, 3]],
          [1, 4]]]
    ]}})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InstancesShapeDiffer_1) {
    RestPredictRequest<float> request;

    // 2x3x2 vs 2x2x2
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{
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
    }})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InstancesShapeDiffer_2) {
    RestPredictRequest<float> request;

    // 2x3x2 vs 2x3x3
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{
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
    }})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST(RestPredictRequestColumnJson, InstancesShapeDiffer_3) {
    RestPredictRequest<float> request;

    // 2x3x2 vs 1x2x3x2
    EXPECT_EQ(request.parse(R"({"signature_name":"","inputs":{
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
    }})"), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}
