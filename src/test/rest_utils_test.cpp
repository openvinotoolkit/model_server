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

#include "../logging.hpp"
#include "../rest_utils.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;

class Base64DecodeTest : public ::testing::Test {};

TEST_F(Base64DecodeTest, Correct) {
    std::string bytes = "abcd";
    std::string decodedBytes;
    EXPECT_EQ(decodeBase64(bytes, decodedBytes), StatusCode::OK);
    EXPECT_EQ(decodedBytes, "i\xB7\x1D");
}

TEST_F(Base64DecodeTest, WrongLength) {
    std::string bytes = "abcde";
    std::string decodedBytes;
    EXPECT_EQ(decodeBase64(bytes, decodedBytes), StatusCode::REST_BASE64_DECODE_ERROR);
}

class TFSMakeJsonFromPredictResponseRawTest : public ::testing::TestWithParam<ovms::Order> {
protected:
    TFSResponseType proto;
    std::string json;
    TFSOutputTensorType *output1, *output2;

    void SetUp() override {
        output1 = &((*proto.mutable_outputs())["output1"]);
        output2 = &((*proto.mutable_outputs())["output2"]);

        output1->set_dtype(tensorflow::DataType::DT_FLOAT);
        output2->set_dtype(tensorflow::DataType::DT_INT8);

        float data1[8] = {5.0f, 10.0f, -3.0f, 2.5f,
            9.0f, 55.5f, -0.5f, -1.5f};
        output1->mutable_tensor_content()->assign(reinterpret_cast<const char*>(data1), 8 * sizeof(float));
        output1->mutable_tensor_shape()->add_dim()->set_size(2);
        output1->mutable_tensor_shape()->add_dim()->set_size(1);
        output1->mutable_tensor_shape()->add_dim()->set_size(4);

        int8_t data2[10] = {5, 2, 3, 8, -2,
            -100, 0, 125, 4, -1};
        output2->mutable_tensor_content()->assign(reinterpret_cast<const char*>(data2), 10 * sizeof(int8_t));
        output2->mutable_tensor_shape()->add_dim()->set_size(2);
        output2->mutable_tensor_shape()->add_dim()->set_size(5);
    }
};

class TFSMakeJsonFromPredictResponseStringTest : public ::testing::Test {
protected:
    TFSResponseType proto;
    std::string json;
    TFSOutputTensorType* output1;

    void SetUp() override {
        output1 = &((*proto.mutable_outputs())["output1_string"]);
        output1->set_dtype(tensorflow::DataType::DT_STRING);
    }
};

TEST_F(TFSMakeJsonFromPredictResponseStringTest, PositiveRow) {
    output1->add_string_val("Hello");
    output1->mutable_tensor_shape()->add_dim()->set_size(1);
    std::string expected_json = "{\n    \"predictions\": [\"Hello\"\n    ]\n}";
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, expected_json);
}

TEST_F(TFSMakeJsonFromPredictResponseStringTest, PositiveRowBatchSize2) {
    output1->add_string_val("Hello");
    output1->add_string_val("World");
    output1->mutable_tensor_shape()->add_dim()->set_size(2);
    std::string expected_json = "{\n    \"predictions\": [\"Hello\", \"World\"\n    ]\n}";
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, expected_json);
}

TEST_F(TFSMakeJsonFromPredictResponseStringTest, PositiveColumn) {
    output1->add_string_val("Hello");
    output1->mutable_tensor_shape()->add_dim()->set_size(1);
    std::string expected_json = "{\n    \"outputs\": [\n        \"Hello\"\n    ]\n}";
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, expected_json);
}

TEST_F(TFSMakeJsonFromPredictResponseStringTest, PositiveColumnBatchSize2) {
    output1->add_string_val("Hello");
    output1->add_string_val("World");
    output1->mutable_tensor_shape()->add_dim()->set_size(2);
    std::string expected_json = "{\n    \"outputs\": [\n        \"Hello\",\n        \"World\"\n    ]\n}";
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, expected_json);
}

TEST_F(TFSMakeJsonFromPredictResponseRawTest, CannotConvertUnknownOrder) {
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::UNKNOWN), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
}

TEST_F(TFSMakeJsonFromPredictResponseRawTest, CannotConvertInvalidPrecision) {
    output1->set_dtype(tensorflow::DataType::DT_INVALID);
    output1->mutable_tensor_content()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::REST_UNSUPPORTED_PRECISION);
}

const char* rawPositiveFirstOrderResponseRow = R"({
    "predictions": [
        {
            "output1": [[5.0, 10.0, -3.0, 2.5]],
            "output2": [5, 2, 3, 8, -2]
        },
        {
            "output1": [[9.0, 55.5, -0.5, -1.5]],
            "output2": [-100, 0, 125, 4, -1]
        }
    ]
})";

const char* rawPositiveFirstOrderResponseColumn = R"({
    "outputs": {
        "output1": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ],
        "output2": [
            [
                5,
                2,
                3,
                8,
                -2
            ],
            [
                -100,
                0,
                125,
                4,
                -1
            ]
        ]
    }
})";

static std::string getJsonResponseDependsOnOrder(ovms::Order order, const char* rowOrderResponse, const char* columnOrderResponse) {
    switch (order) {
    case Order::ROW:
        return rowOrderResponse;
    case Order::COLUMN:
        return columnOrderResponse;
    default:
        return "";
    }
}

const char* rawPositiveSecondOrderResponseRow = R"({
    "predictions": [
        {
            "output2": [5, 2, 3, 8, -2],
            "output1": [[5.0, 10.0, -3.0, 2.5]]
        },
        {
            "output2": [-100, 0, 125, 4, -1],
            "output1": [[9.0, 55.5, -0.5, -1.5]]
        }
    ]
})";

const char* rawPositiveSecondOrderResponseColumn = R"({
    "outputs": {
        "output2": [
            [
                5,
                2,
                3,
                8,
                -2
            ],
            [
                -100,
                0,
                125,
                4,
                -1
            ]
        ],
        "output1": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ]
    }
})";

TEST_P(TFSMakeJsonFromPredictResponseRawTest, PositiveNamed) {
    auto order = GetParam();
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    bool is_in_first_order = (json == getJsonResponseDependsOnOrder(order, rawPositiveFirstOrderResponseRow, rawPositiveFirstOrderResponseColumn));

    bool is_in_second_order = (json == getJsonResponseDependsOnOrder(order, rawPositiveSecondOrderResponseRow, rawPositiveSecondOrderResponseColumn));

    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}

const char* rawPositiveNonameResponseRow = R"({
    "predictions": [[[5.0, 10.0, -3.0, 2.5]], [[9.0, 55.5, -0.5, -1.5]]
    ]
})";

const char* rawPositiveNonameResponseColumn = R"({
    "outputs": [
        [
            [
                5.0,
                10.0,
                -3.0,
                2.5
            ]
        ],
        [
            [
                9.0,
                55.5,
                -0.5,
                -1.5
            ]
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponseRawTest, Positive_Noname) {
    auto order = GetParam();
    proto.mutable_outputs()->erase("output2");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, rawPositiveNonameResponseRow, rawPositiveNonameResponseColumn));
}

std::vector<ovms::Order> SupportedOrders = {Order::ROW, Order::COLUMN};

static std::string toString(ovms::Order order) {
    switch (order) {
    case Order::ROW:
        return "ROW";
    case Order::COLUMN:
        return "COLUMN";
    default:
        return "";
    }
}

TEST_P(TFSMakeJsonFromPredictResponseRawTest, EmptyTensorContentError) {
    auto order = GetParam();
    output1->mutable_tensor_content()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_SERIALIZE_NO_DATA);
}

TEST_P(TFSMakeJsonFromPredictResponseRawTest, InvalidTensorContentSizeError) {
    auto order = GetParam();
    output1->mutable_tensor_content()->assign("\xFF\xFF\x55\x55", 4);
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE);
}

TEST_P(TFSMakeJsonFromPredictResponseRawTest, ErrorWhenNoOutputs) {
    auto order = GetParam();
    proto.mutable_outputs()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_PROTO_TO_STRING_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    TestGrpcRestResponseConversion,
    TFSMakeJsonFromPredictResponseRawTest,
    ::testing::ValuesIn(SupportedOrders),
    [](const ::testing::TestParamInfo<TFSMakeJsonFromPredictResponseRawTest::ParamType>& info) {
        return toString(info.param);
    });

class TFSMakeJsonFromPredictResponsePrecisionTest : public ::testing::TestWithParam<ovms::Order> {
protected:
    TFSResponseType proto;
    std::string json;
    TFSOutputTensorType* output;

    void SetUp() override {
        output = &((*proto.mutable_outputs())["output"]);
        output->mutable_tensor_shape()->add_dim()->set_size(1);
        output->mutable_tensor_shape()->add_dim()->set_size(1);
    }
};

const char* floatResponseRow = R"({
    "predictions": [[92.5]
    ]
})";

const char* floatResponseColumn = R"({
    "outputs": [
        [
            92.5
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Float) {
    auto order = GetParam();
    float data = 92.5f;
    output->set_dtype(tensorflow::DataType::DT_FLOAT);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(float));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, floatResponseRow, floatResponseColumn));
}

const char* doubleResponseRow = R"({
    "predictions": [[15.99]
    ]
})";

const char* doubleResponseColumn = R"({
    "outputs": [
        [
            15.99
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Double) {
    auto order = GetParam();
    double data = 15.99;
    output->set_dtype(tensorflow::DataType::DT_DOUBLE);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(double));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, GetParam()), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, doubleResponseRow, doubleResponseColumn));
}

const char* int32ResponseRow = R"({
    "predictions": [[-82]
    ]
})";

const char* int32ResponseColumn = R"({
    "outputs": [
        [
            -82
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Int32) {
    auto order = GetParam();
    int32_t data = -82;
    output->set_dtype(tensorflow::DataType::DT_INT32);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int32_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, int32ResponseRow, int32ResponseColumn));
}

const char* int16ResponseRow = R"({
    "predictions": [[-945]
    ]
})";

const char* int16ResponseColumn = R"({
    "outputs": [
        [
            -945
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Int16) {
    auto order = GetParam();
    int16_t data = -945;
    output->set_dtype(tensorflow::DataType::DT_INT16);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int16_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, int16ResponseRow, int16ResponseColumn));
}

const char* int8ResponseRow = R"({
    "predictions": [[-53]
    ]
})";

const char* int8ResponseColumn = R"({
    "outputs": [
        [
            -53
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Int8) {
    auto order = GetParam();
    int8_t data = -53;
    output->set_dtype(tensorflow::DataType::DT_INT8);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int8_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, int8ResponseRow, int8ResponseColumn));
}

const char* uint8ResponseRow = R"({
    "predictions": [[250]
    ]
})";

const char* uint8ResponseColumn = R"({
    "outputs": [
        [
            250
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Uint8) {
    auto order = GetParam();
    uint8_t data = 250;
    output->set_dtype(tensorflow::DataType::DT_UINT8);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint8_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, uint8ResponseRow, uint8ResponseColumn));
}

const char* int64ResponseRow = R"({
    "predictions": [[-658324]
    ]
})";

const char* int64ResponseColumn = R"({
    "outputs": [
        [
            -658324
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Int64) {
    auto order = GetParam();
    int64_t data = -658324;
    output->set_dtype(tensorflow::DataType::DT_INT64);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int64_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, int64ResponseRow, int64ResponseColumn));
}

const char* uint32ResponseRow = R"({
    "predictions": [[1245353]
    ]
})";

const char* uint32ResponseColumn = R"({
    "outputs": [
        [
            1245353
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Uint32) {
    auto order = GetParam();
    uint32_t data = 1245353;
    output->set_dtype(tensorflow::DataType::DT_UINT32);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint32_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, uint32ResponseRow, uint32ResponseColumn));
}

const char* uint64ResponseRow = R"({
    "predictions": [[63456412]
    ]
})";

const char* uint64ResponseColumn = R"({
    "outputs": [
        [
            63456412
        ]
    ]
})";

TEST_P(TFSMakeJsonFromPredictResponsePrecisionTest, Uint64) {
    auto order = GetParam();
    uint64_t data = 63456412;
    output->set_dtype(tensorflow::DataType::DT_UINT64);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint64_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, uint64ResponseRow, uint64ResponseColumn));
}

INSTANTIATE_TEST_SUITE_P(
    TestGrpcRestResponseConversion,
    TFSMakeJsonFromPredictResponsePrecisionTest,
    ::testing::ValuesIn(SupportedOrders),
    [](const ::testing::TestParamInfo<TFSMakeJsonFromPredictResponseRawTest::ParamType>& info) {
        return toString(info.param);
    });

class TFSMakeJsonFromPredictResponseValTest : public ::testing::TestWithParam<ovms::Order> {
protected:
    TFSResponseType proto;
    std::string json;
    TFSOutputTensorType *tensor_content_output, *single_uint64_val, *two_uint32_vals;

    void SetUp() override {
        tensor_content_output = &((*proto.mutable_outputs())["tensor_content_output"]);
        single_uint64_val = &((*proto.mutable_outputs())["single_uint64_val"]);
        two_uint32_vals = &((*proto.mutable_outputs())["two_uint32_vals"]);

        tensor_content_output->set_dtype(tensorflow::DataType::DT_FLOAT);
        single_uint64_val->set_dtype(tensorflow::DataType::DT_UINT64);
        two_uint32_vals->set_dtype(tensorflow::DataType::DT_UINT32);

        float data[8] = {5.0f, 10.0f, -3.0f, 2.5f,
            9.0f, 55.5f, -0.5f, -1.5f};
        tensor_content_output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(data), 8 * sizeof(float));
        tensor_content_output->mutable_tensor_shape()->add_dim()->set_size(2);
        tensor_content_output->mutable_tensor_shape()->add_dim()->set_size(1);
        tensor_content_output->mutable_tensor_shape()->add_dim()->set_size(4);

        single_uint64_val->mutable_tensor_shape()->add_dim()->set_size(1);
        single_uint64_val->add_uint64_val(5000000000);

        two_uint32_vals->mutable_tensor_shape()->add_dim()->set_size(2);
        two_uint32_vals->add_uint32_val(4000000000);
        two_uint32_vals->add_uint32_val(1);
    }
};

TEST_F(TFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainSingleUint64Val) {
    proto.mutable_outputs()->erase("two_uint32_vals");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);

    bool is_in_first_order = json == R"({
    "outputs": {
        "tensor_content_output": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ],
        "single_uint64_val": [
            5000000000
        ]
    }
})";

    bool is_in_second_order = json == R"({
    "outputs": {
        "single_uint64_val": [
            5000000000
        ],
        "tensor_content_output": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ]
    }
})";

    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}

TEST_F(TFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainTwoUint32Vals) {
    proto.mutable_outputs()->erase("single_uint64_val");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);

    bool is_in_first_order = json == R"({
    "outputs": {
        "tensor_content_output": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ],
        "two_uint32_vals": [
            4000000000,
            1
        ]
    }
})";

    bool is_in_second_order = json == R"({
    "outputs": {
        "two_uint32_vals": [
            4000000000,
            1
        ],
        "tensor_content_output": [
            [
                [
                    5.0,
                    10.0,
                    -3.0,
                    2.5
                ]
            ],
            [
                [
                    9.0,
                    55.5,
                    -0.5,
                    -1.5
                ]
            ]
        ]
    }
})";

    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}

class KFSMakeJsonFromPredictResponseRawTest : public ::testing::Test {
protected:
    KFSResponse proto;
    std::string json;
    KFSTensorOutputProto *output1, *output2;
    std::optional<int> inferenceHeaderContentLength;
    const float data1[8] = {5.0f, 10.0f, -3.0f, 2.5f,
        9.0f, 55.5f, -0.5f, -1.5f};
    const int8_t data2[10] = {5, 2, 3, 8, -2,
        -100, 0, 125, 4, -1};

    void SetUp() override {
        proto.set_model_name("model");
        proto.set_id("id");

        output1 = proto.add_outputs();
        output2 = proto.add_outputs();

        output1->set_datatype("FP32");
        output2->set_datatype("INT8");

        output1->set_name("output1");
        output2->set_name("output2");

        auto* output1_contents = proto.add_raw_output_contents();
        output1_contents->assign(reinterpret_cast<const char*>(data1), 8 * sizeof(float));
        output1->mutable_shape()->Add(2);
        output1->mutable_shape()->Add(1);
        output1->mutable_shape()->Add(4);

        auto* output2_contents = proto.add_raw_output_contents();
        output2_contents->assign(reinterpret_cast<const char*>(data2), 10 * sizeof(int8_t));
        output2->mutable_shape()->Add(2);
        output2->mutable_shape()->Add(5);
    }
};

TEST_F(KFSMakeJsonFromPredictResponseRawTest, CannotConvertInvalidPrecision) {
    output1->set_datatype("INVALID");
    proto.mutable_raw_output_contents()->Clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::REST_UNSUPPORTED_PRECISION);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
}

TEST_F(KFSMakeJsonFromPredictResponseRawTest, Positive) {
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output1",
            "shape": [2, 1, 4],
            "datatype": "FP32",
            "data": [5.0, 10.0, -3.0, 2.5, 9.0, 55.5, -0.5, -1.5]
        }, {
            "name": "output2",
            "shape": [2, 5],
            "datatype": "INT8",
            "data": [5, 2, 3, 8, -2, -100, 0, 125, 4, -1]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponseRawTest, EmptyRawOutputContentsError) {
    proto.mutable_raw_output_contents()->Clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::REST_SERIALIZE_NO_DATA);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
}

TEST_F(KFSMakeJsonFromPredictResponseRawTest, InvalidTensorContentSizeError) {
    proto.mutable_raw_output_contents(0)->assign("\xFF\xFF\x55\x55", 4);
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
}

TEST_F(KFSMakeJsonFromPredictResponseRawTest, ErrorWhenNoOutputs) {
    proto.mutable_outputs()->Clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::REST_PROTO_TO_STRING_ERROR);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
}

template <typename T>
static void assertBinaryOutput(T data, std::string json, std::string expectedJson, std::optional<int> inferenceHeaderContentLength) {
    ASSERT_TRUE(inferenceHeaderContentLength.has_value());
    ASSERT_EQ(inferenceHeaderContentLength.value(), expectedJson.size());
    ASSERT_EQ(json.size(), expectedJson.size() + sizeof(T));
    EXPECT_EQ(json.substr(0, inferenceHeaderContentLength.value()), expectedJson);
    EXPECT_EQ(*(T*)json.substr(inferenceHeaderContentLength.value()).data(), data);
}

class KFSMakeJsonFromPredictResponsePrecisionTest : public ::testing::Test {
protected:
    KFSResponse proto;
    std::string json;
    KFSTensorOutputProto* output;
    std::optional<int> inferenceHeaderContentLength;
    std::string outputName = "output";

    void SetUp() override {
        proto.set_model_name("model");
        proto.set_id("id");

        output = proto.add_outputs();
        output->set_name(outputName);
        output->mutable_shape()->Add(1);
        output->mutable_shape()->Add(1);
    }

    template <typename T>
    void prepareData(T data, std::string datatype) {
        output->set_datatype(datatype);
        auto* output_contents = proto.add_raw_output_contents();
        output_contents->assign(reinterpret_cast<const char*>(&data), sizeof(T));
        ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
        ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    }

    template <typename T>
    void prepareDataBinary(T data, std::string datatype) {
        output->set_datatype(datatype);
        auto* output_contents = proto.add_raw_output_contents();
        output_contents->assign(reinterpret_cast<const char*>(&data), sizeof(T));
        std::set<std::string> binaryOutputs;
        binaryOutputs.insert(outputName);
        ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, binaryOutputs), StatusCode::OK);
        ASSERT_EQ(inferenceHeaderContentLength.has_value(), true);
    }

    template <typename T>
    void assertDataBinary(T data, std::string expectedJson) {
        assertBinaryOutput(data, json, expectedJson, inferenceHeaderContentLength);
    }
};

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Float) {
    float data = 92.5f;
    prepareData(data, "FP32");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "FP32",
            "data": [92.5]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Float_binary) {
    float data = 50000000000.99;
    prepareDataBinary(data, "FP32");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "FP32",
            "parameters": {
                "binary_data_size": 4
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Double) {
    double data = 50000000000.99;
    prepareData(data, "FP64");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "FP64",
            "data": [50000000000.99]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Double_binary) {
    double data = 50000000000.99;
    prepareDataBinary(data, "FP64");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "FP64",
            "parameters": {
                "binary_data_size": 8
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int32) {
    int32_t data = -82;
    prepareData(data, "INT32");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [-82]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int32_binary) {
    int32_t data = -82;
    prepareDataBinary(data, "INT32");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT32",
            "parameters": {
                "binary_data_size": 4
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int16) {
    int16_t data = -945;
    prepareData(data, "INT16");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT16",
            "data": [-945]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int16_binary) {
    int16_t data = -945;
    prepareDataBinary(data, "INT16");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT16",
            "parameters": {
                "binary_data_size": 2
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int8) {
    int8_t data = -53;
    prepareData(data, "INT8");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT8",
            "data": [-53]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int8_binary) {
    int8_t data = -53;
    prepareDataBinary(data, "INT8");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT8",
            "parameters": {
                "binary_data_size": 1
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint8) {
    uint8_t data = 250;
    prepareData(data, "UINT8");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT8",
            "data": [250]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint8_binary) {
    uint8_t data = 250;
    prepareDataBinary(data, "UINT8");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT8",
            "parameters": {
                "binary_data_size": 1
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int64) {
    int64_t data = -658324;
    prepareData(data, "INT64");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT64",
            "data": [-658324]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Int64_binary) {
    int64_t data = -658324;
    prepareDataBinary(data, "INT64");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "INT64",
            "parameters": {
                "binary_data_size": 8
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint32) {
    uint32_t data = 1245353;
    prepareData(data, "UINT32");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT32",
            "data": [1245353]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint32_binary) {
    uint32_t data = 1245353;
    prepareDataBinary(data, "UINT32");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT32",
            "parameters": {
                "binary_data_size": 4
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint64) {
    uint64_t data = 63456412;
    prepareData(data, "UINT64");
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT64",
            "data": [63456412]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, Uint64_binary) {
    uint64_t data = 63456412;
    prepareDataBinary(data, "UINT64");
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [1, 1],
            "datatype": "UINT64",
            "parameters": {
                "binary_data_size": 8
            }
        }]
})";
    assertDataBinary(data, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, raw_BYTES_string) {
    uint8_t data[] = {4, 0, 0, 0, 'a', 'b', 'c', 'd', 3, 0, 0, 0, 'e', 'f', 'g'};
    int dataSize = 15;
    output->set_datatype("BYTES");
    output->mutable_shape()->Clear();
    output->mutable_shape()->Add(2);  // batch size
    auto* output_contents = proto.add_raw_output_contents();
    output_contents->assign(reinterpret_cast<const char*>(&data), dataSize);
    auto status = makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [2],
            "datatype": "BYTES",
            "data": ["abcd", "efg"]
        }]
})";
    SPDLOG_INFO(json);
    ASSERT_EQ(json.size(), expectedJson.size());
    ASSERT_EQ(json, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponsePrecisionTest, raw_BYTES_string_binary) {
    uint8_t data[] = {4, 0, 0, 0, 'a', 'b', 'c', 'd', 3, 0, 0, 0, 'e', 'f', 'g'};
    int dataSize = 15;
    output->set_datatype("BYTES");
    output->mutable_shape()->Clear();
    output->mutable_shape()->Add(2);  // batch size
    auto* output_contents = proto.add_raw_output_contents();
    output_contents->assign(reinterpret_cast<const char*>(&data), dataSize);
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, {"output"}), StatusCode::OK);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "output",
            "shape": [2],
            "datatype": "BYTES",
            "parameters": {
                "binary_data_size": 15
            }
        }]
})";
    ASSERT_TRUE(inferenceHeaderContentLength.has_value());
    ASSERT_EQ(inferenceHeaderContentLength.value(), expectedJson.size());
    ASSERT_EQ(json.size(), expectedJson.size() + dataSize);
    EXPECT_EQ(json.substr(0, inferenceHeaderContentLength.value()), expectedJson);
    for (int i = 0; i < dataSize; i++) {
        EXPECT_EQ((uint8_t)json.substr(inferenceHeaderContentLength.value())[i], data[i]);
    }
}

class KFSMakeJsonFromPredictResponseValTest : public ::testing::Test {
protected:
    KFSResponse proto;
    std::string json;
    KFSTensorOutputProto *single_uint64_val, *two_uint32_vals;
    std::optional<int> inferenceHeaderContentLength;

    void SetUp() override {
        proto.set_model_name("model");
        proto.set_id("id");

        single_uint64_val = proto.add_outputs();
        two_uint32_vals = proto.add_outputs();

        single_uint64_val->set_name("single_uint64_val");
        two_uint32_vals->set_name("two_uint32_vals");

        single_uint64_val->set_datatype("UINT64");
        two_uint32_vals->set_datatype("UINT32");

        single_uint64_val->mutable_shape()->Add(1);
        auto single_uint64_val_1 = single_uint64_val->mutable_contents()->mutable_uint64_contents()->Add();
        *single_uint64_val_1 = 5000000000;

        two_uint32_vals->mutable_shape()->Add(2);
        auto two_uint32_vals_1 = two_uint32_vals->mutable_contents()->mutable_uint_contents()->Add();
        auto two_uint32_vals_2 = two_uint32_vals->mutable_contents()->mutable_uint_contents()->Add();
        *two_uint32_vals_1 = 4000000000;
        *two_uint32_vals_2 = 1;
    }
};

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_Positive_BYTES) {
    KFSTensorOutputProto* bytes_val_proto = proto.add_outputs();
    bytes_val_proto->set_name("bytes_val_proto");
    bytes_val_proto->set_datatype("BYTES");
    bytes_val_proto->mutable_shape()->Add(2);
    auto bytes_val = bytes_val_proto->mutable_contents()->mutable_bytes_contents()->Add();
    bytes_val->assign("string_1");
    bytes_val = bytes_val_proto->mutable_contents()->mutable_bytes_contents()->Add();
    bytes_val->assign("string_2");

    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000]
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }, {
            "name": "bytes_val_proto",
            "shape": [2],
            "datatype": "BYTES",
            "data": ["string_1", "string_2"]
        }]
})";
    ASSERT_EQ(json.size(), expectedJson.size());
    EXPECT_EQ(json, expectedJson);
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_Positive_BYTES_binary) {
    KFSTensorOutputProto* bytes_val_proto = proto.add_outputs();
    bytes_val_proto->set_name("bytes_val_proto");
    bytes_val_proto->set_datatype("BYTES");
    bytes_val_proto->mutable_shape()->Add(2);
    auto bytes_val = bytes_val_proto->mutable_contents()->mutable_bytes_contents()->Add();
    bytes_val->assign("string_1");
    bytes_val = bytes_val_proto->mutable_contents()->mutable_bytes_contents()->Add();
    bytes_val->assign("string_2");

    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, {"bytes_val_proto"}), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), true);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000]
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }, {
            "name": "bytes_val_proto",
            "shape": [2],
            "datatype": "BYTES",
            "parameters": {
                "binary_data_size": 24
            }
        }]
})";
    std::vector<std::uint8_t> expectedBinaryData = {8, 0, 0, 0, 's', 't', 'r', 'i', 'n', 'g', '_', '1', 8, 0, 0, 0, 's', 't', 'r', 'i', 'n', 'g', '_', '2'};
    ASSERT_EQ(inferenceHeaderContentLength.value(), expectedJson.size());
    ASSERT_EQ(json.size(), expectedJson.size() + expectedBinaryData.size());
    EXPECT_EQ(json.substr(0, inferenceHeaderContentLength.value()), expectedJson);
    for (uint8_t i = 0; i < expectedBinaryData.size(); i++) {
        EXPECT_EQ((uint8_t)json.substr(inferenceHeaderContentLength.value())[i], expectedBinaryData[i]);
    }
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_Positive) {
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000]
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_Positive_oneOutputsBinary) {
    std::set<std::string> binaryOutputs;
    binaryOutputs.insert("single_uint64_val");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, binaryOutputs), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), true);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "parameters": {
                "binary_data_size": 8
            }
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})";
    uint64_t expectedData = 5000000000;
    assertBinaryOutput(expectedData, json, expectedJson, inferenceHeaderContentLength);
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_Positive_bothOutputsBinary) {
    std::set<std::string> binaryOutputs;
    binaryOutputs.insert("single_uint64_val");
    binaryOutputs.insert("two_uint32_vals");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, binaryOutputs), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), true);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "parameters": {
                "binary_data_size": 8
            }
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "parameters": {
                "binary_data_size": 8
            }
        }]
})";
    ASSERT_EQ(inferenceHeaderContentLength.value(), expectedJson.size());
    ASSERT_EQ(json.size(), expectedJson.size() + sizeof(uint64_t) + 2 * sizeof(uint32_t));
    EXPECT_EQ(json.substr(0, inferenceHeaderContentLength.value()), expectedJson);
    uint64_t firstOutputExpectedData = 5000000000;
    EXPECT_EQ(*(uint64_t*)json.substr(inferenceHeaderContentLength.value()).data(), firstOutputExpectedData);
    uint32_t secondOutputExpectedData_1 = 4000000000;
    uint32_t secondOutputExpectedData_2 = 1;
    EXPECT_EQ(*(uint32_t*)json.substr(inferenceHeaderContentLength.value() + sizeof(uint64_t)).data(), secondOutputExpectedData_1);
    EXPECT_EQ(*(uint32_t*)json.substr(inferenceHeaderContentLength.value() + sizeof(uint64_t) + sizeof(uint32_t)).data(), secondOutputExpectedData_2);
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_OptionalModelVersion) {
    proto.set_model_version("version");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "model_version": "version",
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000]
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_OptionalStringParameter) {
    auto protoParameters = proto.mutable_parameters();
    (*protoParameters)["key"].set_string_param("param");
    auto outputParameters = single_uint64_val->mutable_parameters();
    (*outputParameters)["key"].set_string_param("param");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "parameters": {
        "key": "param"
    },
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000],
            "parameters": {
                "key": "param"
            }
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_OptionalIntParameter) {
    auto protoParameters = proto.mutable_parameters();
    (*protoParameters)["key"].set_int64_param(100);
    auto outputParameters = single_uint64_val->mutable_parameters();
    (*outputParameters)["key"].set_int64_param(100);
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);
    ASSERT_EQ(inferenceHeaderContentLength.has_value(), false);
    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "parameters": {
        "key": 100
    },
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000],
            "parameters": {
                "key": 100
            }
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})");
}

TEST_F(KFSMakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_OptionalBoolParameter) {
    auto protoParameters = proto.mutable_parameters();
    (*protoParameters)["key"].set_bool_param(true);
    auto outputParameters = single_uint64_val->mutable_parameters();
    (*outputParameters)["key"].set_bool_param(true);
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength), StatusCode::OK);

    EXPECT_EQ(json, R"({
    "model_name": "model",
    "id": "id",
    "parameters": {
        "key": true
    },
    "outputs": [{
            "name": "single_uint64_val",
            "shape": [1],
            "datatype": "UINT64",
            "data": [5000000000],
            "parameters": {
                "key": true
            }
        }, {
            "name": "two_uint32_vals",
            "shape": [2],
            "datatype": "UINT32",
            "data": [4000000000, 1]
        }]
})");
}

class KFSMakeJsonFromPredictResponseStringTest : public ::testing::Test {
protected:
    KFSResponse proto;
    std::string json;
    KFSTensorOutputProto *string_output_1, *string_output_2;
    std::optional<int> inferenceHeaderContentLength;

    void SetUp() override {
        proto.set_model_name("model");
        proto.set_id("id");

        string_output_1 = proto.add_outputs();
        string_output_1->set_name("string_output_1");
        string_output_1->set_datatype("BYTES");
        string_output_1->mutable_shape()->Add(2);
        string_output_1->mutable_contents()->mutable_bytes_contents()->Add()->assign("hello world");
        string_output_1->mutable_contents()->mutable_bytes_contents()->Add()->assign("welcome to kfs");

        string_output_2 = proto.add_outputs();
        string_output_2->set_name("string_output_2_string");
        string_output_2->set_datatype("BYTES");
        string_output_2->mutable_shape()->Add(2);
        string_output_2->mutable_contents()->mutable_bytes_contents()->Add()->assign("my 1 string");
        string_output_2->mutable_contents()->mutable_bytes_contents()->Add()->assign("my second string");
    }
};

TEST_F(KFSMakeJsonFromPredictResponseStringTest, Positive) {
    auto status = makeJsonFromPredictResponse(proto, &json, inferenceHeaderContentLength, {"string_output_1"});
    ASSERT_EQ(status, StatusCode::OK) << status.string();

    ASSERT_EQ(inferenceHeaderContentLength.has_value(), true);
    std::string expectedJson = R"({
    "model_name": "model",
    "id": "id",
    "outputs": [{
            "name": "string_output_1",
            "shape": [2],
            "datatype": "BYTES",
            "parameters": {
                "binary_data_size": 33
            }
        }, {
            "name": "string_output_2_string",
            "shape": [2],
            "datatype": "BYTES",
            "data": ["my 1 string", "my second string"]
        }]
})";
    ASSERT_EQ(inferenceHeaderContentLength.value(), expectedJson.size());
    ASSERT_EQ(json.size(), expectedJson.size() + 33);
    EXPECT_EQ(json.substr(0, inferenceHeaderContentLength.value()), expectedJson);
    std::vector<uint8_t> binaryData = {
        11, 0, 0, 0, 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd',
        14, 0, 0, 0, 'w', 'e', 'l', 'c', 'o', 'm', 'e', ' ', 't', 'o', ' ', 'k', 'f', 's'};
    EXPECT_EQ(std::memcmp(json.substr(inferenceHeaderContentLength.value()).data(), binaryData.data(), 33), 0);
}
