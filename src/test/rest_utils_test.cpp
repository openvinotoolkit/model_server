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

#include "test_utils.hpp"

#include "../rest_utils.hpp"
#include "../logging.hpp"

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

class MakeJsonFromPredictResponseRawTest  : public ::testing::TestWithParam<ovms::Order> {
protected:
    tensorflow::serving::PredictResponse proto;
    std::string json;
    tensorflow::TensorProto *output1, *output2;

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

TEST_F(MakeJsonFromPredictResponseRawTest, CannotConvertUnknownOrder) {
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::UNKNOWN), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
}

TEST_F(MakeJsonFromPredictResponseRawTest, CannotConvertInvalidPrecision) {
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

std::string getJsonResponseInFirstOrder(ovms::Order order, const char* firstOrderRow, const char* firstOrderColumn) {
    switch(order){
        case Order::ROW:
            return firstOrderRow;
        case Order::COLUMN:
            return firstOrderColumn;
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

std::string getJsonResponseInSecondOrder(ovms::Order order, const char* secondOrderRow, const char* secondOrderColumn) {
    switch(order){
        case Order::ROW:
            return secondOrderRow;
        case Order::COLUMN:
            return secondOrderColumn;
        default:
            return "";
    }
}

TEST_P(MakeJsonFromPredictResponseRawTest, PositiveNamed) {
    auto order = GetParam();
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    bool is_in_first_order = (json == getJsonResponseInFirstOrder(order, rawPositiveFirstOrderResponseRow, rawPositiveFirstOrderResponseColumn));

    bool is_in_second_order = (json == getJsonResponseInSecondOrder(order, rawPositiveSecondOrderResponseRow, rawPositiveSecondOrderResponseColumn));

    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}


std::string getJsonResponseDependsOnOrder(ovms::Order order, const char* rowOrderResponse, const char* columnOrderResponse) {
    switch(order){
        case Order::ROW:
            return rowOrderResponse;
        case Order::COLUMN:
            return  columnOrderResponse;
        default:
            return "";
    }
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

TEST_P(MakeJsonFromPredictResponseRawTest, Positive_Noname) {
    auto order = GetParam();
    proto.mutable_outputs()->erase("output2");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    SPDLOG_ERROR(json);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, rawPositiveNonameResponseRow, rawPositiveNonameResponseColumn));
}

std::vector<ovms::Order> SupportedOrders = {Order::ROW, Order::COLUMN};

std::string toString(ovms::Order order) {
    switch(order){
        case Order::ROW:
            return "ROW";
        case Order::COLUMN:
            return  "COLUMN";
        default:
            return "";
    }
}

TEST_P(MakeJsonFromPredictResponseRawTest, EmptyTensorContentError) {
    auto order = GetParam();
    output1->mutable_tensor_content()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_SERIALIZE_NO_DATA);
}

TEST_P(MakeJsonFromPredictResponseRawTest, InvalidTensorContentSizeError) {
    auto order = GetParam();
    output1->mutable_tensor_content()->assign("\xFF\xFF\x55\x55", 4);
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE);
}

TEST_P(MakeJsonFromPredictResponseRawTest, ErrorWhenNoOutputs) {
    auto order = GetParam();
    proto.mutable_outputs()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::REST_PROTO_TO_STRING_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    TestGrpcRestResponseConversion,
    MakeJsonFromPredictResponseRawTest,
    ::testing::ValuesIn(SupportedOrders),
    [](const ::testing::TestParamInfo<MakeJsonFromPredictResponseRawTest::ParamType>& info) {
        return toString(info.param);
    });

class MakeJsonFromPredictResponsePrecisionTest : public ::testing::TestWithParam<ovms::Order> {
protected:
    tensorflow::serving::PredictResponse proto;
    std::string json;
    tensorflow::TensorProto* output;

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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Float) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Double) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Int32) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Int16) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Int8) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Uint8) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Int64) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Uint32) {
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

TEST_P(MakeJsonFromPredictResponsePrecisionTest, Uint64) {
    auto order = GetParam();
    uint64_t data = 63456412;
    output->set_dtype(tensorflow::DataType::DT_UINT64);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint64_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, order), StatusCode::OK);
    EXPECT_EQ(json, getJsonResponseDependsOnOrder(order, uint64ResponseRow, uint64ResponseColumn));
}

INSTANTIATE_TEST_SUITE_P(
    TestGrpcRestResponseConversion,
    MakeJsonFromPredictResponsePrecisionTest,
    ::testing::ValuesIn(SupportedOrders),
    [](const ::testing::TestParamInfo<MakeJsonFromPredictResponseRawTest::ParamType>& info) {
        return toString(info.param);
    });

class MakeJsonFromPredictResponseValTest : public ::testing::TestWithParam<ovms::Order> {
protected:
    tensorflow::serving::PredictResponse proto;
    std::string json;
    tensorflow::TensorProto *tensor_content_output, *single_uint64_val, *two_uint32_vals;

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


TEST_F(MakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainSingleUint64Val) {
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

TEST_F(MakeJsonFromPredictResponseValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainTwoUint32Vals) {
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
