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

#include "../rest_utils.hpp"

using namespace ovms;

class RestUtilsTest : public ::testing::Test {
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

class RestUtilsPrecisionTest : public ::testing::Test {
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

class RestUtilsValTest : public ::testing::Test {
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

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_CannotConvertUnknownOrder) {
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::UNKNOWN), StatusCode::REST_PREDICT_UNKNOWN_ORDER);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_CannotConvertInvalidPrecision) {
    output1->set_dtype(tensorflow::DataType::DT_INVALID);
    output1->mutable_tensor_content()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::REST_UNSUPPORTED_PRECISION);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_ColumnOrder) {
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);

    bool is_in_first_order = json == R"({
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

    bool is_in_second_order = json == R"({
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

    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_RowOrder) {
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);

    bool is_in_first_order = json == R"({
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

    bool is_in_second_order = json == R"({
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
    EXPECT_TRUE(is_in_first_order || is_in_second_order);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_ColumnOrder_Noname) {
    proto.mutable_outputs()->erase("output2");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
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
})");
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_RowOrder_Noname) {
    proto.mutable_outputs()->erase("output1");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[5, 2, 3, 8, -2], [-100, 0, 125, 4, -1]
    ]
})");
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_EmptyTensorContentError) {
    output1->mutable_tensor_content()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::REST_SERIALIZE_NO_DATA);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_InvalidTensorContentSizeError) {
    output1->mutable_tensor_content()->assign("\xFF\xFF\x55\x55", 4);
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE);
}

TEST_F(RestUtilsTest, MakeJsonFromPredictResponse_ErrorWhenNoOutputs) {
    proto.mutable_outputs()->clear();
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::REST_PROTO_TO_STRING_ERROR);
    EXPECT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::REST_PROTO_TO_STRING_ERROR);
}

TEST_F(RestUtilsTest, Base64DecodeCorrect) {
    std::string bytes = "abcd";
    std::string decodedBytes;
    EXPECT_EQ(decodeBase64(bytes, decodedBytes), StatusCode::OK);
    EXPECT_EQ(decodedBytes, "i\xB7\x1D");
}

TEST_F(RestUtilsTest, Base64DecodeWrongLength) {
    std::string bytes = "abcde";
    std::string decodedBytes;
    EXPECT_EQ(decodeBase64(bytes, decodedBytes), StatusCode::REST_BASE64_DECODE_ERROR);
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Float) {
    float data = 92.5f;
    output->set_dtype(tensorflow::DataType::DT_FLOAT);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(float));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[92.5]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            92.5
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Double) {
    double data = 15.99;
    output->set_dtype(tensorflow::DataType::DT_DOUBLE);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(double));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[15.99]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            15.99
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Int32) {
    int32_t data = -82;
    output->set_dtype(tensorflow::DataType::DT_INT32);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int32_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[-82]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            -82
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Int16) {
    int16_t data = -945;
    output->set_dtype(tensorflow::DataType::DT_INT16);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int16_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[-945]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            -945
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Int8) {
    int8_t data = -53;
    output->set_dtype(tensorflow::DataType::DT_INT8);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int8_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[-53]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            -53
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Uint8) {
    uint8_t data = 250;
    output->set_dtype(tensorflow::DataType::DT_UINT8);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint8_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[250]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            250
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Int64) {
    int64_t data = -658324;
    output->set_dtype(tensorflow::DataType::DT_INT64);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(int64_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[-658324]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            -658324
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Uint32) {
    uint32_t data = 1245353;
    output->set_dtype(tensorflow::DataType::DT_UINT32);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint32_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[1245353]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            1245353
        ]
    ]
})");
}

TEST_F(RestUtilsPrecisionTest, MakeJsonFromPredictResponse_Uint64) {
    uint64_t data = 63456412;
    output->set_dtype(tensorflow::DataType::DT_UINT64);
    output->mutable_tensor_content()->assign(reinterpret_cast<const char*>(&data), sizeof(uint64_t));
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::ROW), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "predictions": [[63456412]
    ]
})");
    ASSERT_EQ(makeJsonFromPredictResponse(proto, &json, Order::COLUMN), StatusCode::OK);
    EXPECT_EQ(json, R"({
    "outputs": [
        [
            63456412
        ]
    ]
})");
}

TEST_F(RestUtilsValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainSingleUint64Val) {
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

TEST_F(RestUtilsValTest, MakeJsonFromPredictResponse_ColumnOrder_ContainTwoUint32Vals) {
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
