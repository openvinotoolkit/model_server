//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rest_parser.hpp"
#include "absl/strings/escaping.h"
#include "test_utils.hpp"

using namespace ovms;

using ::testing::ElementsAre;

class TFSRestParserBinaryInputs : public ::testing::Test {
protected:
    void SetUp() override {
        readRgbJpg(filesize, image_bytes);
        std::string_view bytes(image_bytes.get(), filesize);

        absl::Base64Escape(bytes, &b64encoded);
    }

    size_t filesize;
    std::string b64encoded;
    std::unique_ptr<char[]> image_bytes;
};

TEST_F(TFSRestParserBinaryInputs, ColumnName) {
    std::string request = R"({"signature_name":"","inputs":{"k":[{"b64":")" + b64encoded + R"("}]}})";

    TFSRestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(TFSRestParserBinaryInputs, BatchSize2) {
    std::string request = R"({"signature_name":"","instances":[{"k":[{"b64":")" + b64encoded + R"("}]},{"i":[{"b64":")" + b64encoded + R"("}]}]})";

    TFSRestParser parser(TFSRestParser(prepareTensors({{"i", {1, 1}}, {"k", {1, 1}}})));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 2);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowStringMixedPrecision) {
    std::string request = R"({"signature_name":"","instances":[{"i": "abcd"}, {"i": 1234}]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST_F(TFSRestParserBinaryInputs, ColumnStringMixedPrecision) {
    std::string request = R"({"signature_name":"","inputs":{"i":["abcd", "efg", 52.1, "xyz"]}})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(TFSRestParserBinaryInputs, ColumnStringMixedPrecision2) {
    std::string request = R"({"signature_name":"","inputs":{"i":[[2,3,4],[5,"abcd",7]]}})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(TFSRestParserBinaryInputs, RowString) {
    std::string request = R"({"signature_name":"","instances":[{"i":"abcd"}]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowStringInvalidPrecision) {
    std::string request = R"({"signature_name":"","instances":[{"i":"abcd"}]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::FP32));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
}

TEST_F(TFSRestParserBinaryInputs, RowStringInvalidShape) {
    std::string request = R"({"signature_name":"","instances":[{"i":"abcd"}]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowStringStaticShape) {
    std::string request = R"({"signature_name":"","instances":[{"i":"abcd"}]})";

    TFSRestParser parser(prepareTensors({{"i", {1, 4}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, ColumnString) {
    std::string request = R"({"signature_name":"","inputs":{"i":["abcd"]}})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, ColumnStringUnnamed) {
    std::string request = R"({"signature_name":"","inputs":["abcd"]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowStringUnnamed) {
    std::string request = R"({"signature_name":"","instances":["abcd"]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(1));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowStringBatchSize2) {
    std::string request = R"({"signature_name":"","instances":[{"i":"abcd"}, {"i":"efgh"}]})";

    TFSRestParser parser(prepareTensors({{"i", {-1, -1}}}, ovms ::Precision::U8));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 2);
    EXPECT_THAT(asVector(parser.getProto().inputs().at("i").tensor_shape()), ElementsAre(2));
    ASSERT_EQ(parser.getProto().inputs().at("i").dtype(), tensorflow::DataType::DT_STRING);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), "abcd"), 0);
    EXPECT_EQ(strcmp(parser.getProto().inputs().find("i")->second.string_val(1).c_str(), "efgh"), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowName) {
    std::string request = R"({"signature_name":"","instances":[{"k":[{"b64":")" + b64encoded + R"("}]}]})";

    TFSRestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(TFSRestParserBinaryInputs, InvalidObject) {
    std::string request = R"({"signature_name":"","inputs":{"k":[{"b64":")" + b64encoded + R"(", "AdditionalField":"someValue"}]}})";

    TFSRestParser parser(prepareTensors({}, ovms::Precision::FP16));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(TFSRestParserBinaryInputs, ColumnNoNamed) {
    std::string request = R"({"signature_name":"","inputs":[{"b64":")" + b64encoded + R"("}]})";

    TFSRestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(TFSRestParserBinaryInputs, RowNoNamed) {
    std::string request = R"({"signature_name":"","instances":[[{"b64":")" + b64encoded + R"("}]]})";

    TFSRestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}
