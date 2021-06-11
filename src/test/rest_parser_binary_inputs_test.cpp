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

class RestParserBinaryInputs : public ::testing::Test {
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

TEST_F(RestParserBinaryInputs, ColumnName) {
    std::string request = R"({"signature_name":"","inputs":{"k":[{"b64":")" + b64encoded + R"("}]}})";

    RestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(RestParserBinaryInputs, BatchSize2) {
    std::string request = R"({"signature_name":"","instances":[{"k":[{"b64":")" + b64encoded + R"("}]},{"i":[{"b64":")" + b64encoded + R"("}]}]})";

    RestParser parser(RestParser(prepareTensors({{"i", {1, 1}}, {"k", {1, 1}}})));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 2);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().count("i"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().find("i")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("i")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(RestParserBinaryInputs, RowName) {
    std::string request = R"({"signature_name":"","instances":[{"k":[{"b64":")" + b64encoded + R"("}]}]})";

    RestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(RestParserBinaryInputs, InvalidObject) {
    std::string request = R"({"signature_name":"","inputs":{"k":[{"b64":")" + b64encoded + R"(", "AdditionalField":"someValue"}]}})";

    RestParser parser(prepareTensors({}, InferenceEngine::Precision::FP16));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(RestParserBinaryInputs, ColumnNoNamed) {
    std::string request = R"({"signature_name":"","inputs":[{"b64":")" + b64encoded + R"("}]})";

    RestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}

TEST_F(RestParserBinaryInputs, RowNoNamed) {
    std::string request = R"({"signature_name":"","instances":[[{"b64":")" + b64encoded + R"("}]]})";

    RestParser parser(prepareTensors({{"k", {1, 1}}}));
    ASSERT_EQ(parser.parse(request.c_str()), StatusCode::OK);
    ASSERT_EQ(parser.getProto().inputs_size(), 1);
    ASSERT_EQ(parser.getProto().inputs().count("k"), 1);
    ASSERT_EQ(parser.getProto().inputs().find("k")->second.string_val_size(), 1);
    EXPECT_EQ(std::memcmp(parser.getProto().inputs().find("k")->second.string_val(0).c_str(), image_bytes.get(), filesize), 0);
}
