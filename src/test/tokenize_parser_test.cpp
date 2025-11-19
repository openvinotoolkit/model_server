//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "../tokenize/tokenize_parser.hpp"
#include "rapidjson/document.h"

TEST(TokenizeDeserialization, positiveTokenize) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_EQ(status, absl::OkStatus());
    auto strings = std::get_if<std::vector<std::string>>(&request.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(TokenizeDeserialization, invalidTextFieldMissing) {
    std::string requestBody = R"(
        {
            "model": "embeddings"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "text field is required");
}

TEST(TokenizeDeserialization, invalidTextFieldType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": 42
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "text should be string, array of strings or array of integers");
}

TEST(TokenizeDeserialization, invalidTextFieldEmptyArray) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": []
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "text array should not be empty");
}

TEST(TokenizeDeserialization, invalidTextFieldMalformedArray) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", 2, "three"]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "text must be homogeneous");
}

TEST(TokenizeDeserialization, positiveTokenizeParamsParse) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "max_length": 100,
            "pad_to_max_length": true,
            "padding_side": "right",
            "add_special_tokens": false
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_EQ(status, absl::OkStatus());
    auto strings = std::get_if<std::vector<std::string>>(&request.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
    auto params = request.parameters;
    ASSERT_EQ(params["max_length"].as<size_t>(), 100);
    ASSERT_EQ(params["pad_to_max_length"].as<bool>(), true);
    ASSERT_EQ(params["padding_side"].as<std::string>(), "right");
    ASSERT_EQ(params["add_special_tokens"].as<bool>(), false);
}

TEST(TokenizeDeserialization, invalidTokenizeMaxLengthType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "max_length": "string"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "max_length should be integer");
}

TEST(TokenizeDeserialization, invalidTokenizePadToMaxLengthType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "pad_to_max_length": "string"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "pad_to_max_length should be boolean");
}

TEST(TokenizeDeserialization, invalidTokenizeAddSpecialTokensType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "add_special_tokens": "string"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "add_special_tokens should be boolean");
}

TEST(TokenizeDeserialization, invalidTokenizePaddingSideType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "padding_side": 42
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "padding_side should be string, either left or right");
}

TEST(TokenizeDeserialization, invalidTokenizePaddingSideValue) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "text": ["one", "two", "three"],
            "padding_side": "invalid_value"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ovms::TokenizeRequest request;
    ASSERT_EQ(ok.Code(), 0);
    auto status = ovms::TokenizeParser::parseTokenizeRequest(d, request);
    ASSERT_NE(status, absl::OkStatus());
    auto error = status.message();
    ASSERT_EQ(error, "padding_side should be either left or right");
}
