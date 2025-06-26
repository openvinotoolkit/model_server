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

#include <string>
#include "../../../llm/response_parsers/partial_json_parser.hpp"

#include <gtest/gtest.h>

class PartialJsonParserTest : public ::testing::Test {};

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"OpenVINO\"}";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"Open";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("Open"));
}

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5}";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithArrayValue) {
    std::string input = "{\"numbers\": [1, 2, 3]}";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    ASSERT_EQ(parsedJson["numbers"].Size(), 3);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithArrayValue) {
    std::string input = "{\"numbers\": [1, 2, 3";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    // Last number might not be completed, so we do not include it and close the array earlier
    ASSERT_EQ(parsedJson["numbers"].Size(), 2);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithArrayValueMultipleNesting) {
    std::string input = "{\"numbers\": [[[1,2,3], [4,5,6";
    // If below method returns it means the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    ASSERT_TRUE(parsedJson["numbers"][0].IsArray());
    // The first inner array ([1,2,3]) is complete, the second ([4,5,6) is incomplete, so we expect two elements
    ASSERT_EQ(parsedJson["numbers"][0].Size(), 2);

    // First element: [1,2,3]
    ASSERT_TRUE(parsedJson["numbers"][0][0].IsArray());
    ASSERT_EQ(parsedJson["numbers"][0][0].Size(), 3);
    ASSERT_EQ(parsedJson["numbers"][0][0][0].GetInt(), 1);
    ASSERT_EQ(parsedJson["numbers"][0][0][1].GetInt(), 2);
    ASSERT_EQ(parsedJson["numbers"][0][0][2].GetInt(), 3);

    // Second element: [4,5]
    ASSERT_TRUE(parsedJson["numbers"][0][1].IsArray());
    ASSERT_EQ(parsedJson["numbers"][0][1].Size(), 2);
    ASSERT_EQ(parsedJson["numbers"][0][1][0].GetInt(), 4);
    ASSERT_EQ(parsedJson["numbers"][0][1][1].GetInt(), 5);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithStringValueWithExtraCharacters) {
    std::string input = "{\"arguments\": \"{\\\"location\\\": \\\"Tokyo, ";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo, "));
}

TEST_F(PartialJsonParserTest, simpleJsonWithKeyWithoutValue) {
    std::string input = "{\"name\": \"OpenVINO\", \"age\": ";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    // The "age" key is incomplete, so it should not be present in the parsed JSON
    ASSERT_FALSE(parsedJson.HasMember("age"));
}

TEST_F(PartialJsonParserTest, simpleJsonWithIncompleteKey) {
    std::string input = "{\"name\": \"OpenVINO\", \"ag";
    // If below method returns it means it the parsedJson is valid JSON, otherwise it would throw an exception
    auto parsedJson = partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    // The "age" key is incomplete, so it should not be present in the parsed JSON
    ASSERT_FALSE(parsedJson.HasMember("ag"));
}