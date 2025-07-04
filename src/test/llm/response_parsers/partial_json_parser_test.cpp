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

using namespace ovms;

class PartialJsonParserTest : public ::testing::Test {};

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"OpenVINO\"}";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
}

TEST_F(PartialJsonParserTest, complexCompleteJsonWithDifferentValueTypes) {
    std::string input = R"({
        "user": {
            "name": "OpenVINO",
            "details": {
                "age": 5,
                "skills": ["C++", "Python", "AI"]
            }
        },
        "numbers": [1, 2, 3]
    })";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("user"));
    ASSERT_TRUE(parsedJson["user"].IsObject());
    ASSERT_TRUE(parsedJson["user"].HasMember("name"));
    ASSERT_TRUE(parsedJson["user"]["name"].IsString());
    ASSERT_EQ(parsedJson["user"]["name"].GetString(), std::string("OpenVINO"));
    ASSERT_TRUE(parsedJson["user"].HasMember("details"));
    ASSERT_TRUE(parsedJson["user"]["details"].IsObject());
    ASSERT_TRUE(parsedJson["user"]["details"].HasMember("age"));
    ASSERT_TRUE(parsedJson["user"]["details"]["age"].IsInt());
    ASSERT_EQ(parsedJson["user"]["details"]["age"].GetInt(), 5);
    ASSERT_TRUE(parsedJson["user"]["details"].HasMember("skills"));
    ASSERT_TRUE(parsedJson["user"]["details"]["skills"].IsArray());
    ASSERT_EQ(parsedJson["user"]["details"]["skills"].Size(), 3);
    ASSERT_EQ(parsedJson["user"]["details"]["skills"][0].GetString(), std::string("C++"));
    ASSERT_EQ(parsedJson["user"]["details"]["skills"][1].GetString(), std::string("Python"));
    ASSERT_EQ(parsedJson["user"]["details"]["skills"][2].GetString(), std::string("AI"));
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    ASSERT_EQ(parsedJson["numbers"].Size(), 3);
    ASSERT_EQ(parsedJson["numbers"][0].GetInt(), 1);
    ASSERT_EQ(parsedJson["numbers"][1].GetInt(), 2);
    ASSERT_EQ(parsedJson["numbers"][2].GetInt(), 3);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"Open";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("Open"));
}

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5}";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithNumberValueTwoKeys) {
    std::string input = "{\"age\": 5, \"height\": 180";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson.HasMember("height"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
    ASSERT_TRUE(parsedJson["height"].IsInt());
    ASSERT_EQ(parsedJson["height"].GetInt(), 180);
}

TEST_F(PartialJsonParserTest, simpleCompleteJsonWithArrayValue) {
    std::string input = "{\"numbers\": [1, 2, 3]}";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    ASSERT_EQ(parsedJson["numbers"].Size(), 3);
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithArrayValue) {
    auto inputs = {"{\"numbers\": [1, 2, 3",
        "{\"numbers\": [1, 2, 3, "};

    for (const auto& input : inputs) {
        JsonBuilder builder;
        auto parsedJson = builder.partialParseToJson(input);
        ASSERT_TRUE(parsedJson.IsObject());
        ASSERT_TRUE(parsedJson.HasMember("numbers"));
        ASSERT_TRUE(parsedJson["numbers"].IsArray());
        ASSERT_EQ(parsedJson["numbers"].Size(), 3);
    }
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithArrayValueMultipleNesting) {
    auto inputs = {"{\"numbers\": [[[1,2,3], [4,5,6",
        "{\"numbers\": [[[1,2,3], [4,5,6,"};

    for (const auto& input : inputs) {
        JsonBuilder builder;
        auto parsedJson = builder.partialParseToJson(input);
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

        // Second element: [4,5,6]
        ASSERT_TRUE(parsedJson["numbers"][0][1].IsArray());
        ASSERT_EQ(parsedJson["numbers"][0][1].Size(), 3);
        ASSERT_EQ(parsedJson["numbers"][0][1][0].GetInt(), 4);
        ASSERT_EQ(parsedJson["numbers"][0][1][1].GetInt(), 5);
        ASSERT_EQ(parsedJson["numbers"][0][1][2].GetInt(), 6);
    }
}

TEST_F(PartialJsonParserTest, simpleUncompleteJsonWithStringValueWithExtraCharacters) {
    std::string input = "{\"arguments\": \"{\\\"location\\\": \\\"Tokyo, ";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo, "));
}

TEST_F(PartialJsonParserTest, simpleJsonWithKeyWithoutValue) {
    std::string input = "{\"name\": \"OpenVINO\", \"age\": ";
    JsonBuilder builder;
    auto parsedJson = builder.partialParseToJson(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    // The "age" key is incomplete, so it should not be present in the parsed JSON
    ASSERT_FALSE(parsedJson.HasMember("age"));
}

TEST_F(PartialJsonParserTest, simpleJsonWithIncompleteKey) {
    auto inputs = {"{\"name\": \"OpenVINO\", \"ag",
        "{\"name\": \"OpenVINO\",",
        "{\"name\": \"OpenVINO\""};
    for (const auto& input : inputs) {
        JsonBuilder builder;
        auto parsedJson = builder.partialParseToJson(input);
        ASSERT_TRUE(parsedJson.IsObject());
        EXPECT_EQ(parsedJson.MemberCount(), 1);
        ASSERT_TRUE(parsedJson.HasMember("name"));
        ASSERT_TRUE(parsedJson["name"].IsString());
        ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    }
}

TEST_F(PartialJsonParserTest, complexJsonWithIncompleteKey) {
    // Nested object of objects with incomplete key
    auto inputs = {"{\"tool\": {\"name\": \"OpenVINO\", \"ag",
        "{\"tool\": {\"name\": \"OpenVINO\",",
        "{\"tool\": {\"name\": \"OpenVINO\""};
    for (const auto& input : inputs) {
        JsonBuilder builder;
        auto parsedJson = builder.partialParseToJson(input);
        ASSERT_TRUE(parsedJson.IsObject());
        EXPECT_EQ(parsedJson.MemberCount(), 1);
        ASSERT_TRUE(parsedJson.HasMember("tool"));
        ASSERT_TRUE(parsedJson["tool"].IsObject());
        EXPECT_EQ(parsedJson["tool"].MemberCount(), 1);
        ASSERT_TRUE(parsedJson["tool"].HasMember("name"));
        ASSERT_TRUE(parsedJson["tool"]["name"].IsString());
        ASSERT_EQ(parsedJson["tool"]["name"].GetString(), std::string("OpenVINO"));
    }

    // Nested array of objects with incomplete key
    auto inputsArray = {"{\"tools\": [{\"name\": \"OpenVINO\"}, {\"ag",
        "{\"tools\": [{\"name\": \"OpenVINO\"},",
        "{\"tools\": [{\"name\": \"OpenVINO\"}"};

    for (const auto& input : inputsArray) {
        JsonBuilder builder;
        auto parsedJson = builder.partialParseToJson(input);
        ASSERT_TRUE(parsedJson.IsObject());
        EXPECT_EQ(parsedJson.MemberCount(), 1);
        ASSERT_TRUE(parsedJson.HasMember("tools"));
        ASSERT_TRUE(parsedJson["tools"].IsArray());
        ASSERT_EQ(parsedJson["tools"].Size(), 1);  // One object in the array
        ASSERT_TRUE(parsedJson["tools"][0].IsObject());
        EXPECT_EQ(parsedJson["tools"][0].MemberCount(), 1);
        ASSERT_TRUE(parsedJson["tools"][0].HasMember("name"));
        ASSERT_TRUE(parsedJson["tools"][0]["name"].IsString());
        ASSERT_EQ(parsedJson["tools"][0]["name"].GetString(), std::string("OpenVINO"));
    }
}

TEST_F(PartialJsonParserTest, complexJsonIncrementalParsingSanityCheck) {
    std::string targetJson = R"({
        "major_object": {
            "string": "OpenVINO",
            "minor_object": {
                "number": 5,
                "number_array": [1, 2, 3],
                "string_array": ["C++", "Python", "\"Java\"", "AI"]
            }
        },
        "boolean": true,
        "boolean_array": [true, false, true],
        "null_value": null,
        "null_array": [null, null, null],
        "empty_object": {}
    })";
    JsonBuilder builder;
    rapidjson::Document parsedJson;
    for (size_t i = 0; i < targetJson.size(); ++i) {
        std::string partialInput(1, targetJson[i]);
        parsedJson = builder.partialParseToJson(partialInput);
    }

    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("major_object"));
    ASSERT_TRUE(parsedJson["major_object"].IsObject());
    ASSERT_TRUE(parsedJson["major_object"].HasMember("string"));
    ASSERT_TRUE(parsedJson["major_object"]["string"].IsString());
    ASSERT_EQ(parsedJson["major_object"]["string"].GetString(), std::string("OpenVINO"));

    ASSERT_TRUE(parsedJson["major_object"].HasMember("minor_object"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].IsObject());
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].HasMember("number"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"]["number"].IsInt());
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["number"].GetInt(), 5);

    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].HasMember("number_array"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"]["number_array"].IsArray());
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["number_array"].Size(), 3);
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["number_array"][0].GetInt(), 1);
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["number_array"][1].GetInt(), 2);
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["number_array"][2].GetInt(), 3);

    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].HasMember("string_array"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"]["string_array"].IsArray());
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["string_array"].Size(), 4);
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["string_array"][0].GetString(), std::string("C++"));
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["string_array"][1].GetString(), std::string("Python"));
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["string_array"][2].GetString(), std::string("\"Java\""));
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["string_array"][3].GetString(), std::string("AI"));

    ASSERT_TRUE(parsedJson.HasMember("boolean"));
    ASSERT_TRUE(parsedJson["boolean"].IsBool());
    ASSERT_EQ(parsedJson["boolean"].GetBool(), true);

    ASSERT_TRUE(parsedJson.HasMember("boolean_array"));
    ASSERT_TRUE(parsedJson["boolean_array"].IsArray());
    ASSERT_EQ(parsedJson["boolean_array"].Size(), 3);
    ASSERT_TRUE(parsedJson["boolean_array"][0].IsBool());
    ASSERT_TRUE(parsedJson["boolean_array"][1].IsBool());
    ASSERT_TRUE(parsedJson["boolean_array"][2].IsBool());
    ASSERT_EQ(parsedJson["boolean_array"][0].GetBool(), true);
    ASSERT_EQ(parsedJson["boolean_array"][1].GetBool(), false);
    ASSERT_EQ(parsedJson["boolean_array"][2].GetBool(), true);

    ASSERT_TRUE(parsedJson.HasMember("null_value"));
    ASSERT_TRUE(parsedJson["null_value"].IsNull());

    ASSERT_TRUE(parsedJson.HasMember("null_array"));
    ASSERT_TRUE(parsedJson["null_array"].IsArray());
    ASSERT_EQ(parsedJson["null_array"].Size(), 3);
    ASSERT_TRUE(parsedJson["null_array"][0].IsNull());
    ASSERT_TRUE(parsedJson["null_array"][1].IsNull());
    ASSERT_TRUE(parsedJson["null_array"][2].IsNull());

    ASSERT_TRUE(parsedJson.HasMember("empty_object"));
    ASSERT_TRUE(parsedJson["empty_object"].IsObject());
    ASSERT_EQ(parsedJson["empty_object"].MemberCount(), 0);
}

TEST_F(PartialJsonParserTest, simpleJsonIncrementalParsing) {
    std::string targetJson = R"({
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\", \"date\": \"2025-01-01\"}"
    })";
    JsonBuilder builder;
    rapidjson::Document parsedJson;
    builder.partialParseToJson("{");
    builder.partialParseToJson("\"");
    parsedJson = builder.partialParseToJson("name");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_EQ(parsedJson.MemberCount(), 0);  // Should not be complete yet

    builder.partialParseToJson("\": \"");
    builder.partialParseToJson("get");
    parsedJson = builder.partialParseToJson("_");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_"));

    builder.partialParseToJson("weather");
    builder.partialParseToJson("\", ");
    parsedJson = builder.partialParseToJson("\"arguments\":");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_weather"));
    ASSERT_EQ(parsedJson.MemberCount(), 1);  // Only "name" should be present

    builder.partialParseToJson("\"{");
    parsedJson = builder.partialParseToJson("\\\"location\\\": \\\"");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \""));

    builder.partialParseToJson("Tokyo");
    builder.partialParseToJson("\\\", \\\"");
    parsedJson = builder.partialParseToJson("date");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo\", \"date"));

    builder.partialParseToJson("\\\": \\\"");
    builder.partialParseToJson("2025-01-01");
    builder.partialParseToJson("\\\"}\"");
    parsedJson = builder.partialParseToJson("}");

    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_weather"));
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo\", \"date\": \"2025-01-01\"}"));
}
