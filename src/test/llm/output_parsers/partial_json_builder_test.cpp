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
#include "../../../llm/io_processing/partial_json_builder.hpp"

#include <gtest/gtest.h>

using namespace ovms;

class PartialJsonBuilderTest : public ::testing::Test {};

TEST_F(PartialJsonBuilderTest, simpleCompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"OpenVINO\"}";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
}

TEST_F(PartialJsonBuilderTest, complexCompleteJsonWithDifferentValueTypes) {
    std::string input = R"({
        "user": {
            "name": "OpenVINO",
            "details": {
                "age": 5,
                "skills": ["C++", "Python", "AI"]
            }
        },
        "numbers": [1, 2, 3],
        "complex_string": "This is a complex string with special characters: \n, \r, \t, \", \\ \""
    })";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
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
    ASSERT_TRUE(parsedJson.HasMember("complex_string"));
    ASSERT_TRUE(parsedJson["complex_string"].IsString());
    ASSERT_EQ(parsedJson["complex_string"].GetString(), std::string("This is a complex string with special characters: \n, \r, \t, \", \\ \""));
}

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithStringValue) {
    std::string input = "{\"name\": \"Open";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("Open"));
}

TEST_F(PartialJsonBuilderTest, simpleCompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5}";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithNumberValue) {
    std::string input = "{\"age\": 5";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
}

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithNumberValueTwoKeys) {
    std::string input = "{\"age\": 5, \"height\": 180";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson.HasMember("height"));
    ASSERT_TRUE(parsedJson["age"].IsInt());
    ASSERT_EQ(parsedJson["age"].GetInt(), 5);
    ASSERT_TRUE(parsedJson["height"].IsInt());
    ASSERT_EQ(parsedJson["height"].GetInt(), 180);
}

TEST_F(PartialJsonBuilderTest, simpleCompleteJsonWithArrayValue) {
    std::string input = "{\"numbers\": [1, 2, 3]}";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("numbers"));
    ASSERT_TRUE(parsedJson["numbers"].IsArray());
    ASSERT_EQ(parsedJson["numbers"].Size(), 3);
}

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithArrayValue) {
    auto inputs = {"{\"numbers\": [1, 2, 3",
        "{\"numbers\": [1, 2, 3, "};

    for (const auto& input : inputs) {
        PartialJsonBuilder builder;
        auto parsedJson = builder.add(input);
        ASSERT_TRUE(parsedJson.IsObject());
        ASSERT_TRUE(parsedJson.HasMember("numbers"));
        ASSERT_TRUE(parsedJson["numbers"].IsArray());
        ASSERT_EQ(parsedJson["numbers"].Size(), 3);
    }
}

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithArrayValueMultipleNesting) {
    auto inputs = {"{\"numbers\": [[[1,2,3], [4,5,6",
        "{\"numbers\": [[[1,2,3], [4,5,6,"};

    for (const auto& input : inputs) {
        PartialJsonBuilder builder;
        auto parsedJson = builder.add(input);
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

TEST_F(PartialJsonBuilderTest, simpleUncompleteJsonWithStringValueWithExtraCharacters) {
    std::string input = "{\"arguments\": \"{\\\"location\\\": \\\"Tokyo, ";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo, "));
}

TEST_F(PartialJsonBuilderTest, simpleJsonWithKeyWithoutValue) {
    std::string input = "{\"name\": \"OpenVINO\", \"age\": ";
    PartialJsonBuilder builder;
    auto parsedJson = builder.add(input);
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    // The "age" key exists but its value is null since it is incomplete
    ASSERT_TRUE(parsedJson.HasMember("age"));
    ASSERT_TRUE(parsedJson["age"].IsNull());
}

TEST_F(PartialJsonBuilderTest, simpleJsonWithIncompleteKey) {
    auto inputs = {"{\"name\": \"OpenVINO\", \"ag",
        "{\"name\": \"OpenVINO\",",
        "{\"name\": \"OpenVINO\""};
    for (const auto& input : inputs) {
        PartialJsonBuilder builder;
        auto parsedJson = builder.add(input);
        ASSERT_TRUE(parsedJson.IsObject());
        EXPECT_EQ(parsedJson.MemberCount(), 1);
        ASSERT_TRUE(parsedJson.HasMember("name"));
        ASSERT_TRUE(parsedJson["name"].IsString());
        ASSERT_EQ(parsedJson["name"].GetString(), std::string("OpenVINO"));
    }
}

TEST_F(PartialJsonBuilderTest, complexJsonWithIncompleteKey) {
    // Nested object of objects with incomplete key
    auto inputs = {"{\"tool\": {\"name\": \"OpenVINO\", \"ag",
        "{\"tool\": {\"name\": \"OpenVINO\",",
        "{\"tool\": {\"name\": \"OpenVINO\""};
    for (const auto& input : inputs) {
        PartialJsonBuilder builder;
        auto parsedJson = builder.add(input);
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
        PartialJsonBuilder builder;
        auto parsedJson = builder.add(input);
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

TEST_F(PartialJsonBuilderTest, complexJsonIncrementalParsingSanityCheck) {
    std::string targetJson = R"(
    
    {
        "major_object": {
            "string": "OpenVINO",
            "minor_object": {
                "number": 5,
                "number_array": [1, 2, 3],
                "float": 3.14,
                "float_array": [1.1, 2.2, 3.3],
                "string_array": ["C++", "Python", "\"Java\"", "AI"]
            }
        },
        "boolean": true,
        "boolean_array": [true, false, true],
        "null_value": null,
        "null_array": [null, null, null],
        "empty_object": {}
    })";
    PartialJsonBuilder builder;
    rapidjson::Document parsedJson;
    for (size_t i = 0; i < targetJson.size(); ++i) {
        std::string partialInput(1, targetJson[i]);
        parsedJson = builder.add(partialInput);
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

    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].HasMember("float"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"]["float"].IsDouble());
    ASSERT_DOUBLE_EQ(parsedJson["major_object"]["minor_object"]["float"].GetDouble(), 3.14);
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"].HasMember("float_array"));
    ASSERT_TRUE(parsedJson["major_object"]["minor_object"]["float_array"].IsArray());
    ASSERT_EQ(parsedJson["major_object"]["minor_object"]["float_array"].Size(), 3);
    ASSERT_DOUBLE_EQ(parsedJson["major_object"]["minor_object"]["float_array"][0].GetDouble(), 1.1);
    ASSERT_DOUBLE_EQ(parsedJson["major_object"]["minor_object"]["float_array"][1].GetDouble(), 2.2);
    ASSERT_DOUBLE_EQ(parsedJson["major_object"]["minor_object"]["float_array"][2].GetDouble(), 3.3);

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

TEST_F(PartialJsonBuilderTest, simpleJsonIncrementalParsing) {
    std::string targetJson = R"({
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\", \"date\": \"2025-01-01\"}"
    })";
    PartialJsonBuilder builder;
    rapidjson::Document parsedJson;
    builder.add("{");
    builder.add("\"");
    parsedJson = builder.add("name");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_EQ(parsedJson.MemberCount(), 0);  // Should not be complete yet

    builder.add("\": \"");
    builder.add("get");
    parsedJson = builder.add("_");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_"));

    builder.add("weather");
    builder.add("\", ");
    parsedJson = builder.add("\"arguments\":");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_weather"));
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsNull());

    builder.add("\"{");
    parsedJson = builder.add("\\\"location\\\": \\\"");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \""));

    builder.add("Tokyo");
    builder.add("\\\", \\\"");
    parsedJson = builder.add("date");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo\", \"date"));

    builder.add("\\\": \\\"");
    builder.add("2025-01-01");
    builder.add("\\\"}\"");
    parsedJson = builder.add("}");

    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_weather"));
    ASSERT_TRUE(parsedJson.HasMember("arguments"));
    ASSERT_TRUE(parsedJson["arguments"].IsString());
    ASSERT_EQ(parsedJson["arguments"].GetString(), std::string("{\"location\": \"Tokyo\", \"date\": \"2025-01-01\"}"));
}

TEST_F(PartialJsonBuilderTest, negativeCases) {
    std::vector<std::pair<std::string, std::string>> negativeCases = {
        {R"(a)", "Invalid JSON: Expected '{' or '[' at the beginning."},
        {R"({"name",)", "Invalid JSON: Expected ':' after key."},
        {R"({"object": {"string":"1", "string",)", "Invalid JSON: Expected ':' after key."},
        {R"({"name": "get_weather",  1)", "Invalid JSON: Expected key to start with a quote or a proper object closure."},
        {R"({"name": a)", "Invalid JSON: Expected value to start with '{', '[', '\"', digit, 't', 'f', or 'n'."},
        {R"({"numbers": []])", "Invalid JSON. Content:\n{\"numbers\": []]}"},                    // invalid closure
        {R"({"numbers": [1, 2, 3})", "Invalid JSON. Content:\n{\"numbers\": [1, 2, 3}]}"},       // invalid closure
        {R"({"numbers": [1, 2, 3b)", "Invalid JSON. Content:\n{\"numbers\": [1, 2, 3b]}"},       // invalid value
        {R"({"numbers": [1, 2, 3")", "Invalid JSON. Content:\n{\"numbers\": [1, 2, 3\"\"]}"},    // invalid value
        {R"({"string": "string\""1)", "Invalid JSON. Content:\n{\"string\": \"string\\\"\"1}"},  // invalid value
        {R"({"bool": tak,)", "Invalid JSON. Content:\n{\"bool\": tak}"},                         // invalid special value
    };

    for (const auto& [json, expectedError] : negativeCases) {
        PartialJsonBuilder builder;
        for (size_t i = 0; i < json.size(); ++i) {
            std::string s(1, json[i]);
            if (i == json.size() - 1) {
                try {
                    builder.add(s);
                    FAIL() << "Expected exception not thrown";
                } catch (const std::exception& ex) {
                    EXPECT_EQ(std::string(ex.what()), expectedError);
                }
            } else {
                builder.add(s);
            }
        }
    }
}

TEST_F(PartialJsonBuilderTest, postJsonEndAdditions) {
    PartialJsonBuilder builder;
    rapidjson::Document parsedJson;
    builder.add("{\"name\": \"get_weather\"");
    ASSERT_FALSE(builder.isComplete());
    parsedJson = builder.add("}, {");
    ASSERT_TRUE(builder.isComplete());
    ASSERT_EQ(builder.getUnprocessedBuffer(), ", {");
    ASSERT_TRUE(parsedJson.IsObject());
    ASSERT_TRUE(parsedJson.HasMember("name"));
    ASSERT_TRUE(parsedJson["name"].IsString());
    ASSERT_EQ(parsedJson["name"].GetString(), std::string("get_weather"));
}

TEST_F(PartialJsonBuilderTest, computeDeltaWithEmptyJson) {
    rapidjson::Document previous;
    previous.SetObject();
    rapidjson::Document current;
    current.SetObject();
    auto delta = PartialJsonBuilder::computeDelta(previous, current);
    ASSERT_TRUE(delta.IsObject());
    ASSERT_TRUE(delta.Empty());
}

TEST_F(PartialJsonBuilderTest, computeDeltaWithAddedMember) {
    const char* previousJson = R"({
        "name": "get_weather"
    })";
    rapidjson::Document previous;
    previous.Parse(previousJson);

    const char* currentJson = R"({
        "name": "get_weather",
        "arguments": "\""
    })";
    rapidjson::Document current;
    current.Parse(currentJson);

    auto delta = PartialJsonBuilder::computeDelta(previous, current);
    // Expecting delta {"arguments": "\""}
    ASSERT_TRUE(delta.IsObject());
    ASSERT_FALSE(delta.Empty());
    ASSERT_FALSE(delta.HasMember("name"));
    ASSERT_TRUE(delta.HasMember("arguments"));
    ASSERT_TRUE(delta["arguments"].IsString());
    ASSERT_EQ(delta["arguments"].GetString(), std::string("\""));
}

TEST_F(PartialJsonBuilderTest, computeDeltaWithAddedNestedMember) {
    const char* previousJson = R"({
        "name": "get_weather",
        "object": {
            "key": "value"
        }
    })";
    rapidjson::Document previous;
    previous.Parse(previousJson);

    const char* currentJson = R"({
        "name": "get_weather",
        "object": {
            "key": "value",
            "new_key": null
        } 
    })";
    rapidjson::Document current;
    current.Parse(currentJson);

    auto delta = PartialJsonBuilder::computeDelta(previous, current);
    // Expecting delta {"object": {"new_key": null}}
    ASSERT_TRUE(delta.IsObject());
    ASSERT_FALSE(delta.Empty());
    ASSERT_TRUE(delta.HasMember("object"));
    ASSERT_TRUE(delta["object"].IsObject());
    ASSERT_EQ(delta["object"].MemberCount(), 1);
    ASSERT_TRUE(delta["object"].HasMember("new_key"));
    ASSERT_TRUE(delta["object"]["new_key"].IsNull());
}

TEST_F(PartialJsonBuilderTest, computeDeltaWithAddedNestedArrayElement) {
    const char* previousJson = R"({
        "name": "get_weather",
        "objects": [
            {
                "key": "value1"
            }
        ]
    })";
    rapidjson::Document previous;
    previous.Parse(previousJson);

    const char* currentJson = R"({
        "name": "get_weather",
        "objects": [
            {
                "key": "value1"
            },
            {
                "key": "value2"
            }
        ]
    })";
    rapidjson::Document current;
    current.Parse(currentJson);

    auto delta = PartialJsonBuilder::computeDelta(previous, current);
    // Expecting delta {"objects": [{"key": "value2"}]}
    ASSERT_TRUE(delta.IsObject());
    ASSERT_FALSE(delta.Empty());
    ASSERT_TRUE(delta.HasMember("objects"));
    ASSERT_TRUE(delta["objects"].IsArray());
    ASSERT_EQ(delta["objects"].Size(), 1);
    ASSERT_TRUE(delta["objects"][0].IsObject());
    ASSERT_TRUE(delta["objects"][0].HasMember("key"));
    ASSERT_EQ(delta["objects"][0]["key"].GetString(), std::string("value2"));
}

TEST_F(PartialJsonBuilderTest, computeDeltaWithModifiedStringMember) {
    const char* previousJson = R"({
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\""
    })";
    rapidjson::Document previous;
    previous.Parse(previousJson);

    const char* currentJson = R"({
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\", \"date\":"
    })";
    rapidjson::Document current;
    current.Parse(currentJson);

    auto delta = PartialJsonBuilder::computeDelta(previous, current);
    // Expecting delta {"arguments": ", \"date\": "}
    ASSERT_TRUE(delta.IsObject());
    ASSERT_FALSE(delta.Empty());
    ASSERT_TRUE(delta.HasMember("arguments"));
    ASSERT_TRUE(delta["arguments"].IsString());
    // Only the new part should be present in arguments
    ASSERT_EQ(delta["arguments"].GetString(), std::string(", \"date\":"));
}
