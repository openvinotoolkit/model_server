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

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../../../llm/input_workarounds.hpp"

using namespace ovms;

class InputWorkaroundsTest : public ::testing::Test {
protected:
    rapidjson::Document parseJson(const std::string& json) {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        EXPECT_FALSE(doc.HasParseError()) << "Failed to parse test JSON";
        return doc;
    }

    std::string serializeJson(const rapidjson::Document& doc) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        return buffer.GetString();
    }
};

// --- funcArgsToObjectJson ---

TEST_F(InputWorkaroundsTest, funcArgsToObjectConvertsStringArgs) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"London\", \"units\": \"celsius\"}"
                }}
            ]}
        ]
    })");

    input_workarounds::funcArgsToObjectJson(doc);

    auto& args = doc["messages"][1]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.IsObject());
    ASSERT_TRUE(args.HasMember("city"));
    EXPECT_STREQ(args["city"].GetString(), "London");
    ASSERT_TRUE(args.HasMember("units"));
    EXPECT_STREQ(args["units"].GetString(), "celsius");
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectHandlesMultipleToolCalls) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "function": {"name": "fn1", "arguments": "{\"a\": 1}"}},
                {"id": "call_2", "function": {"name": "fn2", "arguments": "{\"b\": true}"}}
            ]}
        ]
    })");

    input_workarounds::funcArgsToObjectJson(doc);

    auto& args1 = doc["messages"][0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args1.IsObject());
    EXPECT_EQ(args1["a"].GetInt(), 1);

    auto& args2 = doc["messages"][0]["tool_calls"][1]["function"]["arguments"];
    ASSERT_TRUE(args2.IsObject());
    EXPECT_TRUE(args2["b"].GetBool());
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectSkipsAlreadyObjectArgs) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "fn", "arguments": {"key": "value"}}}
            ]}
        ]
    })");

    input_workarounds::funcArgsToObjectJson(doc);

    auto& args = doc["messages"][0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.IsObject());
    EXPECT_STREQ(args["key"].GetString(), "value");
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectSkipsInvalidJsonString) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "fn", "arguments": "not valid json {"}}
            ]}
        ]
    })");

    input_workarounds::funcArgsToObjectJson(doc);

    auto& args = doc["messages"][0]["tool_calls"][0]["function"]["arguments"];
    EXPECT_TRUE(args.IsString());
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectNoopWithoutMessages) {
    auto doc = parseJson(R"({"model": "test"})");
    input_workarounds::funcArgsToObjectJson(doc);
    // Should not crash
    EXPECT_TRUE(doc.HasMember("model"));
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectNoopWithoutToolCalls) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "user", "content": "hello"}
        ]
    })");
    input_workarounds::funcArgsToObjectJson(doc);
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "hello");
}

// --- ensureNonNullContentJson ---

TEST_F(InputWorkaroundsTest, ensureNonNullContentSetsNullToEmpty) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"function": {"name": "fn"}}
            ]}
        ]
    })");

    input_workarounds::ensureNonNullContentJson(doc);

    ASSERT_TRUE(doc["messages"][0]["content"].IsString());
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentAddsMissingContent) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "fn"}}
            ]}
        ]
    })");

    input_workarounds::ensureNonNullContentJson(doc);

    ASSERT_TRUE(doc["messages"][0].HasMember("content"));
    ASSERT_TRUE(doc["messages"][0]["content"].IsString());
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentPreservesExistingString) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": "some text", "tool_calls": [
                {"function": {"name": "fn"}}
            ]}
        ]
    })");

    input_workarounds::ensureNonNullContentJson(doc);

    ASSERT_TRUE(doc["messages"][0]["content"].IsString());
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "some text");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentSkipsMessagesWithoutToolCalls) {
    auto doc = parseJson(R"({
        "messages": [
            {"role": "user", "content": null}
        ]
    })");

    input_workarounds::ensureNonNullContentJson(doc);

    // User message without tool_calls should not be modified
    EXPECT_TRUE(doc["messages"][0]["content"].IsNull());
}

// --- applyToJson ---

TEST_F(InputWorkaroundsTest, applyToJsonAppliesObjectArgsWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;

    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "fn", "arguments": "{\"x\": 42}"}}
            ]}
        ]
    })");

    input_workarounds::applyToJson(caps, "gemma4", doc);

    ASSERT_TRUE(doc["messages"][0]["tool_calls"][0]["function"]["arguments"].IsObject());
    EXPECT_EQ(doc["messages"][0]["tool_calls"][0]["function"]["arguments"]["x"].GetInt(), 42);
}

TEST_F(InputWorkaroundsTest, applyToJsonAppliesNonNullContentWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresNonNullContent = true;

    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"function": {"name": "fn"}}
            ]}
        ]
    })");

    input_workarounds::applyToJson(caps, "llama3", doc);

    ASSERT_TRUE(doc["messages"][0]["content"].IsString());
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "");
}

TEST_F(InputWorkaroundsTest, applyToJsonDoesNothingWhenNoCapsSet) {
    ChatTemplateCaps caps;  // all defaults (false)

    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"function": {"name": "fn", "arguments": "{\"x\": 1}"}}
            ]}
        ]
    })");

    std::string before = serializeJson(doc);
    input_workarounds::applyToJson(caps, "", doc);
    std::string after = serializeJson(doc);

    EXPECT_EQ(before, after);
}

TEST_F(InputWorkaroundsTest, applyToJsonAppliesBothWorkarounds) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;
    caps.requiresNonNullContent = true;

    auto doc = parseJson(R"({
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"function": {"name": "fn", "arguments": "{\"key\": \"val\"}"}}
            ]}
        ]
    })");

    input_workarounds::applyToJson(caps, "test", doc);

    // Arguments should be converted to object
    ASSERT_TRUE(doc["messages"][0]["tool_calls"][0]["function"]["arguments"].IsObject());
    // Content should be non-null
    ASSERT_TRUE(doc["messages"][0]["content"].IsString());
    EXPECT_STREQ(doc["messages"][0]["content"].GetString(), "");
}
