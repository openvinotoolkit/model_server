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
#include <openvino/genai/chat_history.hpp>

#include "../../llm/input_workarounds.hpp"

using namespace ovms;

class InputWorkaroundsTest : public ::testing::Test {
protected:
    ov::genai::ChatHistory buildHistory(const std::string& messagesJson) {
        ov::genai::ChatHistory history;
        auto container = ov::genai::JsonContainer::from_json_string(messagesJson);
        for (size_t i = 0; i < container.size(); ++i) {
            history.push_back(container[i]);
        }
        return history;
    }
};

// --- funcArgsToObjectHistory ---

TEST_F(InputWorkaroundsTest, funcArgsToObjectConvertsStringArgs) {
    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"London\", \"units\": \"celsius\"}"
            }}
        ]}
    ])");

    input_workarounds::funcArgsToObjectHistory(history);

    auto args = history[1]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args["city"].get_string(), "London");
    EXPECT_EQ(args["units"].get_string(), "celsius");
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectHandlesMultipleToolCalls) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "fn1", "arguments": "{\"a\": 1}"}},
            {"id": "call_2", "function": {"name": "fn2", "arguments": "{\"b\": true}"}}
        ]}
    ])");

    input_workarounds::funcArgsToObjectHistory(history);

    auto args1 = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args1.is_object());
    EXPECT_EQ(args1.to_json_string(), R"({"a":1})");

    auto args2 = history[0]["tool_calls"][1]["function"]["arguments"];
    ASSERT_TRUE(args2.is_object());
    EXPECT_EQ(args2.to_json_string(), R"({"b":true})");
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectSkipsAlreadyObjectArgs) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": {"key": "value"}}}
        ]}
    ])");

    input_workarounds::funcArgsToObjectHistory(history);

    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args["key"].get_string(), "value");
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectSkipsInvalidJsonString) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": "not valid json {"}}
        ]}
    ])");

    input_workarounds::funcArgsToObjectHistory(history);

    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    EXPECT_TRUE(args.is_string());
}

TEST_F(InputWorkaroundsTest, funcArgsToObjectNoopWithoutToolCalls) {
    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"}
    ])");

    input_workarounds::funcArgsToObjectHistory(history);

    EXPECT_EQ(history[0]["content"].get_string(), "hello");
}

// --- ensureNonNullContentHistory ---

TEST_F(InputWorkaroundsTest, ensureNonNullContentSetsNullToEmpty) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    input_workarounds::ensureNonNullContentHistory(history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentAddsMissingContent) {
    auto history = buildHistory(R"([
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    input_workarounds::ensureNonNullContentHistory(history);

    EXPECT_TRUE(history[0].contains("content"));
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentPreservesExistingString) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "some text", "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    input_workarounds::ensureNonNullContentHistory(history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "some text");
}

TEST_F(InputWorkaroundsTest, ensureNonNullContentSkipsMessagesWithoutToolCalls) {
    auto history = buildHistory(R"([
        {"role": "user", "content": null}
    ])");

    input_workarounds::ensureNonNullContentHistory(history);

    // User message without tool_calls should not be modified
    EXPECT_TRUE(history[0]["content"].is_null());
}

// --- applyToHistory ---

TEST_F(InputWorkaroundsTest, applyToHistoryAppliesObjectArgsWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"x\": 42}"}}
        ]}
    ])");

    input_workarounds::applyToHistory(caps, "gemma4", history);

    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args.to_json_string(), R"({"x":42})");
}

TEST_F(InputWorkaroundsTest, applyToHistoryAppliesNonNullContentWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresNonNullContent = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    input_workarounds::applyToHistory(caps, "llama3", history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(InputWorkaroundsTest, applyToHistoryDoesNothingWhenNoCapsSet) {
    ChatTemplateCaps caps;  // all defaults (false)

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"x\": 1}"}}
        ]}
    ])");

    std::string before = history.get_messages().to_json_string();
    input_workarounds::applyToHistory(caps, "", history);
    std::string after = history.get_messages().to_json_string();

    EXPECT_EQ(before, after);
}

TEST_F(InputWorkaroundsTest, applyToHistoryAppliesBothWorkarounds) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;
    caps.requiresNonNullContent = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"key\": \"val\"}"}}
        ]}
    ])");

    input_workarounds::applyToHistory(caps, "test", history);

    // Arguments should be converted to object
    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    // Content should be non-null
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}
