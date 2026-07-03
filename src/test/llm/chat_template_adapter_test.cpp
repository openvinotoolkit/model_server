//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "../../llm/io_processing/chat_template_adapter.hpp"

using namespace ovms;

class ChatTemplateAdapterTest : public ::testing::Test {
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

TEST_F(ChatTemplateAdapterTest, funcArgsToObjectConvertsStringArgs) {
    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"London\", \"units\": \"celsius\"}"
            }}
        ]}
    ])");

    chat_template_adapter::funcArgsToObjectHistory(history);

    ASSERT_GE(history.size(), 2u);
    auto toolCalls = history[1]["tool_calls"];
    ASSERT_TRUE(toolCalls.is_array());
    ASSERT_GE(toolCalls.size(), 1u);
    auto args = toolCalls[0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args["city"].get_string(), "London");
    EXPECT_EQ(args["units"].get_string(), "celsius");
}

TEST_F(ChatTemplateAdapterTest, funcArgsToObjectHandlesMultipleToolCalls) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "fn1", "arguments": "{\"a\": 1}"}},
            {"id": "call_2", "function": {"name": "fn2", "arguments": "{\"b\": true}"}}
        ]}
    ])");

    chat_template_adapter::funcArgsToObjectHistory(history);

    ASSERT_GE(history.size(), 1u);
    auto toolCalls = history[0]["tool_calls"];
    ASSERT_TRUE(toolCalls.is_array());
    ASSERT_GE(toolCalls.size(), 2u);

    auto args1 = toolCalls[0]["function"]["arguments"];
    ASSERT_TRUE(args1.is_object());
    EXPECT_EQ(args1.to_json_string(), R"({"a":1})");

    auto args2 = toolCalls[1]["function"]["arguments"];
    ASSERT_TRUE(args2.is_object());
    EXPECT_EQ(args2.to_json_string(), R"({"b":true})");
}

TEST_F(ChatTemplateAdapterTest, funcArgsToObjectSkipsAlreadyObjectArgs) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": {"key": "value"}}}
        ]}
    ])");

    chat_template_adapter::funcArgsToObjectHistory(history);

    ASSERT_GE(history.size(), 1u);
    auto toolCalls = history[0]["tool_calls"];
    ASSERT_TRUE(toolCalls.is_array());
    ASSERT_GE(toolCalls.size(), 1u);
    auto args = toolCalls[0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args["key"].get_string(), "value");
}

TEST_F(ChatTemplateAdapterTest, funcArgsToObjectSkipsInvalidJsonString) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": "not valid json {"}}
        ]}
    ])");

    chat_template_adapter::funcArgsToObjectHistory(history);

    ASSERT_GE(history.size(), 1u);
    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    EXPECT_TRUE(args.is_string());
}

TEST_F(ChatTemplateAdapterTest, funcArgsToObjectNoopWithoutToolCalls) {
    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"}
    ])");

    chat_template_adapter::funcArgsToObjectHistory(history);

    ASSERT_GE(history.size(), 1u);
    EXPECT_EQ(history[0]["content"].get_string(), "hello");
}

// --- ensureNonNullContentHistory ---

TEST_F(ChatTemplateAdapterTest, ensureNonNullContentSetsNullToEmpty) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    chat_template_adapter::ensureNonNullContentHistory(history);

    ASSERT_GE(history.size(), 1u);
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(ChatTemplateAdapterTest, ensureNonNullContentAddsMissingContent) {
    auto history = buildHistory(R"([
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    chat_template_adapter::ensureNonNullContentHistory(history);

    ASSERT_GE(history.size(), 1u);
    EXPECT_TRUE(history[0].contains("content"));
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(ChatTemplateAdapterTest, ensureNonNullContentPreservesExistingString) {
    auto history = buildHistory(R"([
        {"role": "assistant", "content": "some text", "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    chat_template_adapter::ensureNonNullContentHistory(history);

    ASSERT_GE(history.size(), 1u);
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "some text");
}

TEST_F(ChatTemplateAdapterTest, ensureNonNullContentSkipsMessagesWithoutToolCalls) {
    auto history = buildHistory(R"([
        {"role": "user", "content": null}
    ])");

    chat_template_adapter::ensureNonNullContentHistory(history);

    // User message without tool_calls should not be modified
    ASSERT_GE(history.size(), 1u);
    EXPECT_TRUE(history[0]["content"].is_null());
}

// --- applyToHistory ---

TEST_F(ChatTemplateAdapterTest, applyToHistoryAppliesObjectArgsWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"x\": 42}"}}
        ]}
    ])");

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_GE(history.size(), 1u);
    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    EXPECT_EQ(args.to_json_string(), R"({"x":42})");
}

TEST_F(ChatTemplateAdapterTest, applyToHistoryAppliesNonNullContentWhenRequired) {
    ChatTemplateCaps caps;
    caps.requiresNonNullContent = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn"}}
        ]}
    ])");

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_GE(history.size(), 1u);
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}

TEST_F(ChatTemplateAdapterTest, applyToHistoryDoesNothingWhenNoCapsSet) {
    ChatTemplateCaps caps;  // all defaults (false)

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"x\": 1}"}}
        ]}
    ])");

    std::string before = history.get_messages().to_json_string();
    chat_template_adapter::applyToHistory(caps, history);
    std::string after = history.get_messages().to_json_string();

    EXPECT_EQ(before, after);
}

TEST_F(ChatTemplateAdapterTest, applyToHistoryAppliesBothWorkarounds) {
    ChatTemplateCaps caps;
    caps.requiresObjectArguments = true;
    caps.requiresNonNullContent = true;

    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "fn", "arguments": "{\"key\": \"val\"}"}}
        ]}
    ])");

    chat_template_adapter::applyToHistory(caps, history);

    // Arguments should be converted to object
    ASSERT_GE(history.size(), 1u);
    auto args = history[0]["tool_calls"][0]["function"]["arguments"];
    ASSERT_TRUE(args.is_object());
    // Content should be non-null
    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "");
}
