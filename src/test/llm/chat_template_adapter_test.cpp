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

#include "../../llm/io_processing/input_processors/chat_template_adapter.hpp"

using namespace ovms;

class ChatTemplateAdapterTest : public ::testing::Test {
protected:
    ov::genai::ChatHistory buildHistory(const std::string& messagesJson, std::string tools = {}) {
        ov::genai::ChatHistory history;
        auto container = ov::genai::JsonContainer::from_json_string(messagesJson);
        if (!tools.empty()) {
            history.set_tools(ov::genai::JsonContainer::from_json_string(tools));
        }
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

// --- toolResponseJsonContentToObjectHistory ---

TEST_F(ChatTemplateAdapterTest, toolResponseJsonContentConvertsObjectAndPreservesOpaqueValue) {
    static const std::string expectedSha = "798e99e04d53fba2b1c87bd6b88260f0d6c3ca83";
    auto history = buildHistory(R"([
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_repo", "type": "function", "function": {"name": "inspect_repository_state", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_repo", "name": "inspect_repository_state",
         "content": "{\"head_sha\":\"798e99e04d53fba2b1c87bd6b88260f0d6c3ca83\",\"nested\":{\"dirty\":true}}"}
    ])");

    chat_template_adapter::toolResponseJsonContentToObjectHistory(history);

    ASSERT_GE(history.size(), 2u);
    auto content = history[1]["content"];
    ASSERT_TRUE(content.is_object());
    EXPECT_EQ(content["head_sha"].get_string(), expectedSha);
    EXPECT_EQ(content["nested"].to_json_string(), R"({"dirty":true})");
}

TEST_F(ChatTemplateAdapterTest, toolResponseJsonContentLeavesNonJsonStringUntouched) {
    auto history = buildHistory(R"([
        {"role": "tool", "tool_call_id": "call_1", "content": "not json at all"}
    ])");

    chat_template_adapter::toolResponseJsonContentToObjectHistory(history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "not json at all");
}

TEST_F(ChatTemplateAdapterTest, toolResponseJsonContentLeavesJsonArrayStringUntouched) {
    auto history = buildHistory(R"([
        {"role": "tool", "tool_call_id": "call_1", "content": "[1,2,3]"}
    ])");

    chat_template_adapter::toolResponseJsonContentToObjectHistory(history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), "[1,2,3]");
}

TEST_F(ChatTemplateAdapterTest, toolResponseJsonContentSkipsNonToolMessages) {
    auto history = buildHistory(R"([
        {"role": "user", "content": "{\"head_sha\":\"do-not-touch\"}"},
        {"role": "assistant", "content": "{\"head_sha\":\"also-do-not-touch\"}"}
    ])");

    chat_template_adapter::toolResponseJsonContentToObjectHistory(history);

    ASSERT_TRUE(history[0]["content"].is_string());
    EXPECT_EQ(history[0]["content"].get_string(), R"({"head_sha":"do-not-touch"})");
    ASSERT_TRUE(history[1]["content"].is_string());
    EXPECT_EQ(history[1]["content"].get_string(), R"({"head_sha":"also-do-not-touch"})");
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

TEST_F(ChatTemplateAdapterTest, applyToHistoryStructuresToolResponseWhenCapabilitySet) {
    ChatTemplateCaps caps;
    caps.parseToolResponseJsonContent = true;

    auto history = buildHistory(R"([
        {"role": "tool", "tool_call_id": "call_repo",
         "content": "{\"commit_sha\":\"798e99e04d53fba2b1c87bd6b88260f0d6c3ca83\",\"verdict\":\"PASS\"}"}
    ])");

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_TRUE(history[0]["content"].is_object());
    EXPECT_EQ(history[0]["content"]["commit_sha"].get_string(), "798e99e04d53fba2b1c87bd6b88260f0d6c3ca83");
    EXPECT_EQ(history[0]["content"]["verdict"].get_string(), "PASS");
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

TEST_F(ChatTemplateAdapterTest, applyToHistoryHandlesToolDefinitionsWithoutResponses) {
    ChatTemplateCaps caps;
    caps.supportsResponseFieldInToolDefinition = true;

    const std::string toolDefinitions = R"([{"type": "function", "function": {"name": "cat", "description": "This tool belongs to the Gorilla file system...", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]}, "response": {"type": "dict", "properties": {"file_content": {"type": "string", "description": "The content of the file."}}}}},
                                            {"type": "function", "function": {"name": "cd", "description": "This tool belongs to the Gorilla file system...", "parameters": {"type": "object", "properties": {"folder": {"type": "string", "description": "The folder of the directory to change to. You can only change one folder level at a time. "}}, "required": ["folder"]}, "response": {"type": "dict", "properties": {"current_working_directory": {"type": "string", "description": "The new current working directory path."}}}}}])";

    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"}
    ])",
        toolDefinitions);

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_EQ(history.size(), 1);
    auto toolDefinitionArray = history.get_tools();
    ASSERT_TRUE(toolDefinitionArray.is_array());
    EXPECT_EQ(toolDefinitionArray.size(), 2u);
    EXPECT_EQ(toolDefinitionArray[0]["function"]["name"], "cat");
    EXPECT_EQ(toolDefinitionArray[1]["function"]["name"], "cd");
    EXPECT_FALSE(toolDefinitionArray[0]["function"].contains("response"));
    EXPECT_FALSE(toolDefinitionArray[1]["function"].contains("response"));
}

TEST_F(ChatTemplateAdapterTest, applyToHistoryHandlesRemovedResponsesInToolDefinitions) {
    ChatTemplateCaps caps;
    caps.supportsResponseFieldInToolDefinition = true;

    const std::string toolDefinitions = R"([{"type": "function", "function": {"name": "cat", "description": "This tool belongs to the Gorilla file system...", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]}}},
                                            {"type": "function", "function": {"name": "cd", "description": "This tool belongs to the Gorilla file system...", "parameters": {"type": "object", "properties": {"folder": {"type": "string", "description": "The folder of the directory to change to. You can only change one folder level at a time. "}}, "required": ["folder"]}}}])";

    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"}
    ])",
        toolDefinitions);

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_EQ(history.size(), 1);
    auto toolDefinitionArray = history.get_tools();
    ASSERT_TRUE(toolDefinitionArray.is_array());
    EXPECT_EQ(toolDefinitionArray.size(), 2u);
    EXPECT_EQ(toolDefinitionArray[0]["function"]["name"], "cat");
    EXPECT_EQ(toolDefinitionArray[1]["function"]["name"], "cd");
    EXPECT_FALSE(toolDefinitionArray[0]["function"].contains("response"));
    EXPECT_FALSE(toolDefinitionArray[1]["function"].contains("response"));
}

TEST_F(ChatTemplateAdapterTest, applyToHistoryHandlesEmptyToolDefinitions) {
    ChatTemplateCaps caps;
    caps.supportsResponseFieldInToolDefinition = true;

    auto history = buildHistory(R"([
        {"role": "user", "content": "hello"}
    ])");

    chat_template_adapter::applyToHistory(caps, history);

    ASSERT_EQ(history.size(), 1);
    auto toolDefinitionArray = history.get_tools();
    ASSERT_TRUE(toolDefinitionArray.is_array());
    EXPECT_EQ(toolDefinitionArray.size(), 0);
}
