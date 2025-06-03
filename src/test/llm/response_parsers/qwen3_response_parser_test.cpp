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
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "../../../llm/response_parsers/base_response_parser.hpp"
#include "../../../llm/response_parsers/response_parser.hpp"

using namespace ovms;

const std::string tokenizerPath = "/ovms/src/test/llm_testing/Qwen/Qwen3-8B";

class Qwen3ResponseParserTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    std::unique_ptr<ResponseParser> responseParser;

    void SetUp() override {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        responseParser = std::make_unique<ResponseParser>(*tokenizer, "qwen3");
    }
};

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithSingleToolCallNoThinking) {
    std::string input = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "");
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);
    ASSERT_EQ(parsedResponse.toolCalls.size(), 1);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithSingleToolCallAndThinking) {
    std::string input = "<think>Thinking about the tool call</think>"
                        "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "");
    EXPECT_EQ(parsedResponse.reasoning, "Thinking about the tool call");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 5);  // Number of tokens in "Thinking about the tool call"
    ASSERT_EQ(parsedResponse.toolCalls.size(), 1);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithThreeToolCallsNoThinking) {
    std::string input = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                        "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                        "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "");
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);

    ASSERT_EQ(parsedResponse.toolCalls.size(), 3);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
    auto firstToolCallId = parsedResponse.toolCalls[0].id;

    EXPECT_EQ(parsedResponse.toolCalls[1].name, "another_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
    EXPECT_EQ(parsedResponse.toolCalls[1].id.empty(), false);  // ID should be generated
    auto secondToolCallId = parsedResponse.toolCalls[1].id;
    EXPECT_NE(firstToolCallId, secondToolCallId);  // IDs should be different

    EXPECT_EQ(parsedResponse.toolCalls[2].name, "third_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[2].arguments, "{\"key\":\"value\"}");
    EXPECT_EQ(parsedResponse.toolCalls[2].id.empty(), false);  // ID should be generated
    auto thirdToolCallId = parsedResponse.toolCalls[2].id;
    EXPECT_NE(firstToolCallId, thirdToolCallId);   // IDs should be different
    EXPECT_NE(secondToolCallId, thirdToolCallId);  // IDs should be different
}

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithThreeToolCallsAndThinking) {
    std::string input = "<think>Thinking about the tool calls</think>"
                        "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                        "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                        "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "");
    EXPECT_EQ(parsedResponse.reasoning, "Thinking about the tool calls");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 5);  // Number of tokens in "Thinking about the tool calls"

    ASSERT_EQ(parsedResponse.toolCalls.size(), 3);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
    auto firstToolCallId = parsedResponse.toolCalls[0].id;

    EXPECT_EQ(parsedResponse.toolCalls[1].name, "another_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
    EXPECT_EQ(parsedResponse.toolCalls[1].id.empty(), false);  // ID should be generated
    auto secondToolCallId = parsedResponse.toolCalls[1].id;
    EXPECT_NE(firstToolCallId, secondToolCallId);  // IDs should be different

    EXPECT_EQ(parsedResponse.toolCalls[2].name, "third_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[2].arguments, "{\"key\":\"value\"}");
    EXPECT_EQ(parsedResponse.toolCalls[2].id.empty(), false);  // ID should be generated
    auto thirdToolCallId = parsedResponse.toolCalls[2].id;
    EXPECT_NE(firstToolCallId, thirdToolCallId);   // IDs should be different
    EXPECT_NE(secondToolCallId, thirdToolCallId);  // IDs should be different
}

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedResponse.toolCalls.size(), 0);
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);
}

TEST_F(Qwen3ResponseParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);
    ASSERT_EQ(parsedResponse.toolCalls.size(), 1);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
}
