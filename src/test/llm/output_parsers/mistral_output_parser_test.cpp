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

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "../../test_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\mistralai\\Mistral-7B-Instruct-v0.3";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/mistralai/Mistral-7B-Instruct-v0.3/";
#endif

static ov::genai::Tokenizer mistralTokenizer(tokenizerPath);

class MistralOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        // For Phi4 model there is only tool parser available
        outputParser = std::make_unique<OutputParser>(mistralTokenizer, "mistral", "");
    }
};

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    SPDLOG_INFO("AAAAAAAAAAA: [{}]", generatedTokens);
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    auto firstToolCallId = parsedOutput.toolCalls[0].id;

    EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
    EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
    auto secondToolCallId = parsedOutput.toolCalls[1].id;
    EXPECT_NE(firstToolCallId, secondToolCallId);  // IDs should be different

    EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
    EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);  // ID should be generated
    auto thirdToolCallId = parsedOutput.toolCalls[2].id;
    EXPECT_NE(firstToolCallId, thirdToolCallId);   // IDs should be different
    EXPECT_NE(secondToolCallId, thirdToolCallId);  // IDs should be different
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithOneValidToolCallAndTwoInvalid) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"tool_name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"options\": {\"key\": \"value\"}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");

    // Only the first tool call is valid, the second one has an invalid name field and the third one has an invalid arguments
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    auto firstToolCallId = parsedOutput.toolCalls[0].id;
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}
TEST_F(MistralOutputParserTest, ParseToolCallOutputWithMultipleFunctoolsReturnsContentOnly) {
    std::string input = "[TOOL_CALLS][{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}]\n\nThis is some content\n\[TOOL_CALLS][{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    // Content after 'functools' cannot be parsed as array of JSON objects, so it is treated as content
    EXPECT_EQ(parsedOutput.content, "functools[{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}]\n\nThis is some content\n\nfunctools[{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);  // No valid tool calls parsed
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithArrayArguments) {
    std::string input = "[TOOL_CALLS][{\"name\": \"extractLastTransactionId\", \"arguments\": { \"filepath\": \"/var/log/db.log\", \"status\": [\"completed\", \"failed\"], \"encoding\": \"utf-8\", \"processFunction\": \"processFunction\"}}]</s>";
    auto generatedTensor = mistralTokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "extractLastTransactionId");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}
