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
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\meta-llama\\Llama-3.1-8B-Instruct";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/meta-llama/Llama-3.1-8B-Instruct";
#endif

// Id of the <|python_tag|> which is a special token used to indicate the start of a tool calls
constexpr int64_t botTokenId = 128010;

class Llama3OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        outputParser = std::make_unique<OutputParser>(*tokenizer, "llama3");
    }
};

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}}";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    generatedTokens.insert(generatedTokens.begin(), botTokenId);  // Prepend bot token ID
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}};"
                        "{\"name\": \"another_tool\", \"parameters\": {\"param1\": \"data\", \"param2\": true}};"
                        "{\"name\": \"third_tool\", \"parameters\": {\"key\": \"value\"}}";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    generatedTokens.insert(generatedTokens.begin(), botTokenId);  // Prepend bot token ID
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);

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

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string content = "This is a content part and next will be a tool call.";
    std::string toolCall = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}}";
    auto generatedContentTensor = tokenizer->encode(content, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedContentTokens(generatedContentTensor.data<int64_t>(), generatedContentTensor.data<int64_t>() + generatedContentTensor.get_size());
    auto generatedToolCallTensor = tokenizer->encode(toolCall, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedToolCallTokens(generatedToolCallTensor.data<int64_t>(), generatedToolCallTensor.data<int64_t>() + generatedToolCallTensor.get_size());
    std::vector<int64_t> generatedTokens;
    generatedTokens.insert(generatedTokens.end(), generatedContentTokens.begin(), generatedContentTokens.end());
    generatedTokens.insert(generatedTokens.end(), botTokenId);  // Add bot token ID
    generatedTokens.insert(generatedTokens.end(), generatedToolCallTokens.begin(), generatedToolCallTokens.end());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}
