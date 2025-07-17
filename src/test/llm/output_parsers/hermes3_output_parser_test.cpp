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
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\NousResearch\\Hermes-3-Llama-3.1-8B";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/NousResearch/Hermes-3-Llama-3.1-8B";
#endif

class Hermes3OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        outputParser = std::make_unique<OutputParser>(*tokenizer, "hermes3");
    }
};

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string inputWithProperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    std::string inputWithImproperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}";

    // Hermes3 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
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
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string inputWithProperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                                         "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                                         "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    std::string inputWithImproperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                                           "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                                           "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}";

    // Hermes3 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
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
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.reasoningTokenCount, 0);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}
