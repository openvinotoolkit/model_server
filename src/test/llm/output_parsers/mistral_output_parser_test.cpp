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

static std::unique_ptr<ov::genai::Tokenizer> mistralTokenizer;

class MistralOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;
    std::unique_ptr<OutputParser> outputParserWithImmediateToolParsing;

    static void SetUpTestSuite() {
        try {
            mistralTokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize mistral tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize mistral tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        mistralTokenizer.reset();
    }

    void SetUp() override {
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*mistralTokenizer, "mistral", "");
        outputParserWithImmediateToolParsing = std::make_unique<OutputParser>(*mistralTokenizer, "mistral", "");
        outputParserWithImmediateToolParsing->enableImmediateToolParsing();
    }
};

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]</s>";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("[TOOL_CALLS]").length());
        }
        auto generatedTensor = mistralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}]</s>";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("[TOOL_CALLS]").length());
        }
        auto generatedTensor = mistralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;
        EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
        auto secondToolCallId = parsedOutput.toolCalls[1].id;
        EXPECT_NE(firstToolCallId, secondToolCallId);
        EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
        EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
        EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);
        auto thirdToolCallId = parsedOutput.toolCalls[2].id;
        EXPECT_NE(firstToolCallId, thirdToolCallId);
        EXPECT_NE(secondToolCallId, thirdToolCallId);
    }
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithOneValidToolCallAndTwoInvalid) {
    std::string input = "[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"tool_name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"options\": {\"key\": \"value\"}}]</s>";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("[TOOL_CALLS]").length());
        }
        auto generatedTensor = mistralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;
    }
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    for (bool immediateParsing : {false, true}) {
        auto generatedTensor = mistralTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
        EXPECT_EQ(parsedOutput.reasoning, "");
    }
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]</s>";
    for (bool immediateParsing : {false, true}) {
        auto generatedTensor = mistralTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n [{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}
TEST_F(MistralOutputParserTest, ParseToolCallOutputWithContentOnBothSidesAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n[TOOL_CALLS][{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]</s> This is a content part after tool call.";
    for (bool immediateParsing : {false, true}) {
        auto generatedTensor = mistralTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n [{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}] This is a content part after tool call.");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}
TEST_F(MistralOutputParserTest, ParseToolCallOutputWithMultipleToolCallsReturnsContentOnly) {
    std::string input = "[TOOL_CALLS][{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}]</s> \n\nThis is some content\n\n[TOOL_CALLS][{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]</s>";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("[TOOL_CALLS]").length());
        }
        auto generatedTensor = mistralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        // Same expected content as tokenizer does not add special tokens
        EXPECT_EQ(parsedOutput.content, "[{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}] \n\nThis is some content\n\n [{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithArrayArguments) {
    std::string input = "[TOOL_CALLS][{\"name\": \"extractLastTransactionId\", \"arguments\": { \"filepath\": \"/var/log/db.log\", \"status\": [\"completed\", \"failed\"], \"encoding\": \"utf-8\", \"processFunction\": \"processFunction\"}}]</s>";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("[TOOL_CALLS]").length());
        }
        auto generatedTensor = mistralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "extractLastTransactionId");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}
