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
#include "../../test_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\microsoft\\Phi-4-mini-instruct";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/microsoft/Phi-4-mini-instruct";
#endif

class Phi4ResponseParserTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    std::unique_ptr<ResponseParser> responseParser;

    void SetUp() override {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        responseParser = std::make_unique<ResponseParser>(*tokenizer, "phi4");
    }
};

TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "functools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]";
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

TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "functools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}]";
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

TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedResponse.toolCalls.size(), 0);
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);
}

TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\nfunctools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
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
TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithMultipleFunctoolsThrows) {
    std::string input = "functools[{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}]\n\nThis is some content\n\nfunctools[{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    EXPECT_THROW({
        responseParser->parse(generatedTokens);
    },
        std::runtime_error);
}

TEST_F(Phi4ResponseParserTest, ParseToolCallOutputWithArrayArguments) {
    //std::string input = "functools[{\"name\": \"extractLastTransactionId\", \"arguments\": {\n  \"filepath\": \"/var/log/db.log\",\n  \"status\": [\"completed\", \"failed\"],\n  \"encoding\": \"utf-8\",\n  \"processFunction\": \"processFunction\"\n}}]";
    std::string input = "functools[{\"name\": \"extractLastTransactionId\", \"arguments\": { \"filepath\": \"/var/log/db.log\", \"status\": [\"completed\", \"failed\"], \"encoding\": \"utf-8\", \"processFunction\": \"processFunction\"}}]";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedResponse parsedResponse = responseParser->parse(generatedTokens);
    EXPECT_EQ(parsedResponse.content, "");
    EXPECT_EQ(parsedResponse.reasoning, "");
    EXPECT_EQ(parsedResponse.reasoningTokenCount, 0);
    ASSERT_EQ(parsedResponse.toolCalls.size(), 1);
    EXPECT_EQ(parsedResponse.toolCalls[0].name, "extractLastTransactionId");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedResponse.toolCalls[0].arguments, "{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}");
    EXPECT_EQ(parsedResponse.toolCalls[0].id.empty(), false);  // ID should be generated
}

