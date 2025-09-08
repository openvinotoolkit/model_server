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

static std::unique_ptr<ov::genai::Tokenizer> hermes3Tokenizer;

class Hermes3OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;
    std::unique_ptr<OutputParser> outputParserWithImmediateToolParsing;

    static void SetUpTestSuite() {
        try {
            hermes3Tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize hermes3 tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize hermes3 tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        hermes3Tokenizer.reset();
    }

    void SetUp() override {
        // For Hermes3 model there is only tool parser available
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*hermes3Tokenizer, "hermes3", "");
        outputParserWithImmediateToolParsing = std::make_unique<OutputParser>(*hermes3Tokenizer, "hermes3", "");
        outputParserWithImmediateToolParsing->enableImmediateToolParsing();
    }
};

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string inputWithProperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    std::string inputWithImproperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}";

    // Hermes3 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    for (bool immediateParsing : {false, true}) {
        std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
        for (auto& input : inputs) {
            if (immediateParsing) {
                // Remove opening tag for immediate parsing
                input = input.substr(std::string("<tool_call>").length());
            }
            auto generatedTensor = hermes3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
            std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
            ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
            EXPECT_EQ(parsedOutput.content, "");
            EXPECT_EQ(parsedOutput.reasoning, "");

            ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
            EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
            // Parser removes whitespaces, so we expect arguments value to be without spaces
            EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
            EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
        }
    }
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithNoToolsInTheRequest) {
    std::string inputWithProperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    std::string inputWithImproperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}";

    // Hermes3 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    for (bool immediateParsing : {false, true}) {
        std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
        for (auto& input : inputs) {
            std::string testInput = input;
            if (immediateParsing) {
                // Remove opening tag for immediate parsing
                testInput = testInput.substr(std::string("<tool_call>").length());
            }
            auto generatedTensor = hermes3Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
            std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
            ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, false) : outputParserWithRegularToolParsing->parse(generatedTokens, false);
            EXPECT_EQ(parsedOutput.content, testInput);
            EXPECT_EQ(parsedOutput.reasoning, "");

            ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
        }
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
    for (bool immediateParsing : {false, true}) {
        std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
        for (auto& input : inputs) {
            if (immediateParsing) {
                input = input.substr(std::string("<tool_call>").length());
            }
            auto generatedTensor = hermes3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
            std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
            ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
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
    }
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithTwoValidToolCallsAndOneInvalid) {
    std::string inputWithProperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                                         "<tool_call>{\"tool_name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                                         "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    std::string inputWithImproperClosure = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                                           "<tool_call>{\"tool_name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                                           "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}";

    // Hermes3 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (bool immediateParsing : {false, true}) {
        for (auto& input : inputs) {
            if (immediateParsing) {
                input = input.substr(std::string("<tool_call>").length());
            }
            auto generatedTensor = hermes3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
            std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
            ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
            EXPECT_EQ(parsedOutput.content, "");
            EXPECT_EQ(parsedOutput.reasoning, "");

            // Expecting two tool calls as the second one does not have a valid name
            ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
            EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
            // Parser removes whitespaces, so we expect arguments value to be without spaces
            EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
            EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
            auto firstToolCallId = parsedOutput.toolCalls[0].id;

            EXPECT_EQ(parsedOutput.toolCalls[1].name, "third_tool");
            // Parser removes whitespaces, so we expect arguments value to be without spaces
            EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"key\":\"value\"}");
            EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
            auto secondToolCallId = parsedOutput.toolCalls[1].id;
            EXPECT_NE(firstToolCallId, secondToolCallId);  // IDs should be different
        }
    }
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = hermes3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");

    // Immediate parsing expects tool call right away, so it fails yielding empty both content and tool calls
    ParsedOutput parsedOutputImmediate = outputParserWithImmediateToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutputImmediate.content, "");
    ASSERT_EQ(parsedOutputImmediate.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutputImmediate.reasoning, "");
}

TEST_F(Hermes3OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = hermes3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

// Major positive test for streaming tool calls with multiple chunks and phase switching
// Attempt thinking, but without reasoning parser, deltas should not contain reasoning content
TEST_F(Hermes3OutputParserTest, HolisticStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<think>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"content\":\"<think>\"}}"},
        {"Some thinking content", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"content\":\"Some thinking content\"}}"},
        {"</think>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"content\":\"</think>\"}}"},
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"<tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"super", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"arguments", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // As we have 'arguments' key present, we can return first delta
        {"\":", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"super_tool\"}}]}}"},
        // Consecutive deltas without 'id' and 'type'. In order to find the end of arguments parser has one chunk delay to handle end of tool.
        {" {", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]}}"},
        {"arg1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"arg1\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"value1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\", ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"value1\"}}]}}"},
        {"arg2", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\", \"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"arg2\"}}]}}"},
        {"{\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_arg1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"\"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_arg1\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_value1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\", ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_value1\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\", \"}}]}}"},
        {"nested_arg2", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_arg2\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_value2", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\"}}}", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_value2\"}}]}}"},
        {"</tool_call>\n", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}}\"}}]}}"},
        // Starting second tool. Collecting chunk until full name is received. Don't return until then.
        {"<tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"super", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_number", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_two", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"arguments", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // As we have 'arguments' key present, we can return first delta
        {"\":", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"super_tool_number_two\"}}]}}"},
        // Consecutive deltas without 'id' and 'type'. In order to find the end of arguments parser has one chunk delay to handle end of tool.
        {" {", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\"}}]}}"},
        {"arg1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"arg1\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"val{{{ue1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"val{{{ue1\"}}]}}"},
        {"}", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"}", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]}}"},  // returning last arguments part
        {"</tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},                                                          // closed main JSON, with the last chunk, now only return nullopt
        // Starting third tool. Collecting chunk until full name is received. Don't return until then.
        {"<tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"super", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_number", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_three", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"arguments", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // As we have 'arguments' key present, we can return first delta
        {"\":", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":2,\"function\":{\"name\":\"super_tool_number_three\"}}]}}"},
        // Consecutive deltas without 'id' and 'type'. In order to find the end of arguments parser has one chunk delay to handle end of tool.
        {" {", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":2,\"function\":{\"arguments\":\"{\"}}]}}"},
        {"arg1", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":2,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":2,\"function\":{\"arguments\":\"arg1\"}}]}}"},
        {"\"", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"index\":2,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        // Simulating hitting max tokens while during tool call generation. We should return the last two chunks as delta to flush the delay window
        {"val,", ov::genai::GenerationFinishReason::LENGTH, "{\"delta\":{\"tool_calls\":[{\"index\":2,\"function\":{\"arguments\":\"\\\"val,\"}}]}}"},  // clo
    };

    for (bool immediateParsing : {false, true}) {
        if (immediateParsing) {
            chunkToDeltaVec.erase(chunkToDeltaVec.begin(), chunkToDeltaVec.begin() + 4);
            chunkToDeltaVec.insert(chunkToDeltaVec.begin(), {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt});
        }
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
            std::optional<rapidjson::Document> doc = immediateParsing ? outputParserWithImmediateToolParsing->parseChunk(chunk, true, finishReason) : outputParserWithRegularToolParsing->parseChunk(chunk, true, finishReason);
            if (!expectedDelta.has_value() && !doc.has_value()) {
                continue;  // Both are nullopt, OK
            }
            if (expectedDelta.has_value() && doc.has_value()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                doc->Accept(writer);
                std::string docStr = buffer.GetString();
                // If both strings contain "id":"...", compare id values by length and alphanumeric, else compare whole strings
                std::string expected = expectedDelta.value();
                std::string idKey = "\"id\":\"";
                auto docIdPos = docStr.find(idKey);
                auto expectedIdPos = expected.find(idKey);
                if (docIdPos != std::string::npos && expectedIdPos != std::string::npos) {
                    auto docIdStart = docIdPos + idKey.size();
                    auto docIdEnd = docStr.find("\"", docIdStart);
                    auto expectedIdStart = expectedIdPos + idKey.size();
                    auto expectedIdEnd = expected.find("\"", expectedIdStart);
                    ASSERT_NE(docIdEnd, std::string::npos);
                    ASSERT_NE(expectedIdEnd, std::string::npos);
                    std::string docId = docStr.substr(docIdStart, docIdEnd - docIdStart);
                    std::string expectedId = expected.substr(expectedIdStart, expectedIdEnd - expectedIdStart);
                    EXPECT_EQ(docId.size(), expectedId.size()) << "ID length mismatch for chunk: " << chunk;
                    EXPECT_TRUE(std::all_of(docId.begin(), docId.end(), ::isalnum)) << "ID not alphanumeric for chunk: " << chunk;
                    // Compare everything except the id value
                    std::string docStrNoId = docStr;
                    std::string expectedNoId = expected;
                    docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                    expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                    EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk (ignoring id value): " << chunk;
                } else {
                    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
                }
            } else {
                std::string expectedStr = expectedDelta.has_value() ? expectedDelta.value() : "std::nullopt";
                std::string docStr = doc.has_value() ? [&]() {
                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    doc->Accept(writer);
                    return std::string(buffer.GetString());
                }()
                                                     : "std::nullopt";
                FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk
                       << "\nexpectedDelta: " << expectedStr
                       << "\ndoc: " << docStr;
            }
        }
    }
}

TEST_F(Hermes3OutputParserTest, ToolCallsWithoutToolsInTheRequestStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Tool parser is available, but tools are not in the request so every chunk is just a regular content
        {"<tool_call>\n", "{\"delta\":{\"content\":\"<tool_call>\\n\"}}"},
        {"{\"", "{\"delta\":{\"content\":\"{\\\"\"}}"},
        {"name", "{\"delta\":{\"content\":\"name\"}}"},
        {"\":", "{\"delta\":{\"content\":\"\\\":\"}}"},
        {" \"", "{\"delta\":{\"content\":\" \\\"\"}}"},
        {"super", "{\"delta\":{\"content\":\"super\"}}"},
        {"_tool", "{\"delta\":{\"content\":\"_tool\"}}"},
        {"_number", "{\"delta\":{\"content\":\"_number\"}}"},
        {"_two", "{\"delta\":{\"content\":\"_two\"}}"},
        {"\",", "{\"delta\":{\"content\":\"\\\",\"}}"},
        {" \"", "{\"delta\":{\"content\":\" \\\"\"}}"},
        {"arguments", "{\"delta\":{\"content\":\"arguments\"}}"},
        {"\":", "{\"delta\":{\"content\":\"\\\":\"}}"},
        {" {", "{\"delta\":{\"content\":\" {\"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"arg1", "{\"delta\":{\"content\":\"arg1\"}}"},
        {"\": ", "{\"delta\":{\"content\":\"\\\": \"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"val{{{ue1", "{\"delta\":{\"content\":\"val{{{ue1\"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"}", "{\"delta\":{\"content\":\"}\"}}"},
        {"}", "{\"delta\":{\"content\":\"}\"}}"},
        {"</tool_call>\n", "{\"delta\":{\"content\":\"</tool_call>\\n\"}}"},
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        // Second argument is false as we simulate the case where tools have not been provided in the request
        std::optional<rapidjson::Document> doc = outputParserWithRegularToolParsing->parseChunk(chunk, false, ov::genai::GenerationFinishReason::NONE);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;  // Both are nullopt, OK
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
        } else {
            FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk;
        }
    }
}
