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
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\Qwen\\Qwen3-8B";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/Qwen/Qwen3-8B";
#endif

class Qwen3OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        // For Qwen3 model we use hermes3 tool parser (due to the same format of generated tool calls) and qwen3 reasoning parser
        outputParser = std::make_unique<OutputParser>(*tokenizer, "hermes3", "qwen3");
    }
};

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithSingleToolCallNoThinking) {
    std::string input = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithSingleToolCallAndThinking) {
    std::string input = "<think>Thinking about the tool call</think>"
                        "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "Thinking about the tool call");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithThreeToolCallsNoThinking) {
    std::string input = "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                        "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                        "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
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

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithThreeToolCallsAndThinking) {
    std::string input = "<think>Thinking about the tool calls</think>"
                        "<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>"
                        "<tool_call>{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}}</tool_call>"
                        "<tool_call>{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "Thinking about the tool calls");

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

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
}

TEST_F(Qwen3OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<tool_call>{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    auto generatedTensor = tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

// Major positive test for streaming tool calls with reasoning and multiple chunks and phase switching
TEST_F(Qwen3OutputParserTest, HolisticStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Thinking phase
        {"<think>", std::nullopt},
        {"Now ", "{\"delta\":{\"reasoning_content\":\"Now \"}}"},
        {"we are ", "{\"delta\":{\"reasoning_content\":\"we are \"}}"},
        {"thinking ", "{\"delta\":{\"reasoning_content\":\"thinking \"}}"},
        {"</think>", std::nullopt},
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"<tool_call>\n", std::nullopt},
        {"{\"", std::nullopt},
        {"name", std::nullopt},
        {"\":", std::nullopt},
        {" \"", std::nullopt},
        {"super", std::nullopt},
        {"_", std::nullopt},
        {"tool", std::nullopt},
        {"\",", std::nullopt},
        {" \"", std::nullopt},
        {"arguments", std::nullopt},
        // As we have 'arguments' key present, we can return first delta
        {"\":", "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"super_tool\"}}]}}"},
        // Consecutive deltas without 'id' and 'type'. In order to find the end of arguments parser has one chunk delay to handle end of tool.
        {" {", std::nullopt},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]}}"},
        {"arg1", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"arg1\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"value1", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\", ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"value1\"}}]}}"},
        {"arg2", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\", \"}}]}}"},
        {"\": ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"arg2\"}}]}}"},
        {"{\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_arg1", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"\"}}]}}"},
        {"\": ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_arg1\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_value1", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\", ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_value1\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\", \"}}]}}"},
        {"nested_arg2", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_arg2\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"nested_value2", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\"}}}", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"nested_value2\"}}]}}"},
        {"</tool_call>\n", "{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}}\"}}]}}"},
        // Starting second tool. Collecting chunk until full name is received. Don't return until then.
        {"<tool_call>\n", std::nullopt},
        {"{\"", std::nullopt},
        {"name", std::nullopt},
        {"\":", std::nullopt},
        {" \"", std::nullopt},
        {"super", std::nullopt},
        {"_tool", std::nullopt},
        {"_number", std::nullopt},
        {"_two", std::nullopt},
        {"\",", std::nullopt},
        {" \"", std::nullopt},
        {"arguments", std::nullopt},
        // As we have 'arguments' key present, we can return first delta
        {"\":", "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"super_tool_number_two\"}}]}}"},
        // Consecutive deltas without 'id' and 'type'. In order to find the end of arguments parser has one chunk delay to handle end of tool.
        {" {", std::nullopt},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\"}}]}}"},
        {"arg1", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\": ", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"arg1\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\": \"}}]}}"},
        {"val{{{ue1", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"\"", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"val{{{ue1\"}}]}}"},
        {"}", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"\"}}]}}"},
        {"}", "{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]}}"},  // returning last arguments part
        {"</tool_call>\n", std::nullopt},                                                          // closed main JSON, with the last chunk, now only return nullopt
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk);
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
            FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk;
        }
    }
}

TEST_F(Qwen3OutputParserTest, ToolCallsInsideReasoningStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Thinking phase
        {"<think>", std::nullopt},
        {"Now ", "{\"delta\":{\"reasoning_content\":\"Now \"}}"},
        {"we are ", "{\"delta\":{\"reasoning_content\":\"we are \"}}"},
        {"thinking ", "{\"delta\":{\"reasoning_content\":\"thinking \"}}"},
        // When tool call starts in a thinking phase we treat it as regular content
        {"<tool_call>\n", "{\"delta\":{\"reasoning_content\":\"<tool_call>\\n\"}}"},
        {"{\"", "{\"delta\":{\"reasoning_content\":\"{\\\"\"}}"},
        {"name", "{\"delta\":{\"reasoning_content\":\"name\"}}"},
        {"\":", "{\"delta\":{\"reasoning_content\":\"\\\":\"}}"},
        {" \"", "{\"delta\":{\"reasoning_content\":\" \\\"\"}}"},
        {"super", "{\"delta\":{\"reasoning_content\":\"super\"}}"},
        {"_tool", "{\"delta\":{\"reasoning_content\":\"_tool\"}}"},
        {"_number", "{\"delta\":{\"reasoning_content\":\"_number\"}}"},
        {"_two", "{\"delta\":{\"reasoning_content\":\"_two\"}}"},
        {"\",", "{\"delta\":{\"reasoning_content\":\"\\\",\"}}"},
        {" \"", "{\"delta\":{\"reasoning_content\":\" \\\"\"}}"},
        {"arguments", "{\"delta\":{\"reasoning_content\":\"arguments\"}}"},
        {"\":", "{\"delta\":{\"reasoning_content\":\"\\\":\"}}"},
        {" {", "{\"delta\":{\"reasoning_content\":\" {\"}}"},
        {"\"", "{\"delta\":{\"reasoning_content\":\"\\\"\"}}"},
        {"arg1", "{\"delta\":{\"reasoning_content\":\"arg1\"}}"},
        {"\": ", "{\"delta\":{\"reasoning_content\":\"\\\": \"}}"},
        {"\"", "{\"delta\":{\"reasoning_content\":\"\\\"\"}}"},
        {"val{{{ue1", "{\"delta\":{\"reasoning_content\":\"val{{{ue1\"}}"},
        {"\"", "{\"delta\":{\"reasoning_content\":\"\\\"\"}}"},
        {"}", "{\"delta\":{\"reasoning_content\":\"}\"}}"},
        {"}", "{\"delta\":{\"reasoning_content\":\"}\"}}"},
        {"</tool_call>\n", "{\"delta\":{\"reasoning_content\":\"</tool_call>\\n\"}}"},
        {"</think>", std::nullopt},
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk);
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

// Negative test cases

TEST_F(Qwen3OutputParserTest, ToolCallsBrokenJson) {
    std::vector<std::pair<std::string, bool>> chunkToErrorVec{
        {"<tool_call>\n", false},
        {"{\"", false},
        {"name", false},
        {"\"}", true},                    // With this chunk, JSON becomes invalid, we expect error.
        {", {", true},                    // The previous chunk already made JSON invalid, so we expect error.
        {"\"arg1\": \"value1\"}", true},  // Still invalid JSON, we expect error.
        {"</tool_call>\n", true},         // Closing invalid tool, throws due to the delay
        {"<tool_call>\n", false},         // Starting a new tool clears the state, so no error from this point
        {"{\"name\": ", false},
        {"\"another_tool\", ", false},
        {"\"arguments\": {\"param1\":", false},
        {"\"data\", \"param2\": true}}", false},
        {"</tool_call>\n", false},
    };
    for (const auto& [chunk, shouldThrow] : chunkToErrorVec) {
        if (shouldThrow) {
            EXPECT_THROW(outputParser->parseChunk(chunk), std::runtime_error) << "Expected error for chunk: " << chunk;
        } else {
            EXPECT_NO_THROW({
                auto doc = outputParser->parseChunk(chunk);
                // No further checks, just ensure no exception
            }) << "Unexpected error for chunk: "
               << chunk;
        }
    }
}

TEST_F(Qwen3OutputParserTest, ToolCallsDataAfterToolCall) {
    std::vector<std::pair<std::string, bool>> chunkToErrorVec{
        {"<tool_call>\n", false},
        {"{\"", false},
        {"name\": \"", false},
        {"some_tool\"", false},
        {", \"arguments\"", false},
        {": {\"arg1\": \"value1\"}", false},
        {"</tool_call>\n", false},
        {"<tool_call>\n", false},
        {"{\"name\": ", false},
        {"\"another_tool\", ", false},
        {"\"arguments\": {\"param1\":", false},
        {"\"data\", \"param2\": true}}", false},
        {"</tool_call>\n", false},
        {"This chunk added to JSON buffer breaks it, so we expect an error", true},
        {"Buffer is not cleared, JSON is still broken", true}};
    for (const auto& [chunk, shouldThrow] : chunkToErrorVec) {
        if (shouldThrow) {
            EXPECT_THROW(outputParser->parseChunk(chunk), std::runtime_error) << "Expected error for chunk: " << chunk;
        } else {
            EXPECT_NO_THROW({
                auto doc = outputParser->parseChunk(chunk);
                // No further checks, just ensure no exception
            }) << "Unexpected error for chunk: "
               << chunk;
        }
    }
}
