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
#include "../../../llm/io_processing/mistral/tool_parser.hpp"
#include "../../test_utils.hpp"

using namespace ovms;

class MistralOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<MistralToolParser> outputParser;
    ov::genai::Tokenizer dummy_tokenizer;  // not used

    void SetUp() override {
        outputParser = std::make_unique<MistralToolParser>(dummy_tokenizer);
    }
};

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    ParsedOutput parsedOutput;
    parsedOutput.content = "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]";
    outputParser->parse(parsedOutput, {});

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\":\"Paris\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    EXPECT_EQ(parsedOutput.content, "");                     // Content should be empty after parsing tool calls
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithMultipleToolCalls) {
    ParsedOutput parsedOutput;
    parsedOutput.content = "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}},{\"name\": \"get_humidity\", \"arguments\": {\"location\": \"Madrid\"}}]";
    outputParser->parse(parsedOutput, {});

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "get_humidity");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\":\"Paris\"}");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"location\":\"Madrid\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
    EXPECT_EQ(parsedOutput.content, "");                     // Content should be empty after parsing tool calls
}

TEST_F(MistralOutputParserTest, ParseToolCallOutputWithMultipleToolCallsWithWhitespaces) {
    ParsedOutput parsedOutput;
    parsedOutput.content = R"(
    [
      {"name": "get_weather", "arguments": {"location": "Paris"}},
      {"name": "get_humidity", "arguments": {"location": "Madrid"}}
    ])";
    outputParser->parse(parsedOutput, {});

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "get_humidity");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\":\"Paris\"}");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"location\":\"Madrid\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
    EXPECT_EQ(parsedOutput.content, "");                     // Content should be empty after parsing tool calls
}

TEST_F(MistralOutputParserTest, Negative_ParseToolCallOutput) {
    ParsedOutput parsedOutput;
    std::string content = R"(
    non whitespace [
      {"name": "get_weather", "arguments": {"location": "Paris"}},
      {"name": "get_humidity", "arguments": {"location": "Madrid"}}
    ])";
    parsedOutput.content = content;
    outputParser->parse(parsedOutput, {});

    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    ASSERT_EQ(parsedOutput.content, content);  // Content should remain unchanged
}
