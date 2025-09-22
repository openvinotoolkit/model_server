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
#include <regex>
#include <string>
#include <vector>

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/output_parser.hpp"
#include "src/llm/io_processing/qwen3coder/qwen3coder_tool_parser.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\Qwen\\Qwen3-8B";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/Qwen/Qwen3-8B";
#endif

using ovms::ParameterType_t;
using ovms::ToolsParameterTypeMap_t;
static std::unique_ptr<ov::genai::Tokenizer> qwen3Tokenizer;
static ovms::ToolsSchemas_t toolsSchemas = {
    {"string_tool", R"({"description": "A tool that takes a string argument.", "parameters": {"type": "object", "properties": {"arg1": {"type": "string", "description": "A string argument."}}, "required": ["arg1"]}})"}};  // can be empty for phi
static ToolsParameterTypeMap_t toolsParametersTypeMap = {
    {"string_tool", {{"arg1", ParameterType_t::STRING}}},
    {"string_int_tool", {{"arg1", ParameterType_t::STRING}, {"arg2", ParameterType_t::NUMBER}}},
    {"string_float_tool", {{"arg1", ParameterType_t::STRING}, {"arg2", ParameterType_t::NUMBER}}},
    {"string_int_float_tool", {{"arg1", ParameterType_t::STRING}, {"arg2", ParameterType_t::NUMBER}, {"arg3", ParameterType_t::NUMBER}}},
    {"object_tool", {{"param1", ParameterType_t::OBJECT}}},
    {"calculate_triangle_area", {{"base", ParameterType_t::NUMBER}, {"height", ParameterType_t::NUMBER}}},
};

class Qwen3CoderOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParser;

    static void SetUpTestSuite() {
        try {
            qwen3Tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize qwen3 tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize qwen3 tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        qwen3Tokenizer.reset();
    }

    void SetUp() override {
        // For Qwen3 model we use hermes3 tool parser (due to the same format of generated tool calls) and qwen3 reasoning parser
        outputParser = std::make_unique<OutputParser>(*qwen3Tokenizer, "qwen3coder", "");
    }
    std::tuple<ov::Tensor, std::vector<int64_t>, ParsedOutput> doTheWork(const std::string& input) {
        auto generatedTensor = qwen3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true, toolsSchemas);
        return {generatedTensor, generatedTokens, parsedOutput};
    }
};
// FIXME after streaming is done
// test \n no \n
// test argument for multiple lines
// check nested tools tags etc
// check nested function/arguments tags
// check non-finished tools calls
// check non-finished function
// check non-finished arguments
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCall1Function1ArgumentTagsNewline) {
    // std::string input = "<tool_call>{\"name\": \"string_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}</tool_call>";
    std::string input = R"(
"<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = doTheWork(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\": \"value1\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserJustToolCall) {
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\": \"value1\"}");
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserWithNoToolCall) {
    std::string input = R"(Unexpected void found. Philosophical crisis imminent.)";
    const std::string expectedContent = input;
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 0) << input;
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(expectedContent, content);
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserWithContent) {
    const std::string input = R"(
Before
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>
After)";
    const std::string expectedContent = "\nBefore\n\nAfter";

    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\": \"value1\"}");
    EXPECT_EQ(expectedContent, content);
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserWithThreeParameters) {
    const std::string input = R"(
<tool_call>
<function=string_int_float_tool>
<parameter=arg1>
value1
</parameter>
<parameter=arg2>
42
</parameter>
<parameter=arg3>
52.32
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_int_float_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\": \"value1\", \"arg2\": 42, \"arg3\": 52.32}");
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserWithJsonObjectArgument) {
    const std::string input = R"(
<tool_call>
<function=object_tool>
<parameter=arg1>
{"a": 1, "b": {"c": "asd"}}
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "object_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\": {\"a\": 1, \"b\": {\"c\": \"asd\"}}}");
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "\n");
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserWithTwoToolCalls) {
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>
<tool_call>
<function=string_float_tool>
<parameter=arg1>
data
</parameter>
<parameter=arg2>
25.2
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 2) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\": \"value1\"}");
    EXPECT_EQ(calls[1].name, "string_float_tool");
    EXPECT_EQ(calls[1].arguments, "{\"arg1\": \"data\", \"arg2\": 25.2}");
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "\n\n");
}

// write parametrized test code where I would just insert PARAM VALUE and EXPECTED ARGUMENTS
// to test various argument types
// int, float, string, json object, array of int, array of string, array of json object
class Qwen3CoderOutputParserParametrizedTest : public Qwen3CoderOutputParserTest, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string>> {
};

TEST_P(Qwen3CoderOutputParserParametrizedTest, TestJustParserWithVariousArgumentTypes) {
    const std::string& toolName = std::get<0>(GetParam());
    const std::string& argName = std::get<1>(GetParam());
    const std::string& paramValue = std::get<2>(GetParam());
    const std::string& expectedArguments = std::get<3>(GetParam());
    const std::string input = R"(
<tool_call>
<function=)" + toolName +
                              R"(>
<parameter=)" + argName + R"(>
)" + paramValue + R"(
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, toolName);
    EXPECT_EQ(calls[0].arguments, expectedArguments) << input;
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "\n");
}

const std::vector<std::tuple<std::string, std::string, std::string, std::string>> paramValueAndExpectedArgumentsVec = {
    {"string_tool", "arg1", "value1", "{\"arg1\": \"value1\"}"},
    {"int_tool", "arg1", "42", "{\"arg1\": 42}"},
    {"float_tool", "arg1", "52.32", "{\"arg1\": 52.32}"},
    {"bool_tool", "arg1", "true", "{\"arg1\": true}"},
    {"bool_tool", "arg1", "false", "{\"arg1\": false}"},
    {"bool_tool", "arg1", "True", "{\"arg1\": True}"},
    {"bool_tool", "arg1", "False", "{\"arg1\": False}"},
    {"object_tool", "arg1", "{\"a\": 1, \"b\": {\"c\": \"asd\"}}", "{\"arg1\": {\"a\": 1, \"b\": {\"c\": \"asd\"}}}"},
    {"list_tool", "arg1", "[1, 2, 3]", "{\"arg1\": [1, 2, 3]}"},
    {"list_tool", "arg1", "[\"a\", \"b\", \"c\"]", "{\"arg1\": [\"a\", \"b\", \"c\"]}"},
    {"object_tool", "arg1", "[{\"a\": 1}, {\"b\": 2}]", "{\"arg1\": [{\"a\": 1}, {\"b\": 2}]}"}};
// write printer for test name with first argument as a name

INSTANTIATE_TEST_SUITE_P(
    Qwen3CoderOutputParserParametrizedTestInstance,
    Qwen3CoderOutputParserParametrizedTest,
    ::testing::ValuesIn(paramValueAndExpectedArgumentsVec),
    [](const ::testing::TestParamInfo<Qwen3CoderOutputParserParametrizedTest::ParamType>& info) {
        std::string name = std::get<0>(info.param) + "_" + std::get<2>(info.param);
        // Replace non-alphanumeric characters with underscore
        std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
        // Limit length to 30 characters
        if (name.length() > 30) {
            name = name.substr(0, 30);
        }
        return name;
    });

TEST_F(Qwen3CoderOutputParserTest, JustToolParserBfclCalculateTriangle) {
    std::string input = R"(<tool_call>
<function=calculate_triangle_area>
<parameter=base>
10
</parameter>
<parameter=height>
5
</parameter>
</function>
</tool_call><|im_end|>)";
    auto content = input;
    ToolCalls calls;
    ovms::Parser parser(content, toolsParametersTypeMap);
    while (parser.step(calls)) {
    }
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "calculate_triangle_area");
    EXPECT_EQ(calls[0].arguments, "{\"base\":10, \"height\":5}");
    EXPECT_EQ(parser.currentState, ovms::Parser::State::End) << input;
    EXPECT_EQ(parser.currentPosition, std::string::npos) << input;
    EXPECT_EQ(content, "<|im_end|>") << input;
}

/*
// Major positive test for streaming tool calls with reasoning and multiple chunks and phase switching
TEST_F(Qwen3CoderOutputParserTest, HolisticStreaming) {
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
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE);
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

TEST_F(Qwen3CoderOutputParserTest, ToolCallsInsideReasoningStreaming) {
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
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE);
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

TEST_F(Qwen3CoderOutputParserTest, ToolCallsBrokenJson) {
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
            EXPECT_THROW(outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE), std::runtime_error) << "Expected error for chunk: " << chunk;
        } else {
            EXPECT_NO_THROW({
                auto doc = outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE);
                // No further checks, just ensure no exception
            }) << "Unexpected error for chunk: "
               << chunk;
        }
    }
}

TEST_F(Qwen3CoderOutputParserTest, ToolCallsDataAfterToolCall) {
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
            EXPECT_THROW(outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE), std::runtime_error) << "Expected error for chunk: " << chunk;
        } else {
            EXPECT_NO_THROW({
                auto doc = outputParser->parseChunk(chunk, true, ov::genai::GenerationFinishReason::NONE);
                // No further checks, just ensure no exception
            }) << "Unexpected error for chunk: "
               << chunk;
        }
    }
}
*/
