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
    {"string_tool", R"({"properties": {"arg1": {"type": "string", "description": "A string argument."}}, "required": ["arg1"]})"}};
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
        outputParser = std::make_unique<OutputParser>(*qwen3Tokenizer, "qwen3coder", "", toolsSchemas);
    }
    std::tuple<ov::Tensor, std::vector<int64_t>, ParsedOutput> doTheWork(const std::string& input) {
        auto generatedTensor = qwen3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
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

TEST_F(Qwen3CoderOutputParserTest, StreamingSimpleToolCall) {
    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<function=string_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"string_tool"}}]}})"},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<parameter=arg1", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"STRING_VALUE", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"arg1\": \"STRING_VALUE\"}"}}]}})"}  //,
                                                                                                                                                                        //        {"I finished", ov::genai::GenerationFinishReason::STOP, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" \"Paris\"}"}}]}})"});
    };
    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        i++;
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
                SPDLOG_ERROR("Expected:\n{}", expected);
                SPDLOG_ERROR("Got:\n{}", docStr);
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
            }
        } else {
            EXPECT_TRUE(false) << "Mismatch between expectedDelta and doc for id: " << i << " chunk:\n"
                               << chunk
                               << "\nexpectedDelta:\n"
                               << (expectedDelta.has_value() ? expectedDelta.value() : "EMPTY_DELTA")
                               << "\nGot doc:\n"
                               << (doc.has_value() ? /*convert doc to string*/ [&]() {
                                      rapidjson::StringBuffer buffer;
                                      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                                      doc->Accept(writer);
                                      return std::string(buffer.GetString());
                                  }()
                                                   : "NO_DOC");
            FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk;
        }
    }
}
