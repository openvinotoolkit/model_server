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

using ovms::ParameterType;
using ovms::ToolsParameterTypeMap_t;
static std::unique_ptr<ov::genai::Tokenizer> qwen3Tokenizer;

static std::map<std::string, std::string> toolSchemasInput = {
    {"string_tool", R"({"properties": {"arg1": {"type": "string", "description": "A string argument."}}, "required": ["arg1"]})"},
    {"cd", R"({"properties": {"folder": {"type": "string", "description": "Path"}}, "required": ["folder"]})"},
    {"string_int_tool", R"({"properties":{"arg1":{"type":"string","description":"A string argument."},"arg2":{"type":"integer","description":"An integer argument."}},"required":["arg1", "arg2"]})"},
    {"some_tool", R"({"properties":{"source":{"type":"string","description":"The name of the file or directory to copy."},"destination":{"type":"string","description":"The destination name to copy the file or directory to. If the destination is a directory, the source will be copied into this directory. No file paths allowed. "}},"required":[]})"},
    {"stringx7_tool", R"({"properties": {"arg1":{"type":"string","description":"Not used"},"arg2":{"type":"string","description":"Not used"},"arg3":{"type":"string","description":"Not used"},"arg4":{"type":"string","description":"Not used"},"arg5":{"type":"string","description":"Not used"},"arg6":{"type":"string","description":"Not used"}, "arg7":{"type":"string","description":"A string argument."}},"required": []})"}};

static std::vector<std::unique_ptr<rapidjson::Document>> schemaDocsStorage;

static ToolsSchemas_t convertStringToolSchemasStringToToolsSchemas(
    const std::map<std::string, std::string>& input) {
    ToolsSchemas_t result;
    schemaDocsStorage.clear();
    for (const auto& [name, schemaStr] : input) {
        auto schemaDoc = std::make_unique<rapidjson::Document>();
        if (schemaDoc->Parse(schemaStr.c_str()).HasParseError()) {
            throw std::runtime_error("Failed to parse schema for tool: " + name);
        }
        result[name] = {schemaDoc.get(), schemaStr};
        schemaDocsStorage.push_back(std::move(schemaDoc));
    }
    return result;
}

static ovms::ToolsSchemas_t toolsSchemas = convertStringToolSchemasStringToToolsSchemas(toolSchemasInput);
static ToolsParameterTypeMap_t toolsParametersTypeMap = {
    {"string_tool", {{"arg1", ParameterType::STRING}}},
    {"bool_tool", {{"arg1", ParameterType::BOOLEAN}}},
    {"string_string_tool", {{"arg1", ParameterType::STRING}, {"arg2", ParameterType::STRING}}},
    {"string_int_tool", {{"arg1", ParameterType::STRING}, {"arg2", ParameterType::NUMBER}}},
    {"string_float_tool", {{"arg1", ParameterType::STRING}, {"arg2", ParameterType::NUMBER}}},
    {"string_int_float_tool", {{"arg1", ParameterType::STRING}, {"arg2", ParameterType::NUMBER}, {"arg3", ParameterType::NUMBER}}},
    {"object_tool", {{"param1", ParameterType::OBJECT}}},
    {"calculate_triangle_area", {{"base", ParameterType::NUMBER}, {"height", ParameterType::NUMBER}}},
    {"stringx7_tool", {{"arg1", ParameterType::STRING},
                          {"arg2", ParameterType::STRING},
                          {"arg3", ParameterType::STRING},
                          {"arg4", ParameterType::STRING},
                          {"arg5", ParameterType::STRING},
                          {"arg6", ParameterType::STRING},
                          {"arg7", ParameterType::STRING}}}};

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
        outputParser = std::make_unique<OutputParser>(*qwen3Tokenizer, "qwen3coder", "", toolsSchemas);
    }
    std::tuple<ov::Tensor, std::vector<int64_t>, ParsedOutput> generateParsedOutput(const std::string& input) {
        auto generatedTensor = qwen3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
        return {generatedTensor, generatedTokens, parsedOutput};
    }
};
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCall1Function1ArgumentTagsNewline) {
    std::string input = R"(
"<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    // Qwen3CoderToolParserImpl removes newlines, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCall1Function1ArgumentNoProperBeginTag) {
    std::string input = R"(
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    // Qwen3CoderToolParserImpl removes newlines, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCallNestedXmlNotFromSchema) {
    std::string input = R"(
"<tool_call>
<function=string_tool>
<parameter=arg1>
<value=abc>value1</value>
</parameter>
</function>
</tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"<value=abc>value1</value>\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, ParseTwoToolCalls1Function1ArgumentTagsNoNewline) {
    std::string input = R"(
"<tool_call><function=string_tool><parameter=arg1>value1</parameter></function></tool_call>"
"<tool_call><function=string_tool><parameter=arg1>value2</parameter></function></tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"arg1\":\"value2\"}");
    EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCall1Function1ArgumentTagsNoNewline) {
    std::string input = R"(
"<tool_call><function=string_tool><parameter=arg1>value1</parameter></function></tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, Parse1ToolCall1Function1ArgumentMultilineValue) {
    std::string input = R"(
"<tool_call>
<function=string_tool>
<parameter=arg1>
value1line1
value1line2
</parameter>
</function>
</tool_call>")";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1line1\\nvalue1line2\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryToolCall) {
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithNoToolCall) {
    std::string input = R"(Unexpected void found. Philosophical crisis imminent.)";
    const std::string expectedContent = input;
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_FALSE(callsOpt.has_value());
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), 0) << input;
    EXPECT_EQ(expectedContent, content);
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithContent) {
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
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(expectedContent, content);
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithThreeParameters) {
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
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_int_float_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42,\"arg3\":52.32}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithEnforcementOfStringParameter) {
    // arg1, arg2, ..., arg7: bool, int, uint, int64, uint64, double, null
    // tool schema defines types as strings but the value of string could be number etc
    const std::string input = R"(
<tool_call>
<function=stringx7_tool>
<parameter=arg1>
true
</parameter>
<parameter=arg2>
-13
</parameter>
<parameter=arg3>
42
</parameter>
<parameter=arg4>
-12345678901234
</parameter>
<parameter=arg5>
12345678901234
</parameter>
<parameter=arg6>
3.14159
</parameter>
<parameter=arg7>
null
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "stringx7_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"true\",\"arg2\":\"-13\",\"arg3\":\"42\",\"arg4\":\"-12345678901234\",\"arg5\":\"12345678901234\",\"arg6\":\"3.14159\",\"arg7\":\"null\"}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithNotPresentToolSchema) {
    // in this case everything will be writtem as parsed type
    // arg1, arg2, ..., arg8: bool, int, uint, int64, uint64, double, null, string
    const std::string input = R"(
<tool_call>
<function=unrecognized_tool>
<parameter=arg1>
true
</parameter>
<parameter=arg2>
-13
</parameter>
<parameter=arg3>
42
</parameter>
<parameter=arg4>
-12345678901234
</parameter>
<parameter=arg5>
12345678901234
</parameter>
<parameter=arg6>
3.14159
</parameter>
<parameter=arg7>
null
</parameter>
<parameter=arg8>
SomeStringHere
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "unrecognized_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":true,\"arg2\":-13,\"arg3\":42,\"arg4\":-12345678901234,\"arg5\":12345678901234,\"arg6\":3.14159,\"arg7\":null,\"arg8\":\"SomeStringHere\"}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n");
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithJsonObjectArgument) {
    const std::string input = R"(
<tool_call>
<function=object_tool>
<parameter=arg1>
{"a": 1, "b": {"c": "asd"}}
</parameter>
</function>
</tool_call>)";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "object_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":{\"a\":1,\"b\":{\"c\":\"asd\"}}}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n");
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryWithTwoToolCalls) {
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
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 2) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(calls[1].name, "string_float_tool");
    EXPECT_EQ(calls[1].arguments, "{\"arg1\":\"data\",\"arg2\":25.2}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.rfind("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryToolCallNoMatchingToolParameterTypeMapEntry) {
    const std::string input = R"(
<tool_call>
<function=SOME_UNRECOGNIZED_TOOL>
<parameter=arg1>
value1
</parameter>
<parameter=arg2>
True
</parameter>
<parameter=arg3>
25.12
</parameter>
</function>
</tool_call>)";
    // in this case since we don't have tool in parameter type map we
    // keep values as is - it won't for non-nested bool, string
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "SOME_UNRECOGNIZED_TOOL");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":\"True\",\"arg3\":25.12}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnaryToolCallWithRepeatedArgument) {
    const std::string input = R"(
<tool_call>
<function=string_string_tool>
<parameter=arg1>
value1
</parameter>
<parameter=arg1>
value2
</parameter>
</function>
</tool_call>)";
    // in this case since we don't have tool in parameter type map we
    // keep values as is - it won't for non-nested bool, string
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithMoreThan1StateChange) {
    ToolCalls_t calls;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>)";
    auto content = input;
    auto stepResult = parser.parseChunk(content);
    ASSERT_TRUE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content);
    auto& resultToolCalls = stepResult.value();
    ASSERT_EQ(resultToolCalls.size(), 1);
    EXPECT_EQ(resultToolCalls[0].name, "string_tool");
    EXPECT_EQ(resultToolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(content, input);
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithNoStateChange) {
    ToolCalls_t calls;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    const std::string input = R"("Some content without tool calls")";
    auto content = input;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content);
    EXPECT_EQ(parser.getLastProcessedPosition(), 0);
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithPartialToolCall) {
    ToolCalls_t calls;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
)";
    auto content = input;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::InsideParameter);
    EXPECT_EQ(content, input);
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find(Qwen3CoderToolParser::PARAMETER_NAME_TAG) + std::string("<parameter=arg1>").size());
    EXPECT_EQ(parser.getCurrentFunctionName().value(), "string_tool");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithTwoToolCalls) {
    ToolCalls_t calls;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>
Some content between
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
    auto stepResult = parser.parseChunk(content);
    ASSERT_TRUE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content);
    auto& resultToolCalls = stepResult.value();
    ASSERT_EQ(resultToolCalls.size(), 2);
    EXPECT_EQ(resultToolCalls[0].name, "string_tool");
    EXPECT_EQ(resultToolCalls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(resultToolCalls[1].name, "string_float_tool");
    EXPECT_EQ(resultToolCalls[1].arguments, "{\"arg1\":\"data\",\"arg2\":25.2}");
    EXPECT_EQ(content, input);
    EXPECT_EQ(parser.getLastProcessedPosition(), input.rfind("</tool_call>") + std::string("</tool_call>").size());
}

TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithToolCallNoArgs) {
    const std::string input = R"(
<tool_call>
<function=noarg_tool>
</function>
</tool_call>)";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto stepResult = parser.parseChunk(content);
    ASSERT_TRUE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content);
    auto& calls = stepResult.value();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "noarg_tool");
    EXPECT_EQ(calls[0].arguments, "{}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
}

class Qwen3CoderOutputParserParametrizedTest : public Qwen3CoderOutputParserTest, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string>> {
};

TEST_P(Qwen3CoderOutputParserParametrizedTest, TestJustParserImplWithVariousArgumentTypes) {
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
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, toolName);
    EXPECT_EQ(calls[0].arguments, expectedArguments) << input;
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n");
}

const std::vector<std::tuple<std::string, std::string, std::string, std::string>> paramValueAndExpectedArgumentsVec = {
    {"string_tool", "arg1", "value1", "{\"arg1\":\"value1\"}"},
    {"int_tool", "arg1", "42", "{\"arg1\":42}"},
    {"float_tool", "arg1", "52.32", "{\"arg1\":52.32}"},
    {"bool_tool", "arg1", "true", "{\"arg1\":true}"},
    {"bool_tool", "arg1", "false", "{\"arg1\":false}"},
    {"bool_tool", "arg1", "True", "{\"arg1\":true}"},
    {"bool_tool", "arg1", "False", "{\"arg1\":false}"},
    {"object_tool", "arg1", "{\"a\":1,\"b\":{\"c\":\"asd\"}}", "{\"arg1\":{\"a\":1,\"b\":{\"c\":\"asd\"}}}"},
    {"list_tool", "arg1", "[1, 2, 3]", "{\"arg1\":[1,2,3]}"},
    {"list_tool", "arg1", "[\"a\",\"b\",\"c\"]", "{\"arg1\":[\"a\",\"b\",\"c\"]}"},
    {"object_tool", "arg1", "[{\"a\":1},{\"b\":2}]", "{\"arg1\":[{\"a\":1},{\"b\":2}]}"}};

INSTANTIATE_TEST_SUITE_P(
    Qwen3CoderOutputParserParametrizedTestInstance,
    Qwen3CoderOutputParserParametrizedTest,
    ::testing::ValuesIn(paramValueAndExpectedArgumentsVec),
    [](const ::testing::TestParamInfo<Qwen3CoderOutputParserParametrizedTest::ParamType>& info) {
        std::string name = std::get<0>(info.param) + "_" + std::get<2>(info.param);
        // Replace non-alphanumeric characters with underscore
        std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
        return name;
    });

TEST_F(Qwen3CoderOutputParserTest, StreamingSimpleToolCall) {
    // since unary reuses streaming we don't need to test for partial tool calls
    // if we don't get closing tag we don't emit tool call
    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // now we test functool improperly beginning with <function=... and then being finished by </tool_call>
        // its important that this is before any <tool_call> tag
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"<function=string_tool><parameter=arg1>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"string_tool"}}]}})"},
        {"value_before_tool_call</parameter></function></tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"arg1\":\"value_before_tool_call\"}"}}]}})"},
        // now we test normal tool call
        {" <too", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"l_cal", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"l>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctio", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"n=st", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ring_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"string_tool"}}]}})"},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<paramete", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"r=a", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"rg1", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"STRI", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"NG_VALUE", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</pa", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"rameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"arg1\":\"STRING_VALUE\"}"}}]}})"},
        {" POTENTIALLY EXISINT CONTENT", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" <tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" <tool_call>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<function=string_int_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"string_int_tool"}}]}})"},
        {"<parameter=arg1>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_STRING_VALUE\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<parameter=arg2>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"314", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"1522\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\"arg1\":\"\\nANOTHER_STRING_VALUE\",\"arg2\":3141522}"}}]}})"},
        {"CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // now we test functool improperly beginning with <function=... and then being finished by </tool_call>
        {"<function=string_tool><parameter=arg1>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":3,"function":{"name":"string_tool"}}]}})"},
        {"value1</parameter></function></tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":3,"function":{"arguments":"{\"arg1\":\"value1\"}"}}]}})"},
        {"NOTHING IMPORTANT HERE", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"part of bfcl 'draft'.\n\n<function=cd>\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":4,"function":{"name":"cd"}}]}})"},
        {"\n<parameter=folder>\nResearchDocs\n</parameter>\n</function>\n</tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":4,"function":{"arguments":"{\"folder\":\"ResearchDocs\"}"}}]}})"},
        // example from cds:
        {R"(
<tool_call>
<function=string_tool>
<parameter=arg1>
)",
            ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":5,"function":{"name":"string_tool"}}]}})"},
        {R"(FUNCTION FC_CreateJsonPayload : STRING
VAR_INPUT
    Value1 : REAL;
    Value2 : INT;
    Value3 : BOOL;
    Value4 : STRING(100);
END_VAR
VAR_OUTPUT
    JsonPayload : STRING(1000);
END_VAR
VAR
    TempStr : STRING(100);
END_VAR

    JsonPayload := '{';
    JsonPayload := JsonPayload + '"value1":' + REAL_TO_STRING(Value1, '', 2) + ',';
    JsonPayload := JsonPayload + '"value2":' + INT_TO_STRING(Value2) + ',';
    JsonPayload := JsonPayload + '"value3":' + BOOL_TO_STRING(Value3) + ',';
    JsonPayload := JsonPayload + '"value4":"' + Value4 + '"';
    JsonPayload := JsonPayload + '}';

END_FUNCTION</parameter>
</function>
</tool_call>)",
            ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":5,"function":{"arguments":"{\"arg1\":\"FUNCTION FC_CreateJsonPayload : STRING\\nVAR_INPUT\\n    Value1 : REAL;\\n    Value2 : INT;\\n    Value3 : BOOL;\\n    Value4 : STRING(100);\\nEND_VAR\\nVAR_OUTPUT\\n    JsonPayload : STRING(1000);\\nEND_VAR\\nVAR\\n    TempStr : STRING(100);\\nEND_VAR\\n\\n    JsonPayload := '{';\\n    JsonPayload := JsonPayload + '\\\"value1\\\":' + REAL_TO_STRING(Value1, '', 2) + ',';\\n    JsonPayload := JsonPayload + '\\\"value2\\\":' + INT_TO_STRING(Value2) + ',';\\n    JsonPayload := JsonPayload + '\\\"value3\\\":' + BOOL_TO_STRING(Value3) + ',';\\n    JsonPayload := JsonPayload + '\\\"value4\\\":\\\"' + Value4 + '\\\"';\\n    JsonPayload := JsonPayload + '}';\\n\\nEND_FUNCTION\"}"}}]}})"},
        {"<tool_call><function=string_tool><parameter=arg1>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":6,"function":{"name":"string_tool"}}]}})"},
        {R"(
if __name__ == "__main__":
    addresses = {}
    addresses["Hodor"] = """The door"""
    addresses["Arya"] = "Winterfell"
    for name, address in addresses.items():
        print(f'\n\t{name} lives at {address}\n\r')
</parameter></function></tool_call>)",
            ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":6,"function":{"arguments":"{\"arg1\":\"if __name__ == \\\"__main__\\\":\\n    addresses = {}\\n    addresses[\\\"Hodor\\\"] = \\\"\\\"\\\"The door\\\"\\\"\\\"\\n    addresses[\\\"Arya\\\"] = \\\"Winterfell\\\"\\n    for name, address in addresses.items():\\n        print(f'\\\\n\\\\t{name} lives at {address}\\\\n\\\\r')\"}"}}]}})"}};
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
                std::string docStrNoId = docStr;
                std::string expectedNoId = expected;
                docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk (ignoring id value): " << chunk;
            } else {
                SPDLOG_ERROR("Expected:\n{}", expected);
                SPDLOG_ERROR("Got:\n{}", docStr);
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
                // now we do final check
                // we want to ensure that if we extract from expectedDelta["delta"]["tool_calls"][0]["function"]["arguments"] which as a string
                // and then try to read this string as a json
                if (expectedDelta.value().find("arguments") == std::string::npos) {
                    SPDLOG_TRACE("No arguments to check for delta:\n{}", expectedDelta.value());
                    continue;  // no arguments to check
                }
                auto docJsonIt = doc->FindMember("delta");
                ASSERT_NE(docJsonIt, doc->MemberEnd());
                auto toolCallsIt = docJsonIt->value.FindMember("tool_calls");
                ASSERT_NE(toolCallsIt, docJsonIt->value.MemberEnd());
                for (const auto& toolCall : toolCallsIt->value.GetArray()) {
                    auto functionIt = toolCall.FindMember("function");
                    ASSERT_NE(functionIt, toolCall.MemberEnd());
                    auto argumentsIt = functionIt->value.FindMember("arguments");
                    ASSERT_NE(argumentsIt, functionIt->value.MemberEnd());
                    const std::string& argumentsStr = argumentsIt->value.GetString();
                    rapidjson::Document argsDoc;
                    argsDoc.Parse(argumentsStr.c_str());  // now check for errors
                    EXPECT_FALSE(argsDoc.HasParseError()) << "Arguments is not valid JSON for chunk: " << chunk << "\nArguments string:\n"
                                                          << argumentsStr;
                }
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
