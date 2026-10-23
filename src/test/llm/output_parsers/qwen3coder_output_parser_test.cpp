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
#include "output_parser_test_utils.hpp"
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
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*qwen3Tokenizer, *outputParser, generatedTokens, true, true);
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

// =============================================================================
// Recovery when generation stops before the tool call's closing tags ever arrive
// (max_tokens truncation, or the model just omits them). Regression tests for
// Qwen3CoderToolParserImpl::finalizeOnGenerationEnd().
// =============================================================================
TEST_F(Qwen3CoderOutputParserTest, ToolCallDroppedWhenStoppedMidParameterValue) {
    // STOP is not evidence that an open parameter value ended.
    const std::string input = "<tool_call>\\n<function=string_tool>\\n<parameter=arg1>val";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);
    EXPECT_TRUE(parsedOutput.toolCalls.empty());
}
TEST_F(Qwen3CoderOutputParserTest, ToolCallRecoveredWhenStoppedAfterInnerCloseTag) {
    // "</function>" seen but not the outer "</tool_call>".
    std::string input = "<tool_call>\n<function=string_tool>\n<parameter=arg1>value1</parameter>\n</function>\n";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"arg1":"value1"})");
}

TEST_F(Qwen3CoderOutputParserTest, ToolCallDroppedWithDanglingParameterName) {
    const std::string input = "<tool_call>\\n<function=string_tool>\\n<parameter=ar";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);
    EXPECT_TRUE(parsedOutput.toolCalls.empty());
}
TEST_F(Qwen3CoderOutputParserTest, ToolCallDroppedWhenStoppedBeforeFunctionName) {
    // Stops before the function name even closes -- no usable data was ever captured.
    std::string input = "<tool_call>\n<function=string_to";
    auto [generatedTensor, generatedTokens, parsedOutput] = generateParsedOutput(input);

    EXPECT_TRUE(parsedOutput.toolCalls.empty());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterLengthDoesNotPublishIncompleteArguments) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto first = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>partial",
        {}, ov::genai::GenerationFinishReason::LENGTH);
    // A name/id prefix is not executable without an arguments delta.
    if (first) {
        const auto* start = std::get_if<ToolCallDelta>(&*first);
        ASSERT_NE(start, nullptr);
        EXPECT_TRUE(start->arguments.empty());
    }
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterStopDoesNotPublishOpenParameter) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto first = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>partial",
        {}, ov::genai::GenerationFinishReason::STOP);
    if (first) {
        const auto* start = std::get_if<ToolCallDelta>(&*first);
        ASSERT_NE(start, nullptr);
        EXPECT_TRUE(start->arguments.empty());
    }
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
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

TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplUnterminatedToolCallContentCleanedNotError) {
    // Truncated mid-parameter-value, no closing tags at all: removeToolCallsFromContentIfNeeded()
    // must trim the dangling fragment instead of returning INTERNAL_ERROR (begin/end mismatch).
    std::string input = "<tool_call>\n<function=string_tool>\n<parameter=arg1>val";
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_FALSE(callsOpt.has_value());
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(content.find("<tool_call>"), std::string::npos) << content;
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
    // Without a matching parameter type, retain the parsed JSON value.
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
    // Duplicate XML arguments are ambiguous and invalidate the entire call.
    auto content = input;
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    EXPECT_FALSE(callsOpt.has_value());
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.find("</tool_call>") + std::string("</tool_call>").size());
    EXPECT_EQ(content, "\n");
}
// Regression tests for https://github.com/openvinotoolkit/model_server/issues/4487:
// the model sometimes quotes the XML attribute (<function="name">, <parameter="name">).
// The quotes are delimiters, so they must not end up in the tool name or in the argument
// keys, and the schema-driven type handling keyed on those names must keep working.
// The unquoted path is unchanged; TestJustParserImplUnaryToolCall above is its control.
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplQuotedParameterName) {
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter="arg1">
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
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplSingleQuotedParameterName) {
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter='arg1'>
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
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplQuotedFunctionName) {
    const std::string input = R"(
<tool_call>
<function="string_tool">
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
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplQuotedNamesKeepStringEnforcement) {
    // Both names are looked up in toolsParametersTypeMap, so quotes left in place would also
    // silently disable type handling: arg1 would stay a number instead of the declared string.
    const std::string input = R"(
<tool_call>
<function="string_int_tool">
<parameter="arg1">
42
</parameter>
<parameter="arg2">
7
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
    EXPECT_EQ(calls[0].name, "string_int_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"42\",\"arg2\":7}");
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplQuotedParameterNameKeepsBooleanNormalization) {
    const std::string input = R"(
<tool_call>
<function=bool_tool>
<parameter="arg1">
True
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
    EXPECT_EQ(calls[0].name, "bool_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":true}");
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplMixedQuotingOfParameterNames) {
    const std::string input = R"(
<tool_call>
<function=string_int_float_tool>
<parameter="arg1">
value1
</parameter>
<parameter=arg2>
42
</parameter>
<parameter='arg3'>
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
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "string_int_float_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42,\"arg3\":52.32}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplTwoToolCallsWithQuotedNames) {
    const std::string input = R"(
<tool_call>
<function="string_tool">
<parameter="arg1">
value1
</parameter>
</function>
</tool_call>
<tool_call>
<function='string_float_tool'>
<parameter='arg1'>
data
</parameter>
<parameter='arg2'>
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
    ASSERT_EQ(calls.size(), 2) << input;
    EXPECT_EQ(calls[0].name, "string_tool");
    EXPECT_EQ(calls[0].arguments, "{\"arg1\":\"value1\"}");
    EXPECT_EQ(calls[1].name, "string_float_tool");
    EXPECT_EQ(calls[1].arguments, "{\"arg1\":\"data\",\"arg2\":25.2}");
    EXPECT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::Content) << input;
    EXPECT_EQ(parser.getLastProcessedPosition(), input.rfind("</tool_call>") + std::string("</tool_call>").size()) << input;
    EXPECT_EQ(content, "\n\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplParameterNameWithInnerQuoteKept) {
    // Only a clean wrapping pair is a delimiter; a quote between the ends belongs to the name.
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter=arg"1>
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
    EXPECT_EQ(calls[0].arguments, "{\"arg\\\"1\":\"value1\"}");
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplParameterNameWithUnmatchedQuoteKept) {
    // Deliberately out of scope: a single unpaired quote is not a delimiter pair, and the
    // parser cannot tell a dropped closing quote from a name that starts with one.
    const std::string input = R"(
<tool_call>
<function=string_tool>
<parameter="arg1>
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
    EXPECT_EQ(calls[0].arguments, "{\"\\\"arg1\":\"value1\"}");
    EXPECT_EQ(content, "\n");
}
TEST_F(Qwen3CoderOutputParserTest, TestJustParserImplStreamStepWithQuotedFunctionName) {
    // getCurrentFunctionName() feeds the first streamed delta, so it must already be clean.
    ovms::Qwen3CoderToolParserImpl parser(toolsParametersTypeMap);
    const std::string input = R"(
<tool_call>
<function="string_tool">
<parameter="arg1">
value1
)";
    auto content = input;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
    ASSERT_EQ(parser.getCurrentState(), ovms::Qwen3CoderToolParserImpl::State::InsideParameter);
    ASSERT_TRUE(parser.getCurrentFunctionName().has_value());
    EXPECT_EQ(parser.getCurrentFunctionName().value(), "string_tool");
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

// Qwenmonster M1: a completed envelope must always emit name/id before arguments,
// even when both markers arrive in a single decode chunk.
TEST_F(Qwen3CoderOutputParserTest, QwenmonsterCompletedCallsInSingleChunk) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto first = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>value</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(first.has_value());
    const auto* start = std::get_if<ToolCallDelta>(&*first);
    ASSERT_NE(start, nullptr);
    EXPECT_EQ(start->index, 0);
    ASSERT_TRUE(start->name.has_value());
    EXPECT_EQ(*start->name, "string_tool");
    ASSERT_TRUE(start->id.has_value());
    EXPECT_FALSE(start->id->empty());
    EXPECT_TRUE(start->arguments.empty());

    auto second = parser.drainPendingDelta();
    ASSERT_TRUE(second.has_value());
    const auto* args = std::get_if<ToolCallDelta>(&*second);
    ASSERT_NE(args, nullptr);
    EXPECT_EQ(args->index, 0);
    EXPECT_FALSE(args->name.has_value());
    EXPECT_FALSE(args->id.has_value());
    EXPECT_EQ(args->arguments, R"({"arg1":"value"})");
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterMultipleCompletedCallsInSingleChunk) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto first = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>one</parameter></function></tool_call>"
        "<tool_call><function=cd><parameter=folder>two</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(first.has_value());
    std::vector<ToolCallDelta> deltas;
    if (auto* d = std::get_if<ToolCallDelta>(&*first)) deltas.push_back(*d);
    while (auto next = parser.drainPendingDelta()) {
        auto* d = std::get_if<ToolCallDelta>(&*next);
        ASSERT_NE(d, nullptr);
        deltas.push_back(*d);
        ASSERT_LE(deltas.size(), 4u);
    }
    ASSERT_EQ(deltas.size(), 4u);
    EXPECT_EQ(deltas[0].index, 0);
    EXPECT_EQ(deltas[1].index, 0);
    EXPECT_EQ(deltas[2].index, 1);
    EXPECT_EQ(deltas[3].index, 1);
    ASSERT_TRUE(deltas[0].name.has_value());
    ASSERT_TRUE(deltas[2].name.has_value());
    EXPECT_EQ(*deltas[0].name, "string_tool");
    EXPECT_EQ(*deltas[2].name, "cd");
    ASSERT_TRUE(deltas[0].id.has_value());
    ASSERT_TRUE(deltas[2].id.has_value());
    EXPECT_NE(*deltas[0].id, *deltas[2].id);
    EXPECT_FALSE(deltas[1].name.has_value());
    EXPECT_FALSE(deltas[3].name.has_value());
    EXPECT_EQ(deltas[1].arguments, R"({"arg1":"one"})");
    EXPECT_EQ(deltas[3].arguments, R"({"folder":"two"})");
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterCompletedCallFollowedByPartialName) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto first = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>one</parameter></function></tool_call>"
        "<tool_call><function=cd>", {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(first.has_value());
    auto arguments = parser.drainPendingDelta();
    ASSERT_TRUE(arguments.has_value());
    // An incomplete second call has no public identity yet.
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
    auto last = parser.parseChunk("<parameter=folder>two</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(std::get<ToolCallDelta>(*last).index, 1);
    EXPECT_EQ(std::get<ToolCallDelta>(*last).name, "cd");
    const auto body = parser.drainPendingDelta();
    ASSERT_TRUE(body.has_value());
    EXPECT_EQ(std::get<ToolCallDelta>(*body).index, 1);
    EXPECT_EQ(std::get<ToolCallDelta>(*body).arguments, R"({"folder":"two"})");
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterUnknownToolNotExecutable) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto output = parser.parseChunk(
        "<tool_call><function=undeclared_tool><parameter=key>1</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(output.has_value());
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterNonStringArgumentCannotFallbackToString) {
    // The fixture schema declares 'arg2' as integer, hence a textual fragment
    // that is not a JSON number cannot be exposed as executable arguments.
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto output = parser.parseChunk(
        "<tool_call><function=string_int_tool>"
        "<parameter=arg1>ok</parameter><parameter=arg2>oops</parameter>"
        "</function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    // An initial name may be visible, but there must be no completed arguments.
    if (output.has_value()) {
        const auto* delta = std::get_if<ToolCallDelta>(&*output);
        ASSERT_NE(delta, nullptr);
        EXPECT_TRUE(delta->arguments.empty());
    }
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterRequiredSchemaKeysMustBePresent) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto output = parser.parseChunk(
        "<tool_call><function=string_tool></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    // No executable arguments are emitted when required 'arg1' is absent.
    EXPECT_FALSE(output.has_value());
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterZeroArgumentRegisteredToolStillWorks) {
    const std::string schema = R"({"type":"object","properties":{},"required":[]})";
    rapidjson::Document schemaDoc;
    schemaDoc.Parse(schema.c_str());
    ToolsSchemas_t noArgSchemas;
    noArgSchemas.emplace("noarg_tool",
        ToolSchemaWrapper{&schemaDoc, schema});
    Qwen3CoderToolParser parser(*qwen3Tokenizer, noArgSchemas);
    const auto initial = parser.parseChunk(
        "<tool_call><function=noarg_tool></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(initial.has_value());
    ASSERT_TRUE(std::holds_alternative<ToolCallDelta>(*initial));
    EXPECT_EQ(*std::get<ToolCallDelta>(*initial).name, "noarg_tool");
    const auto args = parser.drainPendingDelta();
    ASSERT_TRUE(args.has_value());
    EXPECT_EQ(std::get<ToolCallDelta>(*args).arguments, "{}");
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterMalformedXMLClosingMarkerNeverExecutable) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto initial = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>abc</parameter1>",
        {}, ov::genai::GenerationFinishReason::STOP);
    if (initial) {
        const auto& delta = std::get<ToolCallDelta>(*initial);
        EXPECT_TRUE(delta.arguments.empty());
    }
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

// Qwenmonster independent workstream: the parser must reconstruct the same calls
// regardless of arbitrary boundaries introduced by token decoding / streaming.
// These tests are source-only additions; compile and execution are still pending.
TEST_F(Qwen3CoderOutputParserTest, QwenmonsterAllSingleByteSplitsPreserveCallIdentity) {
    const std::string wire =
        "<tool_call><function=string_tool><parameter=arg1>one</parameter></function></tool_call>"
        "<tool_call><function=cd><parameter=folder>two</parameter></function></tool_call>";
    for (size_t split = 1; split < wire.size(); ++split) {
        SCOPED_TRACE(split);
        Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
        std::vector<ToolCallDelta> deltas;
        const auto append = [&](const std::optional<Delta>& candidate) {
            if (!candidate) return;
            const auto* tool = std::get_if<ToolCallDelta>(&*candidate);
            EXPECT_NE(tool, nullptr);
            if (tool) deltas.push_back(*tool);
        };
        const auto feed = [&](const std::string& part) {
            append(parser.parseChunk(part, {}, ov::genai::GenerationFinishReason::NONE));
            size_t drained = 0;
            while (auto delta = parser.drainPendingDelta()) {
                append(delta);
                ++drained;
                if (drained > 8) {
                    ADD_FAILURE() << "Unbounded queued deltas";
                    break;
                }
            }
        };
        feed(wire.substr(0, split));
        feed(wire.substr(split));
        ASSERT_EQ(deltas.size(), 4u);
        EXPECT_EQ(deltas[0].index, 0);
        EXPECT_EQ(deltas[1].index, 0);
        EXPECT_EQ(deltas[2].index, 1);
        EXPECT_EQ(deltas[3].index, 1);
        ASSERT_TRUE(deltas[0].name.has_value());
        ASSERT_TRUE(deltas[2].name.has_value());
        ASSERT_TRUE(deltas[0].id.has_value());
        ASSERT_TRUE(deltas[2].id.has_value());
        EXPECT_EQ(*deltas[0].name, "string_tool");
        EXPECT_EQ(*deltas[2].name, "cd");
        EXPECT_FALSE(deltas[1].name.has_value());
        EXPECT_FALSE(deltas[3].name.has_value());
        EXPECT_FALSE(deltas[1].id.has_value());
        EXPECT_FALSE(deltas[3].id.has_value());
        EXPECT_NE(*deltas[0].id, *deltas[2].id);
        EXPECT_EQ(deltas[1].arguments, R"({"arg1":"one"})");
        EXPECT_EQ(deltas[3].arguments, R"({"folder":"two"})");
    }
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterResetClearsOldPartialAndRestartsCallIndex) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto partial = parser.parseChunk(
        "<tool_call><function=string_tool><parameter=arg1>unterminated",
        {}, ov::genai::GenerationFinishReason::LENGTH);
    // A speculative name may have streamed; it is not a completed executable call.
    if (partial) {
        ASSERT_TRUE(std::holds_alternative<ToolCallDelta>(*partial));
        EXPECT_TRUE(std::get<ToolCallDelta>(*partial).arguments.empty());
    }
    while (parser.drainPendingDelta()) {
    }
    parser.resetState();
    const auto start = parser.parseChunk(
        "<tool_call><function=cd><parameter=folder>fresh</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(start.has_value());
    const auto* head = std::get_if<ToolCallDelta>(&*start);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->index, 0);
    ASSERT_TRUE(head->name.has_value());
    EXPECT_EQ(*head->name, "cd");
    const auto args = parser.drainPendingDelta();
    ASSERT_TRUE(args.has_value());
    const auto* tail = std::get_if<ToolCallDelta>(&*args);
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->index, 0);
    EXPECT_EQ(tail->arguments, R"({"folder":"fresh"})");
    EXPECT_FALSE(parser.drainPendingDelta().has_value());
}

// OpenAI SSE cannot retract an orphan name. Publish identity only after the
// complete call passes validation; rejected prefixes consume no public index.
TEST_F(Qwen3CoderOutputParserTest, QwenmonsterRejectedPrefixMustNotStealNextCall) {
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    const auto rejectedHead = parser.parseChunk(
        "<tool_call><function=string_int_tool>",
        {}, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(rejectedHead.has_value());
    const auto rejectedBody = parser.parseChunk(
        "<parameter=arg1>ok</parameter><parameter=arg2>invalid-number</parameter>"
        "</function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(rejectedBody.has_value());
    EXPECT_FALSE(parser.drainPendingDelta().has_value());

    const auto acceptedHead = parser.parseChunk(
        "<tool_call><function=cd><parameter=folder>target</parameter></function></tool_call>",
        {}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(acceptedHead.has_value());
    const auto* head = std::get_if<ToolCallDelta>(&*acceptedHead);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->index, 0);
    ASSERT_TRUE(head->name.has_value());
    EXPECT_EQ(*head->name, "cd");
    const auto acceptedBody = parser.drainPendingDelta();
    ASSERT_TRUE(acceptedBody.has_value());
    EXPECT_EQ(std::get<ToolCallDelta>(*acceptedBody).index, 0);
    EXPECT_EQ(std::get<ToolCallDelta>(*acceptedBody).arguments, R"({"folder":"target"})");
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterPartitionCorpusReconstructsSSEIdentity) {
    const std::string first = "<tool_call><function=string_tool><parameter=arg1>one</parameter></function></tool_call>";
    const std::string second = "<tool_call><function=cd><parameter=folder>two</parameter></function></tool_call>";
    const std::string invalid = "<tool_call><function=string_int_tool><parameter=arg1>ok</parameter><parameter=arg2>bad</parameter></function></tool_call>";
    const std::string unknown = "<tool_call><function=undeclared></function></tool_call>";
    const std::string malformed = "<tool_call><function=string_tool><parameter=arg1>bad</parameter1></function></tool_call>";
    const std::string missingFunctionClose = "<tool_call><function=cd><parameter=folder>one</parameter></tool_call>";
    const std::string optionalWithoutFunctionClose = "<tool_call><function=some_tool></tool_call>";
    const std::string missingFunctionNameClose = "<tool_call><function=cd</tool_call>";
    const std::string missingParameterNameClose = "<tool_call><function=cd><parameter=folder</tool_call>";
    const std::string missingEnvelopeClose = "<tool_call><function=cd><parameter=folder>one</parameter></function>";
    const std::string unfinishedParameter = "<tool_call><function=string_tool><parameter=arg1>unfinished";
    const std::string partial = "<tool_call><function=cd><parameter=folder>unfinished";
    const std::string recoverable = "<tool_call><function=cd><parameter=folder>two</parameter></function>";
    struct Case {
        std::string wire;
        ov::genai::GenerationFinishReason finish;
        std::vector<std::pair<std::string, std::string>> expected;
    };
    using Reason = ov::genai::GenerationFinishReason;
    const std::vector<Case> cases = {
        {first + second + first, Reason::STOP, {{"string_tool", R"({"arg1":"one"})"}, {"cd", R"({"folder":"two"})"}, {"string_tool", R"({"arg1":"one"})"}}},
        {invalid + unknown + first + first, Reason::STOP, {{"string_tool", R"({"arg1":"one"})"}, {"string_tool", R"({"arg1":"one"})"}}},
        {malformed + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {missingFunctionClose + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {optionalWithoutFunctionClose + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {missingFunctionNameClose + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {missingParameterNameClose + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {"<tool_call></tool_call>" + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {missingEnvelopeClose + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {unfinishedParameter + second, Reason::STOP, {{"cd", R"({"folder":"two"})"}}},
        {first + partial, Reason::STOP, {{"string_tool", R"({"arg1":"one"})"}}},
        {first + partial, Reason::LENGTH, {{"string_tool", R"({"arg1":"one"})"}}},
        {first + recoverable, Reason::STOP, {{"string_tool", R"({"arg1":"one"})"}, {"cd", R"({"folder":"two"})"}}},
        {first + recoverable, Reason::LENGTH, {{"string_tool", R"({"arg1":"one"})"}}}};
    Qwen3CoderToolParser parser(*qwen3Tokenizer, toolsSchemas);
    size_t partitionRuns = 0;
    for (const auto& test : cases) {
        std::vector<std::vector<size_t>> partitions;
        for (const size_t width : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, test.wire.size()}) {
            std::vector<size_t> ends;
            for (size_t end = width; end < test.wire.size(); end += width)
                ends.push_back(end);
            ends.push_back(test.wire.size());
            partitions.push_back(std::move(ends));
        }
        // Every possible single split, including each side of every XML delimiter.
        for (size_t split = 1; split < test.wire.size(); ++split)
            partitions.push_back({split, test.wire.size()});
        // Simultaneously split before and after each delimiter character.
        std::vector<size_t> boundaries;
        for (size_t i = 1; i < test.wire.size(); ++i)
            if (test.wire[i] == '<' || test.wire[i] == '>' || test.wire[i - 1] == '<' || test.wire[i - 1] == '>')
                boundaries.push_back(i);
        boundaries.push_back(test.wire.size());
        partitions.push_back(std::move(boundaries));
        for (const auto& ends : partitions) {
            ++partitionRuns;
            SCOPED_TRACE(test.wire + " finish=" + std::to_string(static_cast<int>(test.finish)) + " chunks=" + std::to_string(ends.size()));
            parser.resetState();
            std::vector<ToolCall> calls;
            std::unordered_set<std::string> ids;
            size_t completed = 0;
            auto capture = [&](const Delta& delta) {
                // Reconstruct the actual OpenAI wire fields, not internal ToolCall objects.
                auto event = ovms::test::deltaToDocument(delta);
                const auto& wire = event["delta"]["tool_calls"][0];
                const auto index = wire["index"].GetInt();
                ASSERT_GE(index, 0);
                const auto& function = wire["function"];
                if (function.HasMember("name")) {
                    ASSERT_EQ(index, calls.size());
                    ASSERT_EQ(completed, calls.size()) << "Finish arguments before the next identity";
                    ASSERT_TRUE(wire.HasMember("id"));
                    const std::string id = wire["id"].GetString();
                    EXPECT_FALSE(id.empty());
                    EXPECT_TRUE(ids.insert(id).second);
                    calls.push_back({id, function["name"].GetString(), ""});
                } else {
                    ASSERT_LT(index, calls.size());
                    EXPECT_FALSE(wire.HasMember("id"));
                    calls[index].arguments += function["arguments"].GetString();
                    ++completed;
                }
            };
            size_t begin = 0;
            for (const size_t end : ends) {
                if (auto delta = parser.parseChunk(test.wire.substr(begin, end - begin), {}, Reason::NONE))
                    capture(*delta);
                while (auto delta = parser.drainPendingDelta())
                    capture(*delta);
                begin = end;
            }
            if (auto delta = parser.parseChunk("", {}, test.finish))
                capture(*delta);
            while (auto delta = parser.drainPendingDelta())
                capture(*delta);
            ASSERT_EQ(calls.size(), test.expected.size());
            EXPECT_EQ(completed, calls.size());
            for (size_t i = 0; i < calls.size(); ++i) {
                EXPECT_EQ(calls[i].name, test.expected[i].first);
                rapidjson::Document actual, expected;
                actual.Parse(calls[i].arguments.c_str());
                expected.Parse(test.expected[i].second.c_str());
                ASSERT_FALSE(actual.HasParseError());
                EXPECT_EQ(actual, expected);
            }
        }
    }
    RecordProperty("partition_cases", static_cast<int>(cases.size()));
    RecordProperty("partition_runs", static_cast<int>(partitionRuns));
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterStreamerPublishesCompletedArgumentsBeforeEnd) {
    std::vector<ToolCallDelta> calls;
    int lastCount = 0;
    auto parser = std::shared_ptr<OutputParser>(outputParser.get(), [](OutputParser*) {});
    OVMSTextStreamer streamer(*qwen3Tokenizer, parser, true,
        [&](Delta delta, bool last) {
            EXPECT_EQ(lastCount, 0);
            if (auto* call = std::get_if<ToolCallDelta>(&delta))
                calls.push_back(*call);
            lastCount += last ? 1 : 0;
            return ov::genai::StreamingStatus::RUNNING;
        }, {{ov::genai::skip_special_tokens.name(), false}});
    const auto tokens = qwen3Tokenizer->encode(
        "<tool_call><function=cd><parameter=folder>two</parameter></function></tool_call>\n",
        ov::genai::add_special_tokens(false)).input_ids;
    for (size_t i = 0; i < tokens.get_size(); ++i)
        ASSERT_EQ(streamer.write(tokens.data<int64_t>()[i]), ov::genai::StreamingStatus::RUNNING);
    ASSERT_EQ(calls.size(), 2) << "A validated call must reach the consumer before end()";
    EXPECT_EQ(calls[0].name, "cd");
    EXPECT_EQ(calls[0].index, calls[1].index);
    EXPECT_EQ(calls[1].arguments, R"({"folder":"two"})");
    EXPECT_EQ(lastCount, 0);
    streamer.end();
    EXPECT_EQ(calls.size(), 2);
    EXPECT_EQ(lastCount, 1);
    EXPECT_FALSE(streamer.hadParserError());
}

TEST_F(Qwen3CoderOutputParserTest, QwenmonsterSchemaValidationBeforePublication) {
    struct Case { std::string schema; std::string parameters; bool accepted; };
    const std::vector<Case> cases = {
        {R"({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]})", "<parameter=value>42</parameter>", true},
        {R"({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]})", "<parameter=value>42.5</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","enum":["ok"]}}})", "<parameter=value>bad</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","enum":["ok"]}}})", "<parameter=value>ok</parameter>", true},
        {R"({"type":"object","properties":{"value":{"type":"string"}},"additionalProperties":false})", "<parameter=other>x</parameter>", false},
        {R"({"type":"object","properties":{"value":{"anyOf":[{"type":"integer"},{"type":"null"}]}}})", "<parameter=value>null</parameter>", true},
        {R"({"type":"object","properties":{"value":{"anyOf":[{"type":"integer"},{"type":"null"}]}}})", "<parameter=value>bad</parameter>", false},
        {R"({"type":"object","properties":{"value":{"oneOf":[{"type":"integer"},{"type":"number"}]}}})", "<parameter=value>42</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":["string","null"]}}})", "<parameter=value>null</parameter>", true},
        {R"({"type":"object","properties":{"value":{"type":["string","null"]}}})", "<parameter=value>42</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"object","properties":{"items":{"type":"array","items":{"type":"integer"}}},"required":["items"],"additionalProperties":false}}})", "<parameter=value>{\"items\":[1,2]}</parameter>", true},
        {R"({"type":"object","properties":{"value":{"type":"object","properties":{"items":{"type":"array","items":{"type":"integer"}}},"required":["items"],"additionalProperties":false}}})", "<parameter=value>{\"items\":[1,\"wrong\"]}</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string"}},"required":["value"]})", "", false},
        {R"({"type":"object","properties":{"value":{"type":"string"}}})", "<parameter=value>first</parameter><parameter=value>second</parameter>", false},
        {R"({"type":"nonsense","properties":{"value":{"type":"string"}}})", "<parameter=value>x</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":7}}})", "<parameter=value>x</parameter>", false},
        {R"({"type":"object","properties":{"value":{"const":"ok"}}})", "<parameter=value>wrong</parameter>", false},
        {R"({"type":"object","definitions":{"integer":{"type":"integer"}},"properties":{"value":{"$ref":"#/definitions/integer"}}})", "<parameter=value>42</parameter>", true},
        {R"({"type":"object","definitions":{"integer":{"type":"integer"}},"properties":{"value":{"$ref":"#/definitions/integer"}}})", "<parameter=value>wrong</parameter>", false},
        {R"({"type":"object","properties":{"value":{"$ref":"https://example.invalid/schema"}}})", "<parameter=value>42</parameter>", false},
        {R"({"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema","properties":{"value":{}}})", "<parameter=value>42</parameter>", false},
        {"{", "<parameter=value>x</parameter>", false},
        {R"({"type":"object","default":{"const":"ok"},"properties":{"value":{"$ref":"#/default"}}})", "<parameter=value>wrong</parameter>", false},
        {R"({"type":"object","default":{"patternProperties":7}})", "<parameter=value>x</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","default":"fallback"}},"required":["value"]})", "", false},
        {R"({"type":"object","properties":{"value":{"type":"string","default":"fallback"}},"required":["value"]})", "<parameter=value>x</parameter>", true},
        {R"({"type":"object","properties":{"value":{"enum":[{"patternProperties":7}]}}})", "<parameter=value>{\"patternProperties\":7}</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","pattern":"^ok$\u0000never"}}})", "<parameter=value>ok</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","pattern":"^ok$"}}})", std::string("<parameter=value>ok") + '\0' + "bad</parameter>", false},
        {R"({"type":"object","dependencies":{"patternProperties":["x"]}})", "<parameter=value>x</parameter>", false},
        {R"({"type":"object","properties":{"default":{"type":"integer"}},"required":["default"]})", "<parameter=default>42</parameter>", true},
        {R"({"type":"object","properties":{"default":{"type":"integer"}},"required":["default"]})", "<parameter=default>bad</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"object","properties":{"x":{"type":"string","default":"fallback"}},"required":["x"]}}})", "<parameter=value>{}</parameter>", false},
        {R"({"type":"object","definitions":{"foo":{"properties":{"x":{"type":"integer"}}}},"properties":{"value":{"$ref":"#/definitions/foo%2Fproperties"}}})", "<parameter=value>wrong</parameter>", false},
        {R"({"type":"object","properties":{"value":{"type":"string","default":"a","default":"b"}},"required":["value"]})", "", false},
        {R"({"type":"object","properties":{"value":{"type":"object","properties":{"x":{"type":"integer"}}}}})", "<parameter=value>{\"x\":1,\"x\":2}</parameter>", false}};
    for (const auto& test : cases) {
        SCOPED_TRACE(test.schema + " " + test.parameters);
        rapidjson::Document schema;
        schema.Parse(test.schema.c_str());
        ToolsSchemas_t schemas;
        schemas.emplace("tool", ToolSchemaWrapper{&schema, test.schema});
        Qwen3CoderToolParser parser(*qwen3Tokenizer, schemas);
        const auto delta = parser.parseChunk("<tool_call><function=tool>" + test.parameters + "</function></tool_call>", {}, ov::genai::GenerationFinishReason::STOP);
        if (test.accepted) {
            ASSERT_TRUE(delta.has_value());
            EXPECT_EQ(std::get<ToolCallDelta>(*delta).name, "tool");
            ASSERT_TRUE(parser.drainPendingDelta().has_value());
        } else {
            EXPECT_FALSE(delta.has_value()) << "Invalid calls must publish no identity";
            EXPECT_FALSE(parser.drainPendingDelta().has_value());
        }
    }
}

class Qwen3CoderOutputParserParametrizedTest : public Qwen3CoderOutputParserTest, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string>> {
};

TEST_F(Qwen3CoderOutputParserTest, VariousArgumentTypesPreserveDeclaredSchemaBytes) {
    const std::vector<std::tuple<std::string, std::string, std::string>> cases = {
        {"number", "1.10", R"({"arg1":1.10})"},
        {"number", "1e3", R"({"arg1":1e3})"},
        {"integer", "9007199254740993", R"({"arg1":9007199254740993})"},
        {"array", "[1.10, 1e3, 9007199254740993]", R"({"arg1":[1.10,1e3,9007199254740993]})"},
        {"object", R"({"a":1.10, "b":[1e3,9007199254740993]})", R"({"arg1":{"a":1.10,"b":[1e3,9007199254740993]}})"},
        {"string", R"("C:\test")", R"({"arg1":"C:\\test"})"},
        {"string", R"(\\)", R"({"arg1":"\\\\"})"},
        {"string", "#B4E8", R"({"arg1":"#B4E8"})"},
        {"boolean", "True", R"({"arg1":true})"},
        {"boolean", "False", R"({"arg1":false})"},
        {"null", "null", R"({"arg1":null})"}};
    for (const auto& [type, input, expected] : cases) {
        SCOPED_TRACE(type + " " + input);
        const std::string schemaText = R"({"type":"object","properties":{"arg1":{"type":")" + type + R"("}},"required":["arg1"]})";
        rapidjson::Document schema;
        schema.Parse(schemaText.c_str());
        ToolsSchemas_t schemas{{"tool", ToolSchemaWrapper{&schema, schemaText}}};
        Qwen3CoderToolParser parser(*qwen3Tokenizer, schemas);
        const std::string envelope = "<tool_call><function=tool><parameter=arg1>" + input + "</parameter></function></tool_call>";
        std::string arguments;
        size_t headers = 0;
        auto collect = [&](const Delta& delta) {
            const auto& call = std::get<ToolCallDelta>(delta);
            if (call.name.has_value()) {
                EXPECT_EQ(*call.name, "tool");
                ++headers;
            }
            arguments += call.arguments;
        };
        for (const char byte : envelope) {
            if (auto delta = parser.parseChunk(std::string(1, byte), {}, ov::genai::GenerationFinishReason::NONE))
                collect(*delta);
            while (auto delta = parser.drainPendingDelta())
                collect(*delta);
        }
        EXPECT_EQ(headers, 1);
        EXPECT_EQ(arguments, expected);
    }
}

TEST_F(Qwen3CoderOutputParserTest, VariousArgumentTypesRejectInvalidDeclaredJson) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"number", "1e"}, {"number", "01"}, {"number", "1.10 trailing"},
        {"number", "\"1.10\""}, {"array", "[1.10,"}, {"array", "{}"},
        {"object", "{\"a\":1e}"}, {"object", "[]"}};
    for (const auto& [type, input] : cases) {
        SCOPED_TRACE(type + " " + input);
        const std::string schemaText = R"({"type":"object","properties":{"arg1":{"type":")" + type + R"("}}})";
        rapidjson::Document schema;
        schema.Parse(schemaText.c_str());
        ToolsSchemas_t schemas{{"tool", ToolSchemaWrapper{&schema, schemaText}}};
        Qwen3CoderToolParser parser(*qwen3Tokenizer, schemas);
        EXPECT_FALSE(parser.parseChunk("<tool_call><function=tool><parameter=arg1>" + input + "</parameter></function></tool_call>", {}, ov::genai::GenerationFinishReason::STOP).has_value());
        EXPECT_FALSE(parser.drainPendingDelta().has_value());
    }
}

TEST_F(Qwen3CoderOutputParserTest, VariousArgumentTypesClearExactMembersBetweenCalls) {
    const ToolsParameterTypeMap_t types = {
        {"number_tool", {{"arg1", ParameterType::NUMBER}}},
        {"string_tool", {{"arg1", ParameterType::STRING}}}};
    Qwen3CoderToolParserImpl parser(types);
    const auto first = parser.parseChunk("<tool_call><function=number_tool><parameter=arg1>1.10</parameter></function></tool_call>");
    ASSERT_TRUE(first.has_value());
    ASSERT_EQ(first->size(), 1);
    EXPECT_EQ(first->front().arguments, R"({"arg1":1.10})");
    const auto second = parser.parseChunk("<tool_call><function=string_tool><parameter=arg1>text</parameter></function></tool_call>");
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(second->size(), 1);
    EXPECT_EQ(second->front().arguments, R"({"arg1":"text"})");
}

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
    {"object_tool", "arg1", "[{\"a\":1},{\"b\":2}]", "{\"arg1\":[{\"a\":1},{\"b\":2}]}"},
    // A string parameter keeps the model's text as written, even when that text is valid JSON.
    {"string_tool", "arg1", "1.10", "{\"arg1\":\"1.10\"}"},
    {"string_tool", "arg1", "1e3", "{\"arg1\":\"1e3\"}"},
    {"string_tool", "arg1", "[1, 2,   3]", "{\"arg1\":\"[1, 2,   3]\"}"},
    {"string_tool", "arg1", "{\"a\": 1,  \"b\": 2}", "{\"arg1\":\"{\\\"a\\\": 1,  \\\"b\\\": 2}\"}"},
    // A quoted string is literal bytes. JSON would turn \t into a tab.
    {"string_tool", "arg1", "\"C:\\test\"", "{\"arg1\":\"C:\\\\test\"}"},
    // Number spelling is part of the value. A DOM round trip would rewrite these.
    {"float_tool", "arg1", "1.10", "{\"arg1\":1.10}"},
    {"float_tool", "arg1", "1e3", "{\"arg1\":1e3}"},
    {"int_tool", "arg1", "9007199254740993", "{\"arg1\":9007199254740993}"},
    {"object_tool", "arg1", "{\"a\":1.10}", "{\"arg1\":{\"a\":1.10}}"}};

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

TEST_F(Qwen3CoderOutputParserTest, StreamingQuotedFunctionAndParameterNames) {
    // End-to-end streaming counterpart of the parser-level quoted-attribute tests: the quotes
    // must not leak into the first delta (function name) nor into the argument keys.
    auto nameDelta = outputParser->parseChunk(R"(<tool_call><function="string_tool"><parameter="arg1">)", {}, true, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(nameDelta.has_value());
    nameDelta = outputParser->parseChunk("value1</parameter></function></tool_call>", {}, true, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(nameDelta.has_value());
    const std::string nameJson = ovms::test::deltaToJson(*nameDelta);
    EXPECT_NE(nameJson.find(R"("name":"string_tool")"), std::string::npos) << nameJson;

    auto argumentsDelta = outputParser->drainPendingDelta();
    ASSERT_TRUE(argumentsDelta.has_value());
    const std::string argumentsJson = ovms::test::deltaToJson(*argumentsDelta);
    EXPECT_NE(argumentsJson.find(R"("arguments":"{\"arg1\":\"value1\"}")"), std::string::npos) << argumentsJson;
}
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
    // Chunk boundaries must not dictate publication timing. Preserve the full
    // legacy payload corpus and compare its ordered events after validation.
    std::vector<ovms::Delta> events;
    std::vector<std::string> expectedEvents;
    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        if (expectedDelta)
            expectedEvents.push_back(*expectedDelta);
        if (auto event = outputParser->parseChunk(chunk, {}, true, finishReason))
            events.push_back(std::move(*event));
        while (auto event = outputParser->drainPendingDelta())
            events.push_back(std::move(*event));
    }
    ASSERT_EQ(events.size(), expectedEvents.size());
    for (size_t eventIndex = 0; eventIndex < events.size(); ++eventIndex) {
        const std::string chunk = "event " + std::to_string(eventIndex);
        const std::optional<std::string> expectedDelta = expectedEvents[eventIndex];
        i++;
        std::optional<ovms::Delta> doc = events[eventIndex];
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;  // Both are nullopt, OK
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            std::string docStr = ovms::test::deltaToJson(*doc);
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
                rapidjson::Document docJson = ovms::test::deltaToDocument(*doc);
                auto docJsonIt = docJson.FindMember("delta");
                ASSERT_NE(docJsonIt, docJson.MemberEnd());
                auto toolCallsIt = docJsonIt->value.FindMember("tool_calls");
                ASSERT_NE(toolCallsIt, docJsonIt->value.MemberEnd());
                for (const auto& toolCall : toolCallsIt->value.GetArray()) {
                    auto functionIt = toolCall.FindMember("function");
                    ASSERT_NE(functionIt, toolCall.MemberEnd());
                    auto argumentsIt = functionIt->value.FindMember("arguments");
                    ASSERT_NE(argumentsIt, functionIt->value.MemberEnd());
                    const std::string& argumentsStr = argumentsIt->value.GetString();
                    rapidjson::Document argsDoc;
                    argsDoc.Parse(argumentsStr.c_str());
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
                                      return ovms::test::deltaToJson(*doc);
                                  }()
                                                   : "NO_DOC");
            FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk;
        }
    }
}

// An open Qwen think stays reasoning until </think>. A draft <tool_call> inside
// the thought is not a tool call and does not end the thought. Real calls are
// only the <tool_call> blocks that follow the closer.
TEST_F(Qwen3CoderOutputParserTest, OpenThinkDraftToolCallRemainsReasoning) {
    OutputParser parser(*qwen3Tokenizer, "qwen3coder", "qwen3", toolsSchemas);
    parser.detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>");
    const std::string input =
        "Checking both places.\n"
        "<tool_call><function=string_tool>\n"
        "<parameter=arg1>draft\n"
        "Still thinking about both checks\n"
        "</think>\n\n"
        "<tool_call>\n"
        "<function=string_tool>\n"
        "<parameter=arg1>\n"
        "first\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=cd>\n"
        "<parameter=folder>\n"
        "/tmp\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    auto generatedTensor = qwen3Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(
        generatedTensor.data<int64_t>(),
        generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*qwen3Tokenizer, parser, generatedTokens, true, true);

    EXPECT_NE(parsedOutput.reasoning.find("<tool_call><function=string_tool>"), std::string::npos) << parsedOutput.reasoning;
    EXPECT_NE(parsedOutput.reasoning.find("<parameter=arg1>draft"), std::string::npos) << parsedOutput.reasoning;
    EXPECT_NE(parsedOutput.reasoning.find("Still thinking about both checks"), std::string::npos) << parsedOutput.reasoning;
    EXPECT_EQ(parsedOutput.reasoning.find("<function=cd>"), std::string::npos) << parsedOutput.reasoning;

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2u) << parsedOutput.reasoning;
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"arg1":"first"})");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "cd");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, R"({"folder":"/tmp"})");
    for (const auto& call : parsedOutput.toolCalls)
        EXPECT_EQ(call.arguments.find("draft"), std::string::npos) << call.arguments;
}
