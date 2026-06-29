//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/output_parser.hpp"
#include "src/llm/io_processing/minicpm5/minicpm5_tool_parser.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
// Reuse an existing tokenizer from the test assets (content doesn't matter for these unary tests)
const std::string minicpm5TokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\Qwen\\Qwen3-8B";
#else
const std::string minicpm5TokenizerPath = "/ovms/src/test/llm_testing/Qwen/Qwen3-8B";
#endif

using ovms::ParameterType;
using ovms::ToolsParameterTypeMap_t;

static std::unique_ptr<ov::genai::Tokenizer> minicpm5Tokenizer;

// Schemas used to drive type-aware parameter insertion
static std::map<std::string, std::string> minicpm5ToolSchemasInput = {
    {"get_weather",   R"({"properties":{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"]})"},
    {"calculate",     R"({"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]})"},
    {"search",        R"({"properties":{"query":{"type":"string"}},"required":["query"]})"},
    {"get_stock",     R"({"properties":{"ticker":{"type":"string"},"count":{"type":"integer"}},"required":["ticker"]})"},
};

static std::vector<std::unique_ptr<rapidjson::Document>> minicpm5SchemaDocsStorage;

static ToolsSchemas_t convertToToolsSchemas(const std::map<std::string, std::string>& input) {
    ToolsSchemas_t result;
    minicpm5SchemaDocsStorage.clear();
    for (const auto& [name, schemaStr] : input) {
        auto schemaDoc = std::make_unique<rapidjson::Document>();
        if (schemaDoc->Parse(schemaStr.c_str()).HasParseError()) {
            throw std::runtime_error("Failed to parse schema for tool: " + name);
        }
        result[name] = {schemaDoc.get(), schemaStr};
        minicpm5SchemaDocsStorage.push_back(std::move(schemaDoc));
    }
    return result;
}

static ovms::ToolsSchemas_t minicpm5ToolsSchemas = convertToToolsSchemas(minicpm5ToolSchemasInput);

// Type map used when testing Minicpm5ToolParserImpl directly (without going through OutputParser)
static ToolsParameterTypeMap_t minicpm5TypeMap = {
    {"get_weather", {{"city", ParameterType::STRING}, {"unit", ParameterType::STRING}}},
    {"calculate",   {{"a", ParameterType::NUMBER},    {"b", ParameterType::NUMBER}}},
    {"search",      {{"query", ParameterType::STRING}}},
    {"get_stock",   {{"ticker", ParameterType::STRING}, {"count", ParameterType::NUMBER}}},
};

// ---- Test fixture ----

class Minicpm5OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParser;

    static void SetUpTestSuite() {
        try {
            minicpm5Tokenizer = std::make_unique<ov::genai::Tokenizer>(minicpm5TokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        minicpm5Tokenizer.reset();
    }

    void SetUp() override {
        outputParser = std::make_unique<OutputParser>(*minicpm5Tokenizer, "minicpm5", "", minicpm5ToolsSchemas);
    }

    ParsedOutput generateParsedOutput(const std::string& input) {
        auto generatedTensor = minicpm5Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(
            generatedTensor.data<int64_t>(),
            generatedTensor.data<int64_t>() + generatedTensor.get_size());
        return outputParser->parse(generatedTokens, true);
    }
};

// ---- Unary parse tests ----

// Single function call, two params, basic string types
TEST_F(Minicpm5OutputParserTest, ParseSingleFunctionCallTwoParams) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Beijing</param><param name="unit">celsius</param></function>)";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"city":"Beijing","unit":"celsius"})");
    EXPECT_FALSE(parsedOutput.toolCalls[0].id.empty());
}

// Multiple function calls concatenated
TEST_F(Minicpm5OutputParserTest, ParseMultipleFunctionCalls) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Tokyo</param></function>)"
        R"(<function name="search"><param name="query">OpenVINO</param></function>)";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"city":"Tokyo"})");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "search");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, R"({"query":"OpenVINO"})");
}

// Param typed as integer in schema → should appear as number in JSON, not string
TEST_F(Minicpm5OutputParserTest, ParseIntegerTypedParam) {
    const std::string input =
        R"(<function name="calculate"><param name="a">7</param><param name="b">3</param></function>)";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "calculate");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"a":7,"b":3})");
}

// Mixed string + integer schema types in one call
TEST_F(Minicpm5OutputParserTest, ParseMixedStringAndIntegerParams) {
    const std::string input =
        R"(<function name="get_stock"><param name="ticker">INTC</param><param name="count">42</param></function>)";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_stock");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"ticker":"INTC","count":42})");
}

// <think>...</think> blocks must be stripped before parsing
TEST_F(Minicpm5OutputParserTest, ParseWithThinkBlockStripped) {
    const std::string input =
        "<think>This is my internal reasoning about what to call.</think>"
        R"(<function name="search"><param name="query">Intel</param></function>)";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "search");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"query":"Intel"})");
}

// Plain content with no tool calls must pass through unchanged
TEST_F(Minicpm5OutputParserTest, ParseNoToolCalls) {
    const std::string input = "Sure, here is the answer to your question.";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    EXPECT_EQ(parsedOutput.toolCalls.size(), 0u);
    // Content should be preserved
    EXPECT_NE(parsedOutput.content.find("Sure"), std::string::npos);
}

// Content before and after a tool call — tool call removed from content
TEST_F(Minicpm5OutputParserTest, ParseContentAroundToolCall) {
    const std::string input =
        "Let me check the weather. "
        R"(<function name="get_weather"><param name="city">Paris</param></function>)"
        " Done.";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"city":"Paris"})");
    // Tool call XML should have been removed from content
    EXPECT_EQ(parsedOutput.content.find("<function"), std::string::npos);
    EXPECT_EQ(parsedOutput.content.find("</function>"), std::string::npos);
}

// ---- Tests on the impl directly (without tokenizer round-trip) ----

TEST(Minicpm5ToolParserImplTest, SingleCallNoContent) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Berlin</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"city":"Berlin"})");
    EXPECT_EQ(parser.getCurrentState(), Minicpm5ToolParserImpl::State::Content);
    EXPECT_FALSE(calls[0].id.empty());
}

TEST(Minicpm5ToolParserImplTest, TwoCalls) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Rome</param></function>)"
        R"(<function name="calculate"><param name="a">10</param><param name="b">20</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"city":"Rome"})");
    EXPECT_EQ(calls[1].name, "calculate");
    EXPECT_EQ(calls[1].arguments, R"({"a":10,"b":20})");
}

TEST(Minicpm5ToolParserImplTest, ThinkBlockStripped) {
    const std::string input =
        "<think>Reasoning here.</think>"
        R"(<function name="search"><param name="query">test</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "search");
    EXPECT_EQ(calls[0].arguments, R"({"query":"test"})");
}

TEST(Minicpm5ToolParserImplTest, IntegerParam) {
    const std::string input =
        R"(<function name="calculate"><param name="a">5</param><param name="b">15</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].arguments, R"({"a":5,"b":15})");
}

TEST(Minicpm5ToolParserImplTest, NoToolCalls) {
    const std::string input = "Just some content without any function calls.";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    EXPECT_FALSE(callsOpt.has_value());
    EXPECT_EQ(parser.getCurrentState(), Minicpm5ToolParserImpl::State::Content);
    EXPECT_EQ(parser.getLastProcessedPosition(), 0u);
}

TEST(Minicpm5ToolParserImplTest, NewlinesAroundParamValue) {
    // Model may emit values with surrounding newlines; they should be trimmed
    const std::string input =
        "<function name=\"get_weather\"><param name=\"city\">\nBejing\n</param></function>";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    EXPECT_EQ(callsOpt.value()[0].arguments, R"({"city":"Bejing"})");
}

TEST(Minicpm5ToolParserImplTest, RemoveToolCallsFromContent) {
    const std::string input =
        "Before. "
        R"(<function name="search"><param name="query">intel</param></function>)"
        " After.";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(content.find("<function"), std::string::npos);
    EXPECT_EQ(content.find("</function>"), std::string::npos);
}

TEST(Minicpm5ToolParserImplTest, SingleQuoteAttribute) {
    // Defensive: accept name='...' as well as name="..."
    const std::string input =
        "<function name='get_weather'><param name='city'>Sydney</param></function>";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"city":"Sydney"})");
}
