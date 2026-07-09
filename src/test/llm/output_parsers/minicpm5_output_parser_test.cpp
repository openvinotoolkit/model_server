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
#include "src/llm/io_processing/minicpm5/minicpm5_reasoning_parser.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string minicpm5TokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\openbmb\\MiniCPM5-1B";
#else
const std::string minicpm5TokenizerPath = "/ovms/src/test/llm_testing/openbmb/MiniCPM5-1B";
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
    {"dummy",         R"({"properties":{"config":{"type":"object"}},"required":["config"]})"},
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
        outputParser = std::make_unique<OutputParser>(*minicpm5Tokenizer, "minicpm5", "minicpm5", minicpm5ToolsSchemas);
    }

    std::vector<int64_t> encodeInput(const std::string& input) {
        if (input == "<think>") {
            return {8};  // <think> token ID
        }
        if (input == "</think>") {
            return {9};  // </think> token ID
        }
        auto generatedTensor = minicpm5Tokenizer->encode(input, ov::genai::add_special_tokens(true)).input_ids;
        return std::vector<int64_t>(
            generatedTensor.data<int64_t>(),
            generatedTensor.data<int64_t>() + generatedTensor.get_size());
    }

    ParsedOutput generateParsedOutput(const std::string& input) {
        auto generatedTokens = encodeInput(input);
        return outputParser->parse(generatedTokens, true);
    }

    void assertReasoningVec(const std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>>& chunkToDeltaVec) {
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
            std::vector<int64_t> tokens = {};
            if (chunk == "<think>") {
                tokens = {8};  // <think> token ID
            } else if (chunk == "</think>") {
                tokens = {9};  // </think> token ID
            } else {
                tokens = encodeInput(chunk);
            }
            std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, tokens, true, finishReason);
            if (!expectedDelta.has_value() && !doc.has_value()) {
                continue;  // Both are nullopt, OK
            }
            if (expectedDelta.has_value() && doc.has_value()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                doc->Accept(writer);
                std::string docStr = buffer.GetString();
                EXPECT_EQ(docStr, expectedDelta.value()) << "Mismatch for chunk: " << chunk;
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

    void assertStreamingVec(const std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>>& chunkToDeltaVec) {
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
            auto tokens = encodeInput(chunk);
            std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, tokens, true, finishReason);
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

TEST_F(Minicpm5OutputParserTest, ParseWithThinkBlockHandledByReasoningParser) {
    constexpr int64_t thinkStartTokenId = 8;  // <think>, per Minicpm5ReasoningParser
    constexpr int64_t thinkEndTokenId = 9;    // </think>, per Minicpm5ReasoningParser

    auto outputParserWithReasoning =
        std::make_unique<OutputParser>(*minicpm5Tokenizer, "minicpm5", "minicpm5", minicpm5ToolsSchemas);

    auto encode = [](ov::genai::Tokenizer& tok, const std::string& text) {
        auto tensor = tok.encode(text, ov::genai::add_special_tokens(false)).input_ids;
        return std::vector<int64_t>(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
    };

    std::vector<int64_t> generatedTokens;
    generatedTokens.push_back(thinkStartTokenId);
    auto reasoningTokens = encode(*minicpm5Tokenizer, "This is my internal reasoning about what to call.");
    generatedTokens.insert(generatedTokens.end(), reasoningTokens.begin(), reasoningTokens.end());
    generatedTokens.push_back(thinkEndTokenId);
    auto functionCallTokens = encode(*minicpm5Tokenizer, R"(<function name="search"><param name="query">Intel</param></function>)");
    generatedTokens.insert(generatedTokens.end(), functionCallTokens.begin(), functionCallTokens.end());

    ParsedOutput parsedOutput = outputParserWithReasoning->parse(generatedTokens, true);

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1u);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "search");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"query":"Intel"})");
    EXPECT_NE(parsedOutput.reasoning.find("internal reasoning"), std::string::npos);
    EXPECT_EQ(parsedOutput.content.find("<think>"), std::string::npos);
    EXPECT_EQ(parsedOutput.content, "");
}

// MiniCPM5's tool tags (<function>, <param>, ...) are special tokens, so the tool parser must
// request that special tokens are preserved in the stream; the paired reasoning parser agrees.
TEST_F(Minicpm5OutputParserTest, RequiresStreamingWithSpecialTokens) {
    Minicpm5ToolParser toolParser(*minicpm5Tokenizer, minicpm5ToolsSchemas);
    EXPECT_TRUE(toolParser.requiresStreamingWithSpecialTokens());
    Minicpm5ReasoningParser reasoningParser(*minicpm5Tokenizer);
    EXPECT_TRUE(reasoningParser.requiresStreamingWithSpecialTokens());
    // Pairing the two must not throw the consistency check in OutputParser.
    EXPECT_NO_THROW({
        OutputParser parser(*minicpm5Tokenizer, "minicpm5", "minicpm5", minicpm5ToolsSchemas);
        (void)parser;
    });
}

TEST_F(Minicpm5OutputParserTest, ParseNoToolCalls) {
    const std::string input = "Sure, here is the answer to your question.";
    ParsedOutput parsedOutput = generateParsedOutput(input);

    EXPECT_EQ(parsedOutput.toolCalls.size(), 0u);
    // Content should be preserved
    EXPECT_NE(parsedOutput.content.find("Sure"), std::string::npos);
}

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

TEST(Minicpm5ToolParserImplTest, ThinkBlockTreatedAsContent) {
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

// ---- Streaming robustness: fragment-boundary invariance of the underlying state machine ----
// These tests operate on the low-level Minicpm5ToolParserImpl and only assert that the
// AGGREGATED final result (completed call name + arguments) is the same no matter how the
// input is chopped into chunks -- i.e. that the state machine correctly reassembles calls
// regardless of where chunk boundaries fall. They do NOT test what a client actually receives
// mid-stream (when a name/arguments delta is surfaced, or whether it is surfaced exactly
// once at the right moment). That user-visible delta contract is covered separately by
// Minicpm5PublicStreamTest below, which drives the public Minicpm5ToolParser::parseChunk API
// and asserts on the actual emitted deltas.

// Helper: stream `input` in fixed-size fragments through one impl instance,
// collecting every tool call returned across all chunks (in order).
static ToolCalls_t streamInFragments(const std::string& input, size_t fragmentSize,
    const ToolsParameterTypeMap_t& typeMap) {
    Minicpm5ToolParserImpl parser(typeMap);
    ToolCalls_t collected;
    for (size_t i = 0; i < input.size(); i += fragmentSize) {
        std::string chunk = input.substr(i, fragmentSize);
        auto callsOpt = parser.parseChunk(chunk);
        if (callsOpt.has_value()) {
            for (auto& c : callsOpt.value()) {
                collected.push_back(c);
            }
        }
    }
    return collected;
}

// The streaming result (any fragmentation) MUST equal the single-chunk result.
TEST(Minicpm5ToolParserImplStreamTest, CharByCharSingleCall) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Beijing</param><param name="unit">celsius</param></function>)";
    auto calls = streamInFragments(input, 1, minicpm5TypeMap);  // one character per chunk
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"city":"Beijing","unit":"celsius"})");
}

// Tags, attributes and values are split across chunk boundaries at every offset.
TEST(Minicpm5ToolParserImplStreamTest, VariousFragmentSizesSingleCall) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Beijing</param></function>)";
    for (size_t frag = 1; frag <= input.size(); ++frag) {
        auto calls = streamInFragments(input, frag, minicpm5TypeMap);
        ASSERT_EQ(calls.size(), 1u) << "fragment size " << frag;
        EXPECT_EQ(calls[0].name, "get_weather") << "fragment size " << frag;
        EXPECT_EQ(calls[0].arguments, R"({"city":"Beijing"})") << "fragment size " << frag;
    }
}

// Two calls with the chunk boundary falling between/inside them.
TEST(Minicpm5ToolParserImplStreamTest, TwoCallsCharByChar) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Rome</param></function>)"
        R"(<function name="calculate"><param name="a">10</param><param name="b">20</param></function>)";
    auto calls = streamInFragments(input, 1, minicpm5TypeMap);
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"city":"Rome"})");
    EXPECT_EQ(calls[1].name, "calculate");
    EXPECT_EQ(calls[1].arguments, R"({"a":10,"b":20})");
}

// Leading content + <think> block streamed char-by-char before the call. The <think> text
// is just plain content to the tool parser (reasoning stripping happens upstream); the tool
// call must still be parsed correctly.
TEST(Minicpm5ToolParserImplStreamTest, ThinkBlockAndContentCharByChar) {
    const std::string input =
        R"(Let me check.<think>I should call the tool</think><function name="search"><param name="query">Intel</param></function>)";
    auto calls = streamInFragments(input, 1, minicpm5TypeMap);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "search");
    EXPECT_EQ(calls[0].arguments, R"({"query":"Intel"})");
}

// ---- Edge / malformed input: must not crash and must not emit spurious calls ----

// An unterminated function (no </function>) yields no completed tool call.
TEST(Minicpm5ToolParserImplTest, UnterminatedFunctionNoCall) {
    const std::string input =
        R"(<function name="get_weather"><param name="city">Berlin)";  // truncated mid-value
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    EXPECT_FALSE(callsOpt.has_value() && !callsOpt.value().empty());
}

// An empty parameter value produces an empty string argument (for a STRING-typed param).
TEST(Minicpm5ToolParserImplTest, EmptyParamValue) {
    const std::string input =
        R"(<function name="get_weather"><param name="city"></param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ASSERT_EQ(callsOpt.value().size(), 1u);
    EXPECT_EQ(callsOpt.value()[0].name, "get_weather");
    EXPECT_EQ(callsOpt.value()[0].arguments, R"({"city":""})");
}

// Plain text containing an angle bracket but no real function tag yields no calls and no crash.
TEST(Minicpm5ToolParserImplTest, AngleBracketInPlainTextNoCall) {
    const std::string input = "The value of a < b is true, and 3 < 4 as well.";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    EXPECT_FALSE(callsOpt.has_value() && !callsOpt.value().empty());
}

TEST_F(Minicpm5OutputParserTest, StreamingWithToolCallAndResoning) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"This", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":"This"}})"},
        {" is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" is"}})"},
        {" my", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" my"}})"},
        {" internal", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" internal"}})"},
        {" reasoning", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" reasoning"}})"},
        {" about", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" about"}})"},
        {" what", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" what"}})"},
        {" to", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" to"}})"},
        {" call.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" call."}})"},
        {"</think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"dummy\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"dummy"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"config\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"key\":\"value\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"config\":{\"key\":\"value\"}}"}}]}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Minicpm5OutputParserTest, StreamingWithToolCallReasoningAndContent) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"This", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":"This"}})"},
        {" is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" is"}})"},
        {" my", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" my"}})"},
        {" internal", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" internal"}})"},
        {" reasoning", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" reasoning"}})"},
        {" about", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" about"}})"},
        {" what", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" what"}})"},
        {" to", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" to"}})"},
        {" call.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" call."}})"},
        {"</think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"dummy\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"dummy"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"config\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"key\":\"value\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"config\":{\"key\":\"value\"}}"}}]}})"},
        {"SOME CONTENT,", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"IT SHOULDN'T APPREAR", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Minicpm5OutputParserTest, StreamingWithToolCallAndContent) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"SOME CONTENT", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"SOME CONTENT"}})"},
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"dummy\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"dummy"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"config\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"key\":\"value\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"config\":{\"key\":\"value\"}}"}}]}})"},
        {"CONTENT SOHULDN'T BE RETURNED AFTER TOOL CALL", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Minicpm5OutputParserTest, StreamingWithReasoningAndContent) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"This", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":"This"}})"},
        {" is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" is"}})"},
        {" my", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" my"}})"},
        {" internal", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" internal"}})"},
        {" reasoning", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" reasoning"}})"},
        {" about", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" about"}})"},
        {" what", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" what"}})"},
        {" to", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" to"}})"},
        {" call.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" call."}})"},
        {"</think>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Here", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Here"}})"},
        {" is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" is"}})"},
        {" some", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" some"}})"},
        {" content.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" content."}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Minicpm5OutputParserTest, StreamingWithContentOnly) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"Here", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Here"}})"},
        {" is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" is"}})"},
        {" some", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" some"}})"},
        {" content.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" content."}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Minicpm5OutputParserTest, StreamingWithMultipleToolCalls) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"dummy\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"dummy"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"config\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"key\":\"value\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"config\":{\"key\":\"value\"}}"}}]}})"},
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"get_weather\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"get_weather"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"city\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Tokyo", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<param", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"unit\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"celsius", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"city\":\"Tokyo\",\"unit\":\"celsius\"}"}}]}})"},
        {"<function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"search\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"><param", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"search"}}]}})"},
        {" name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"query\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {">", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"What does it mean \"if __name__ == '__main__':\n\tdict = { \"key\": \"value\" }\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</param>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</function>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\"query\":\"What does it mean \\\"if __name__ == '__main__':\\n\\tdict = { \\\"key\\\": \\\"value\\\" }\\\"\"}"}}]}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST(Minicpm5ToolParserImplTest, StringParamWithQuotesAndBackslashes) {
    const std::string input =
        R"(<function name="search"><param name="query">say "hi" and a backslash \ here</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "search");
    // arguments must be valid JSON
    rapidjson::Document argsDoc;
    argsDoc.Parse(calls[0].arguments.c_str());
    ASSERT_FALSE(argsDoc.HasParseError()) << "arguments not valid JSON: " << calls[0].arguments;
    ASSERT_TRUE(argsDoc.HasMember("query"));
    ASSERT_TRUE(argsDoc["query"].IsString());
    EXPECT_EQ(std::string(argsDoc["query"].GetString()), R"(say "hi" and a backslash \ here)");
}

// A code snippet with braces, quotes and newlines as a STRING-typed param value stays a string
// and round-trips to valid JSON.
TEST(Minicpm5ToolParserImplTest, StringParamWithCodeSnippet) {
    const std::string code =
        "def f(x):\n    return {\"a\": x, \"b\": [1, 2, 3]}  // comment";
    const std::string input =
        R"(<function name="search"><param name="query">)" + code + R"(</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    rapidjson::Document argsDoc;
    argsDoc.Parse(calls[0].arguments.c_str());
    ASSERT_FALSE(argsDoc.HasParseError()) << "arguments not valid JSON: " << calls[0].arguments;
    ASSERT_TRUE(argsDoc.HasMember("query"));
    ASSERT_TRUE(argsDoc["query"].IsString());
    // value preserved (newlines are interior, so not trimmed by trimNewline)
    EXPECT_EQ(std::string(argsDoc["query"].GetString()), code);
}

// A param value that itself looks like JSON, for a STRING-typed param, must be kept as a string
// (enforceStringValue), not parsed into a nested object.
TEST(Minicpm5ToolParserImplTest, StringParamThatLooksLikeJson) {
    const std::string input =
        R"(<function name="search"><param name="query">{"nested":true,"n":42}</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    rapidjson::Document argsDoc;
    argsDoc.Parse(calls[0].arguments.c_str());
    ASSERT_FALSE(argsDoc.HasParseError()) << "arguments not valid JSON: " << calls[0].arguments;
    ASSERT_TRUE(argsDoc.HasMember("query"));
    // STRING-typed param: kept as a string, not an object
    ASSERT_TRUE(argsDoc["query"].IsString());
    EXPECT_EQ(std::string(argsDoc["query"].GetString()), R"({"nested":true,"n":42})");
}

// Special characters (unicode, tabs, control-ish punctuation) in a string param round-trip
// to valid JSON.
TEST(Minicpm5ToolParserImplTest, StringParamWithSpecialChars) {
    const std::string value = "tab\tand emoji \xF0\x9F\x98\x80 and slash / and quote \" end";
    const std::string input =
        R"(<function name="search"><param name="query">)" + value + R"(</param></function>)";
    auto content = input;
    Minicpm5ToolParserImpl parser(minicpm5TypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    auto& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 1u);
    rapidjson::Document argsDoc;
    argsDoc.Parse(calls[0].arguments.c_str());
    ASSERT_FALSE(argsDoc.HasParseError()) << "arguments not valid JSON: " << calls[0].arguments;
    ASSERT_TRUE(argsDoc.HasMember("query"));
    ASSERT_TRUE(argsDoc["query"].IsString());
    EXPECT_EQ(std::string(argsDoc["query"].GetString()), value);
}
