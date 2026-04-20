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

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/output_parser.hpp"
#include "test/platform_utils.hpp"

using namespace ovms;

// DeepSeek tool call tags use:
//   U+FF5C  ｜  FULLWIDTH VERTICAL LINE
//   U+2581  ▁  LOWER ONE EIGHTH BLOCK
// These are multi-token text sequences, NOT single special tokens.

// Helper: construct the tag strings using UTF-8 byte sequences
static const std::string TAG_CALLS_BEGIN = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "calls\xE2\x96\x81" "begin\xEF\xBD\x9C>";
static const std::string TAG_CALL_BEGIN = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "call\xE2\x96\x81" "begin\xEF\xBD\x9C>";
static const std::string TAG_TOOL_SEP = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "sep\xEF\xBD\x9C>";
static const std::string TAG_CALL_END = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "call\xE2\x96\x81" "end\xEF\xBD\x9C>";
static const std::string TAG_CALLS_END = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "calls\xE2\x96\x81" "end\xEF\xBD\x9C>";

#ifdef _WIN32
const std::string deepseekTokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\deepseek-ai\\DeepSeek-R1-Distill-Qwen-7B";
#else
const std::string deepseekTokenizerPath = "/ovms/src/test/llm_testing/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B";
#endif

static const ToolsSchemas_t EMPTY_TOOLS_SCHEMA = {};
static std::unique_ptr<ov::genai::Tokenizer> deepseekTokenizer;

class DeepSeekOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParser;

    static void SetUpTestSuite() {
        try {
            deepseekTokenizer = std::make_unique<ov::genai::Tokenizer>(deepseekTokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize deepseek tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize deepseek tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        deepseekTokenizer.reset();
    }

    void SetUp() override {
        outputParser = std::make_unique<OutputParser>(*deepseekTokenizer, "deepseek", "", EMPTY_TOOLS_SCHEMA);
    }
};

// --- Unary parsing tests ---

TEST_F(DeepSeekOutputParserTest, ParseSingleToolCallWithCodeBlock) {
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n```json\n{\"location\": \"Paris\"}\n```" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\": \"Paris\"}");
    EXPECT_FALSE(parsedOutput.toolCalls[0].id.empty());
}

TEST_F(DeepSeekOutputParserTest, ParseSingleToolCallWithRawJSON) {
    // Model might produce raw JSON without code block wrapper
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n{\"location\": \"Paris\"}" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\": \"Paris\"}");
    EXPECT_FALSE(parsedOutput.toolCalls[0].id.empty());
}

TEST_F(DeepSeekOutputParserTest, ParseContentBeforeToolCall) {
    std::string input = "Let me check the weather for you." + TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n```json\n{\"city\": \"London\"}\n```" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "Let me check the weather for you.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"city\": \"London\"}");
}

TEST_F(DeepSeekOutputParserTest, ParseMultipleToolCalls) {
    std::string input = TAG_CALLS_BEGIN +
                         TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n```json\n{\"location\": \"Paris\"}\n```" + TAG_CALL_END + "\n" +
                         TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_time\n```json\n{\"timezone\": \"CET\"}\n```" + TAG_CALL_END +
                         TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\": \"Paris\"}");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "get_time");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"timezone\": \"CET\"}");
}

TEST_F(DeepSeekOutputParserTest, ParseNoToolCalls) {
    std::string input = "The weather in Paris is sunny and 22 degrees.";
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "The weather in Paris is sunny and 22 degrees.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(DeepSeekOutputParserTest, ParseToolCallWithEmptyArguments) {
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "list_tools\n```json\n{}\n```" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "list_tools");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{}");
}

TEST_F(DeepSeekOutputParserTest, ParseToolCallWithNoToolsInRequest) {
    // When tools are not in the request, the parser should not extract tool calls
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n```json\n{\"location\": \"Paris\"}\n```" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, false);
    EXPECT_EQ(parsedOutput.content, input);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(DeepSeekOutputParserTest, ParseToolCallWithMissingEndTag) {
    // Model might not produce closing tag
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "get_weather\n```json\n{\"location\": \"Paris\"}\n```";
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\": \"Paris\"}");
}

TEST_F(DeepSeekOutputParserTest, ParseToolCallWithComplexArguments) {
    std::string input = TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP + "search\n```json\n{\"query\": \"hello world\", \"filters\": [\"type:file\", \"lang:cpp\"], \"limit\": 10}\n```" + TAG_CALL_END + TAG_CALLS_END;
    auto generatedTensor = deepseekTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParser->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "search");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"query\": \"hello world\", \"filters\": [\"type:file\", \"lang:cpp\"], \"limit\": 10}");
}

// --- Streaming tests ---

TEST_F(DeepSeekOutputParserTest, StreamingSingleToolCall) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {TAG_CALLS_BEGIN, ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {TAG_CALL_BEGIN, ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"function", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {TAG_TOOL_SEP, ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get_weather\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        {"```json\n{\"location\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"Paris\"}\n```", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {TAG_CALL_END, ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\": \"Paris\"}"}}]}})"},
        {TAG_CALLS_END, ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            // Compare, ignoring generated IDs
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
                // Compare everything except the id value
                std::string docStrNoId = docStr;
                std::string expectedNoId = expected;
                docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk (ignoring id): " << chunk;
            } else {
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
            }
        } else if (expectedDelta.has_value()) {
            FAIL() << "Expected delta but got nullopt for chunk: " << chunk;
        } else if (doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            FAIL() << "Expected nullopt but got: " << buffer.GetString() << " for chunk: " << chunk;
        }
    }
}

TEST_F(DeepSeekOutputParserTest, StreamingContentThenToolCall) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"Sure, let me ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Sure, let me "}})"},
        {"check that.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"check that."}})"},
        {TAG_CALLS_BEGIN + TAG_CALL_BEGIN + "function" + TAG_TOOL_SEP, ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get_weather\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        {"```json\n{\"location\": \"Paris\"}\n```" + TAG_CALL_END + TAG_CALLS_END, ov::genai::GenerationFinishReason::STOP, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\": \"Paris\"}"}}]}})"},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
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
                EXPECT_EQ(docId.size(), expectedId.size());
                std::string docStrNoId = docStr;
                std::string expectedNoId = expected;
                docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk: " << chunk;
            } else {
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
            }
        } else if (expectedDelta.has_value()) {
            FAIL() << "Expected delta but got nullopt for chunk: " << chunk;
        } else if (doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            FAIL() << "Expected nullopt but got: " << buffer.GetString() << " for chunk: " << chunk;
        }
    }
}

TEST_F(DeepSeekOutputParserTest, StreamingNoToolCalls) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"The weather", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"The weather"}})"},
        {" is sunny.", ov::genai::GenerationFinishReason::STOP, R"({"delta":{"content":" is sunny."}})"},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        ASSERT_TRUE(expectedDelta.has_value() && doc.has_value()) << "Mismatch for chunk: " << chunk;
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc->Accept(writer);
        EXPECT_EQ(std::string(buffer.GetString()), expectedDelta.value()) << "Mismatch for chunk: " << chunk;
    }
}

TEST_F(DeepSeekOutputParserTest, StreamingToolCallWithoutToolsInRequest) {
    // When tools are not in the request, tool parser is skipped - everything is content
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        {TAG_CALLS_BEGIN, "{\"delta\":{\"content\":\"" + TAG_CALLS_BEGIN + "\"}}"},
        {TAG_CALL_BEGIN, "{\"delta\":{\"content\":\"" + TAG_CALL_BEGIN + "\"}}"},
        {"function", "{\"delta\":{\"content\":\"function\"}}"},
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, false, ov::genai::GenerationFinishReason::NONE);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        ASSERT_TRUE(expectedDelta.has_value() && doc.has_value()) << "Mismatch for chunk: " << chunk;
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc->Accept(writer);
        EXPECT_EQ(std::string(buffer.GetString()), expectedDelta.value()) << "Mismatch for chunk: " << chunk;
    }
}
