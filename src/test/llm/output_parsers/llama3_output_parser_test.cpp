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
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\meta-llama\\Llama-3.1-8B-Instruct";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/meta-llama/Llama-3.1-8B-Instruct";
#endif

static ov::genai::Tokenizer llama3Tokenizer(tokenizerPath);

// Id of the <|python_tag|> which is a special token used to indicate the start of a tool calls
constexpr int64_t botTokenId = 128010;

class Llama3OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;
    std::unique_ptr<OutputParser> outputParserWithImmediateToolParsing;

    void SetUp() override {
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(llama3Tokenizer, "llama3", "");
        outputParserWithImmediateToolParsing = std::make_unique<OutputParser>(llama3Tokenizer, "llama3", "");
        outputParserWithImmediateToolParsing->enableImmediateToolParsing();
    }
};

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}}";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    generatedTokens.insert(generatedTokens.begin(), botTokenId);
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputNoToolsInTheRequest) {
    std::string input = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}}";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, false) : outputParserWithRegularToolParsing->parse(generatedTokens, false);
        EXPECT_EQ(parsedOutput.content, input);
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

//  Tool parser assumes entire output are tool calls since it starts with "{", but it's not the case
TEST_F(Llama3OutputParserTest, ParseRegularJsonOutputToolsInTheRequest) {
    std::string input = "{\"name\": \"Jane Doe\", \"location\": \"unknown\"}";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

// Tool parser is available, but there are no tools in the request, so all output should be treated as content
TEST_F(Llama3OutputParserTest, ParseRegularJsonOutputNoToolsInTheRequest) {
    std::string input = "{\"name\": \"Jane Doe\", \"location\": \"unknown\"}";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, false) : outputParserWithRegularToolParsing->parse(generatedTokens, false);
        EXPECT_EQ(parsedOutput.content, input);
        EXPECT_EQ(parsedOutput.reasoning, "");
    }
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}};"
                        "{\"name\": \"another_tool\", \"parameters\": {\"param1\": \"data\", \"param2\": true}};"
                        "{\"name\": \"third_tool\", \"parameters\": {\"key\": \"value\"}}";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;
        EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
        auto secondToolCallId = parsedOutput.toolCalls[1].id;
        EXPECT_NE(firstToolCallId, secondToolCallId);
        EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
        EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
        EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);
        auto thirdToolCallId = parsedOutput.toolCalls[2].id;
        EXPECT_NE(firstToolCallId, thirdToolCallId);
        EXPECT_NE(secondToolCallId, thirdToolCallId);
    }
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = llama3Tokenizer.encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, immediateParsing ? "" : "This is a regular model response without tool calls.");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
        EXPECT_EQ(parsedOutput.reasoning, "");
    }
}

TEST_F(Llama3OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string content = "This is a content part and next will be a tool call.";
    std::string toolCall = "{\"name\": \"example_tool\", \"parameters\": {\"arg1\": \"value1\", \"arg2\": 42}}";
    auto generatedContentTensor = llama3Tokenizer.encode(content, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedContentTokens(generatedContentTensor.data<int64_t>(), generatedContentTensor.data<int64_t>() + generatedContentTensor.get_size());
    auto generatedToolCallTensor = llama3Tokenizer.encode(toolCall, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedToolCallTokens(generatedToolCallTensor.data<int64_t>(), generatedToolCallTensor.data<int64_t>() + generatedToolCallTensor.get_size());
    std::vector<int64_t> generatedTokens;
    generatedTokens.insert(generatedTokens.end(), generatedContentTokens.begin(), generatedContentTokens.end());
    generatedTokens.insert(generatedTokens.end(), botTokenId);
    generatedTokens.insert(generatedTokens.end(), generatedToolCallTokens.begin(), generatedToolCallTokens.end());
    for (bool immediateParsing : {false, true}) {
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, immediateParsing ? "" : "This is a content part and next will be a tool call.");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), immediateParsing ? 0 : 1);
        if (!immediateParsing) {
            EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
            EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
            EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        }
    }
}

TEST_F(Llama3OutputParserTest, HolisticStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
{"<|python_tag|>", std::nullopt},
{"{\"", std::nullopt},
{"name", std::nullopt},
{"\":", std::nullopt},
{" \"", std::nullopt},
{"get", std::nullopt},
{"_h", std::nullopt},
{"umidity", std::nullopt},
{"\",", std::nullopt},
{" \"", std::nullopt},
{"parameters", std::nullopt},//10
{"\":", "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"get_humidity\"}}]}}"},
{" {\"", std::nullopt},
{"location", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}})"},
{"\":", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}})"},
{" \"", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":"}}]}})"},//15
{"Paris", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},//16
{"\"}}", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}})"},
{";", R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"},//18
{" {\"", std::nullopt},
{"name", std::nullopt},
{"\":", std::nullopt},
{" \"", std::nullopt},
{"get", std::nullopt},
{"_temperature", std::nullopt},
{"\",", std::nullopt},
{" \"", std::nullopt},
{"parameters", std::nullopt},
{"\":", R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"get_temperature"}}]}})"},//28
{" {\"", std::nullopt},//29
{"location", R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\""}}]}})"},
{"\":", R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"location"}}]}})"},
{" \"", R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\":"}}]}})"},
{"Paris\"}}", R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" \""}}]}})"},
                                              // closed main JSON, with the last chunk, now only return nullopt
    };

    int64_t i = -1;
    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        i++;
        std::optional<rapidjson::Document> doc = outputParserWithRegularToolParsing->parseChunk(chunk, true, static_cast<size_t>(i) >= chunkToDeltaVec.size() ? ov::genai::GenerationFinishReason::STOP : ov::genai::GenerationFinishReason::NONE);
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
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk: [" << chunk << "] got[" << docStr << "] but expected [" << expected << "]" << i;
            }
        } else if (expectedDelta.has_value()) {
            FAIL() << "Mismatch between expectedDelta and doc for chunk: [" << chunk << "] got nothing but expected [" << expectedDelta.value() << "]" << i;
        } else if (doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            FAIL() << "Mismatch between expectedDelta and doc for chunk: [" << chunk << "] expected nothing but got [" << docStr << "]" << i;
        } else {
            FAIL() << "Mismatch between expectedDelta and doc for chunk: [" << chunk << "] but expected nothing" << i;
        }
    }
}
