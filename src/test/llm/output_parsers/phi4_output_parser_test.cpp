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
#include "../../platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\microsoft\\Phi-4-mini-instruct";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/microsoft/Phi-4-mini-instruct";
#endif

static ovms::ToolsSchemas_t EMPTY_TOOLS_SCHEMA = {};  // not used for phi4
static std::unique_ptr<ov::genai::Tokenizer> phi4Tokenizer;

class Phi4OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;
    std::unique_ptr<OutputParser> outputParserWithImmediateToolParsing;

    static void SetUpTestSuite() {
        try {
            phi4Tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize phi tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize phi tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        phi4Tokenizer.reset();
    }

    void SetUp() override {
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*phi4Tokenizer, "phi4", "", EMPTY_TOOLS_SCHEMA);
        outputParserWithImmediateToolParsing = std::make_unique<OutputParser>(*phi4Tokenizer, "phi4", "", EMPTY_TOOLS_SCHEMA);
        outputParserWithImmediateToolParsing->enableImmediateToolParsing();
    }
};

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "functools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("functools").length());
        }
        auto generatedTensor = phi4Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string input = "functools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"arguments\": {\"key\": \"value\"}}]";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("functools").length());
        }
        auto generatedTensor = phi4Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
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

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithOneValidToolCallAndTwoInvalid) {
    std::string input = "functools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}},"
                        "{\"tool_name\": \"another_tool\", \"arguments\": {\"param1\": \"data\", \"param2\": true}},"
                        "{\"name\": \"third_tool\", \"options\": {\"key\": \"value\"}}]";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("functools").length());
        }
        auto generatedTensor = phi4Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;
    }
}

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    for (bool immediateParsing : {false, true}) {
        auto generatedTensor = phi4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, immediateParsing ? "" : "This is a regular model response without tool calls.");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
        EXPECT_EQ(parsedOutput.reasoning, "");
    }
}

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\nfunctools[{\"name\": \"example_tool\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": 42}}]";
    for (bool immediateParsing : {false, true}) {
        auto generatedTensor = phi4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, immediateParsing ? "" : "This is a content part and next will be a tool call.\n\n");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), immediateParsing ? 0 : 1);
        if (!immediateParsing) {
            EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
            EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
            EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        }
    }
}
TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithMultipleFunctoolsReturnsNothing) {
    std::string input = "functools[{\"name\": \"tool1\", \"arguments\": {\"a\": 1}}]\n\nThis is some content\n\nfunctools[{\"name\": \"tool2\", \"arguments\": {\"b\": 2}}]";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput.erase(std::string("functools").length());
        }
        auto generatedTensor = phi4Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

TEST_F(Phi4OutputParserTest, ParseToolCallOutputWithArrayArguments) {
    std::string input = "functools[{\"name\": \"extractLastTransactionId\", \"arguments\": { \"filepath\": \"/var/log/db.log\", \"status\": [\"completed\", \"failed\"], \"encoding\": \"utf-8\", \"processFunction\": \"processFunction\"}}]";
    for (bool immediateParsing : {false, true}) {
        std::string testInput = input;
        if (immediateParsing) {
            testInput = testInput.substr(std::string("functools").length());
        }
        auto generatedTensor = phi4Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = immediateParsing ? outputParserWithImmediateToolParsing->parse(generatedTokens, true) : outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "extractLastTransactionId");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Phi4OutputParserTest, HolisticStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ct", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ools", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[{", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_h", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"umidity", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"arguments\":", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"get_humidity\"}}]}}"},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}})"},
        {"locations", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"locations"}}]}})"},
        {"\": {\"real_cities\": ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\": {\"real_cities\": "}}]}})"},
        {" [\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" [\""}}]}})"},
        {"Paris\", \"New", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris\", \"New"}}]}})"},
        {"York\"], \"fictional_cities\": [\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"York\"], \"fictional_cities\": [\""}}]}})"},
        {"Cintra\", \"Oxenfurt\"]}}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Cintra\", \"Oxenfurt\"]}}"}}]}})"},
        {"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},

        {" {\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_temperature", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Simulate getting arguments key, value and close of tool call all in one chunk
        {"arguments\": {}},", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"get_temperature"}}]}})"},
        // Such chunk is broken into parts before and after colon, so along with the next chunk we also process ' {}},' part

        // At this point we process ' {}}, {\"' part, but since it's both end and start of tool call, we split it again.
        // So in that call we process ' {}}' part and push ', {\"' part to the next call.
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{}"}}]}})"},
        // At this point we process ', {\"name' which can be processed as a whole, no more delay from that point
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_temperature", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" \"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"arguments", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"get_temperature"}}]}})"},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":" \""}}]}})"},
        // Last chunk is added in the for loop below
    };

    for (auto lastFinishReason : {ov::genai::GenerationFinishReason::NONE, ov::genai::GenerationFinishReason::STOP, ov::genai::GenerationFinishReason::LENGTH}) {
        // Need to have new output parser per case to simulate separate request processing
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*phi4Tokenizer, "phi4", "");
        auto chunkToDeltaVecCopy = chunkToDeltaVec;
        if (lastFinishReason == ov::genai::GenerationFinishReason::NONE) {
            chunkToDeltaVecCopy.push_back({"Paris\"}}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"Paris\"}"}}]}})"});
        } else if (lastFinishReason == ov::genai::GenerationFinishReason::STOP) {
            chunkToDeltaVecCopy.push_back({"Paris\"}}", ov::genai::GenerationFinishReason::STOP, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"Paris\"}"}}]}})"});
        } else {
            chunkToDeltaVecCopy.push_back({"Par", ov::genai::GenerationFinishReason::LENGTH, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"Par"}}]}})"});
        }
        int64_t chunkIteration = -1;
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVecCopy) {
            chunkIteration++;
            std::optional<rapidjson::Document> doc = outputParserWithRegularToolParsing->parseChunk(chunk, true, finishReason);
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
                    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: [" << chunk << "] got [" << docStr << "] but expected [" << expected << "]" << chunkIteration;
                }
            } else if (expectedDelta.has_value()) {
                FAIL() << "Mismatch for chunk: [" << chunk << "] got nothing but expected [" << expectedDelta.value() << "]" << chunkIteration;
            } else if (doc.has_value()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                doc->Accept(writer);
                std::string docStr = buffer.GetString();
                FAIL() << "Mismatch for chunk: [" << chunk << "] expected nothing but got [" << docStr << "]" << chunkIteration;
            } else {
                FAIL() << "Mismatch for chunk: [" << chunk << "] " << chunkIteration;
            }
        }
    }
}

TEST_F(Phi4OutputParserTest, ToolCallsWithoutToolsInTheRequestStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Tool parser is available, but tools are not in the request so every chunk is just a regular content
        {"func", "{\"delta\":{\"content\":\"func\"}}"},
        {"tools", "{\"delta\":{\"content\":\"tools\"}}"},
        {"[", "{\"delta\":{\"content\":\"[\"}}"},
        {"{\"", "{\"delta\":{\"content\":\"{\\\"\"}}"},
        {"name", "{\"delta\":{\"content\":\"name\"}}"},
        {"\":", "{\"delta\":{\"content\":\"\\\":\"}}"},
        {" \"", "{\"delta\":{\"content\":\" \\\"\"}}"},
        {"super", "{\"delta\":{\"content\":\"super\"}}"},
        {"_tool", "{\"delta\":{\"content\":\"_tool\"}}"},
        {"_number", "{\"delta\":{\"content\":\"_number\"}}"},
        {"_two", "{\"delta\":{\"content\":\"_two\"}}"},
        {"\",", "{\"delta\":{\"content\":\"\\\",\"}}"},
        {" \"", "{\"delta\":{\"content\":\" \\\"\"}}"},
        {"arguments", "{\"delta\":{\"content\":\"arguments\"}}"},
        {"\":", "{\"delta\":{\"content\":\"\\\":\"}}"},
        {" {", "{\"delta\":{\"content\":\" {\"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"arg1", "{\"delta\":{\"content\":\"arg1\"}}"},
        {"\": ", "{\"delta\":{\"content\":\"\\\": \"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"val{{{ue1", "{\"delta\":{\"content\":\"val{{{ue1\"}}"},
        {"\"", "{\"delta\":{\"content\":\"\\\"\"}}"},
        {"}", "{\"delta\":{\"content\":\"}\"}}"},
        {"}", "{\"delta\":{\"content\":\"}\"}}"},
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        // Second argument is false as we simulate the case where tools have not been provided in the request
        std::optional<rapidjson::Document> doc = outputParserWithRegularToolParsing->parseChunk(chunk, false, ov::genai::GenerationFinishReason::NONE);
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
