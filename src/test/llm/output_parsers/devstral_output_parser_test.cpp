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

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\unsloth\\Devstral-Small-2507";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/unsloth/Devstral-Small-2507/";
#endif

static ovms::ToolsSchemas_t EMPTY_TOOLS_SCHEMA = {};  // not used for mistral
static std::unique_ptr<ov::genai::Tokenizer> devstralTokenizer;

class DevstralOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;

    static void SetUpTestSuite() {
        try {
            devstralTokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize devstral tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize devstral tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        devstralTokenizer.reset();
    }

    void SetUp() override {
        // declare tools_schema
        static std::map<std::string, std::string> toolSchemasInput = {
            {"example_tool", R"({"properties": {"arg1": {"type": "string", "description": "A string argument."}}, "required": ["arg1"]})"},
        };

        static std::vector<std::unique_ptr<rapidjson::Document>> schemaDocsStorage;

        auto convertStringToolSchemasStringToToolsSchemas = [](
                                                                const std::map<std::string, std::string>& input) -> ToolsSchemas_t {
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
        };

        static ovms::ToolsSchemas_t toolsSchemas = convertStringToolSchemasStringToToolsSchemas(toolSchemasInput);
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*devstralTokenizer, "devstral", "", toolsSchemas);
    }
};

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string input = "[TOOL_CALLS]example_tool[ARGS]{\"arg1\":\"value1 with new line \\n and \"quote\" and slash \\ \",\"arg2\":42}</s>";
    std::string testInput = input;
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1 with new line \\n and \"quote\" and slash \\ \",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithSingleToolCall_MissingEndTag) {
    std::string testInput = "Reasoning before tool call [TOOL_CALLS] example_tool [ARGS]{\"arg1\":\"value1\",\"arg2\":42}";
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "Reasoning before tool call ");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithSingleToolCall_EmptyArguments) {
    std::string testInput = "Reasoning before tool call [TOOL_CALLS]example_tool[ARGS]</s>";
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "Reasoning before tool call ");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = devstralTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string testInput = "Reasoning before tool call [TOOL_CALLS]example_tool[ARGS]{\"arg1\":\"value1\",\"arg2\":42}</s>";
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "Reasoning before tool call ");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithInvalidOrder) {
    std::string testInput = "Reasoning before tool call [ARGS]example_tool[TOOL_CALLS]{\"arg1\":\"value1\",\"arg2\":42}</s>";
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "Reasoning before tool call example_tool{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithMissingArgsTag) {
    std::string input = "Some content [TOOL_CALLS]example_tool{\"arg1\":\"value1\",\"arg2\":42}</s>";
    std::string testInput = input;
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    // Same expected content as tokenizer does not add special tokens
    EXPECT_EQ(parsedOutput.content, "Some content example_tool{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithArrayArguments) {
    std::string input = "[TOOL_CALLS]example_tool[ARGS]{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}</s>";
    std::string testInput = input;
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"filepath\":\"/var/log/db.log\",\"status\":[\"completed\",\"failed\"],\"encoding\":\"utf-8\",\"processFunction\":\"processFunction\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, ParseToolCallOutputWithInvalidArguments) {
    std::string input = "[TOOL_CALLS]example_tool[ARGS]{ \"filepath\": \"/var/log/db.log\", \"status\": </s>";
    std::string testInput = input;
    auto generatedTensor = devstralTokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{ \"filepath\": \"/var/log/db.log\", \"status\": ");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(DevstralOutputParserTest, HolisticStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"Reasoning", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Reasoning"}})"},
        {"example", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"example"}})"},
        {"[TOOL_CALLS]", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" get", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"weather", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" [ARGS]", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        {"{\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}})"},
        {"city\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"city\":"}}]}})"},
        {" \"Paris", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \"Paris"}}]}})"},
        {" \"capital of ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \"capital of "}}]}})"},
        {"art\\vine \\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"art\\vine \\n"}}]}})"},
        // Last chunk is added in the for loop below
    };
    ToolsSchemas_t tools_schemas = {
        {"get_weather", ToolSchemaWrapper{}}};
    for (auto lastFinishReason : {ov::genai::GenerationFinishReason::STOP, ov::genai::GenerationFinishReason::LENGTH}) {
        // Need to have new output parser per case to simulate separate request processing
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*devstralTokenizer, "devstral", "", tools_schemas);
        auto chunkToDeltaVecCopy = chunkToDeltaVec;
        if (lastFinishReason == ov::genai::GenerationFinishReason::STOP) {
            chunkToDeltaVecCopy.push_back({"\"}", ov::genai::GenerationFinishReason::STOP, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"});
        } else {
            chunkToDeltaVecCopy.push_back({"\"}", ov::genai::GenerationFinishReason::LENGTH, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"});
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

TEST_F(DevstralOutputParserTest, EmptyArgumentsStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"[TOOL_CALLS]", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"list", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tools", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[ARGS]", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"list_tools"}}]}})"},
        {"</s>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}})"},
    };
    ToolsSchemas_t tools_schemas = {
        {"list_tools", ToolSchemaWrapper{}}};

    int64_t chunkIteration = 0;
    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
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

TEST_F(DevstralOutputParserTest, ToolCallsWithoutToolsInTheRequestStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Tool parser is available, but tools are not in the request so every chunk is just a regular content
        {"[TOOL_CALLS]", "{\"delta\":{\"content\":\"[TOOL_CALLS]\"}}"},
        {"get_", "{\"delta\":{\"content\":\"get_\"}}"},
        {"weather", "{\"delta\":{\"content\":\"weather\"}}"},
        {"[ARGS]", "{\"delta\":{\"content\":\"[ARGS]\"}}"},
        {"{\"", "{\"delta\":{\"content\":\"{\\\"\"}}"},
        {"city\":", "{\"delta\":{\"content\":\"city\\\":\"}}"},
        {"\"Paris\"", "{\"delta\":{\"content\":\"\\\"Paris\\\"\"}}"},
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
