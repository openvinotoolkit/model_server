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
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\LiquidAI\\LFM2-2.6B";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/LiquidAI/LFM2-2.6B";
#endif

static std::unique_ptr<ov::genai::Tokenizer> lfm2Tokenizer;
static const ToolsSchemas_t& EMPTY_TOOLS_SCHEMA = {};  // not used in lfm2

class LFM2OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;

    static void SetUpTestSuite() {
        try {
            lfm2Tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize lfm2 tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize lfm2 tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        lfm2Tokenizer.reset();
    }

    void SetUp() override {
        // For LFM2 model there is only tool parser available
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*lfm2Tokenizer, "lfm2", "", EMPTY_TOOLS_SCHEMA);
    }
};

TEST_F(LFM2OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string inputWithProperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    }
}

TEST_F(LFM2OutputParserTest, ParseToolCallOutputWithNoToolsInTheRequest) {
    std::string inputWithProperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        std::string testInput = input;
        auto generatedTensor = lfm2Tokenizer->encode(testInput, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, false);
        EXPECT_EQ(parsedOutput.content, testInput);
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

TEST_F(LFM2OutputParserTest, ParseToolCallWithObjectArguments) {
    std::string inputWithProperClosure = "<|tool_call_start|>[dummy(config={'name': 'astro_config', 'value': 99})]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[dummy(config={'name': 'astro_config', 'value': 99})]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "dummy");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"config\":{\"name\":\"astro_config\",\"value\":99}}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    }
}

TEST_F(LFM2OutputParserTest, ParseTwoToolCallsAtOnce) {
    std::string inputWithProperClosure = "<|tool_call_start|>[dummy1(config={'name': 'astro_config', 'value': 99}), dummy2(config={'name': 'second_config', 'value': 199})]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[dummy1(config={'name': 'astro_config', 'value': 99}), dummy2(config={'name': 'second_config', 'value': 199})]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "dummy1");
        EXPECT_EQ(parsedOutput.toolCalls[1].name, "dummy2");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"config\":{\"name\":\"astro_config\",\"value\":99}}");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"config\":{\"name\":\"second_config\",\"value\":199}}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
    }
}

TEST_F(LFM2OutputParserTest, ParseToolCallWithArrayArguments) {
    std::string inputWithProperClosure = "<|tool_call_start|>[sort(array=[42, 17, 89, 5, 33], order=\"descending\")]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[sort(array=[42, 17, 89, 5, 33], order=\"descending\")]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "sort");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"array\":[42,17,89,5,33],\"order\":\"descending\"}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
    }
}

TEST_F(LFM2OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string inputWithProperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]<|tool_call_end|>"
                                         "<|tool_call_start|>[another_tool(param1=\"data\", param2=true)]<|tool_call_end|>"
                                         "<|tool_call_start|>[third_tool(key=\"value\")]<|tool_call_end|>";
    std::string inputWithImproperClosure = "<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]<|tool_call_end|>"
                                           "<|tool_call_start|>[another_tool(param1=\"data\", param2=true)]<|tool_call_end|>"
                                           "<|tool_call_start|>[third_tool(key=\"value\")]";

    // LFM2 may produce last tool call without closing tag, so we test both cases
    // The results should be identical
    std::vector<std::string> inputs = {inputWithProperClosure, inputWithImproperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
        auto firstToolCallId = parsedOutput.toolCalls[0].id;

        EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);  // ID should be generated
        auto secondToolCallId = parsedOutput.toolCalls[1].id;
        EXPECT_NE(firstToolCallId, secondToolCallId);  // IDs should be different

        EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
        // Parser removes whitespaces, so we expect arguments value to be without spaces
        EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
        EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);  // ID should be generated
        auto thirdToolCallId = parsedOutput.toolCalls[2].id;
        EXPECT_NE(firstToolCallId, thirdToolCallId);   // IDs should be different
        EXPECT_NE(secondToolCallId, thirdToolCallId);  // IDs should be different
    }
}



TEST_F(LFM2OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
}

TEST_F(LFM2OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<|tool_call_start|>[example_tool(arg1=\"value1\", arg2=42)]<|tool_call_end|>";
    auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    // generatedTokens should now contain content followed by bot token ID and then tool call
    ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    // Parser removes whitespaces, so we expect arguments value to be without spaces
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);  // ID should be generated
}

TEST_F(LFM2OutputParserTest, HolisticStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"<|tool_call_start|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"sort", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"(array", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"sort"}}]}})"},
        {"=[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"42", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 17", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 89", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 5", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 33", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"],", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" order", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"desc", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ending", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"),", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":{"array":[42,17,89,5,33],"order":"descending"}}}]}})"},
        {" d", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ummy", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"(config", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"dummy"}}]}})"},
        {"={", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"'", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"':", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" '", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"astro_config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"',", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" '", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"value", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"':", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 99", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"})]", ov ::genai ::GenerationFinishReason ::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":{"config":{"name":"astro_config","value":99}}}}]}})"},
        {"<|tool_call_end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"ANOTHER_CONTENT_AFTER_TOOL_CALL"}})"},

    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
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

TEST_F(LFM2OutputParserTest, StreamingWithContentBetweenToolCalls) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Tool call phase
        // Starting first tool. Collecting chunk until full name is received. Don't return until then.
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"<|tool_call_start|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"sort", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"(array", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"sort"}}]}})"},
        {"=[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"42", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 17", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 89", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 5", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 33", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"],", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" order", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=\"", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"desc", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ending", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\")]", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":{"array":[42,17,89,5,33],"order":"descending"}}}]}})"},
        {"<|tool_call_end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Some ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Some "}})"},
        {"content ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"content "}})"},
        {"between ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"between "}})"},
        {"tool ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"tool "}})"},
        {"calls.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"calls."}})"},
        {"<|tool_call_start|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"d", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ummy", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"(config", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"dummy"}}]}})"},
        {"={", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"'", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"':", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" '", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"astro_config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"',", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" '", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"value", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"':", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 99", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"})]", ov ::genai ::GenerationFinishReason ::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":{"config":{"name":"astro_config","value":99}}}}]}})"},
        {"<|tool_call_end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"ANOTHER_CONTENT_AFTER_TOOL_CALL"}})"},

    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
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

TEST_F(LFM2OutputParserTest, ToolCallsWithoutToolsInTheRequestStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        // Tool parser is available, but tools are not in the request so every chunk is just a regular content
        {"<|tool_call_start|>\n", "{\"delta\":{\"content\":\"<|tool_call_start|>\\n\"}}"},
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
        {"<|tool_call_end|>\n", "{\"delta\":{\"content\":\"<|tool_call_end|>\\n\"}}"},
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

TEST_F(LFM2OutputParserTest, DISABLED_ParseToolCallWithStringArgumentsContainingSpecialCharacters) {
    {
    std::string input = R"x(<|tool_call_start|>[search(query="price >= 100, (sale)", limit=5)]<|tool_call_end|>)x";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "search");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"query":"price >= 100, (sale)","limit":5})x");
    }

    {
        std::string input = R"(<|tool_call_start|>[format(template="Hello {name}, items: [a, b, c]", count=3)]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "format");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"template":"Hello {name}, items: [a, b, c]","count":3})");
    }

    {
        std::string input = R"x(<|tool_call_start|>[execute(code="print(\"hello world\")", verbose=true)]<|tool_call_end|>)x";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "execute");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"code":"print(\"hello world\")","verbose":true})x");
    }

    {
        std::string input = R"(<|tool_call_start|>[log(message="it's a test, isn't it?", level="warn")]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "log");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"message":"it's a test, isn't it?","level":"warn"})");
    }

    {
        std::string input = R"(<|tool_call_start|>[read_file(path="C:\Users\test\file.txt", encoding="utf-8")]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "read_file");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"path":"C:\\Users\\test\\file.txt","encoding":"utf-8"})");
    }

    {
        std::string input = R"(<|tool_call_start|>[send(payload="{'key': 'value', 'count': 42}", endpoint="api")]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "send");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"payload":"{\"key\":\"value\",\"count\":42}","endpoint":"api"})");
    }

    {
        std::string input = R"(<|tool_call_start|>[create(name="", value=0)]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "create");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"name":"","value":0})");
    }

    {
        std::string input = R"(<|tool_call_start|>[translate(text=\"café résumé naïve\", lang=\"fr\")]<|tool_call_end|>)";
        auto generatedTensor = lfm2Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = outputParserWithRegularToolParsing->parse(generatedTokens, true);
        EXPECT_EQ(parsedOutput.content, "");
        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "translate");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"text":"café résumé naïve","lang":"fr"})");
    }
}
