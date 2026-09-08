//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************

#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>

#include <memory>
#include <string>
#include <vector>

#include "../../../llm/io_processing/output_parser.hpp"
#include "output_parser_test_utils.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;

namespace {
#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\OpenVINO\\gemma-4-E4B-it-int4-ov";
#else
const std::string tokenizerPath = "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov";
#endif

const std::string questionSchema = R"({"type":"object","properties":{"questions":{"type":"array","items":{"type":"object","properties":{"question":{"type":"string"},"header":{"type":"string"},"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}}}},"multiple":{"type":"boolean"},"custom":{"type":"boolean"}}}}}})";

class Gemma4V2ContractTest : public ::testing::Test {
protected:
    static std::unique_ptr<ov::genai::Tokenizer> tokenizer;

    static void SetUpTestSuite() {
        tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
    }

    static void TearDownTestSuite() {
        tokenizer.reset();
    }

    static ToolsSchemas_t questionTools() {
        ToolsSchemas_t tools;
        tools.emplace("question", ToolSchemaWrapper{nullptr, questionSchema});
        return tools;
    }

    ParsedOutput parse(const std::string& input, const ToolsSchemas_t& tools = questionTools()) {
        OutputParser parser(*tokenizer, "gemma4", "gemma4", tools);
        auto tensor = tokenizer->encode(input).input_ids;
        std::vector<int64_t> tokens(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
        return ovms::test::parseWithStreamer(*tokenizer, parser, tokens, true, true);
    }
};

std::unique_ptr<ov::genai::Tokenizer> Gemma4V2ContractTest::tokenizer;

const std::string nativeQuestionArgs =
    R"(questions:[{question:<|"|>Pick one?<|"|>,header:<|"|>Test<|"|>,options:[{label:<|"|>A<|"|>,description:<|"|>Alpha<|"|>},{label:<|"|>B<|"|>,description:<|"|>Beta<|"|>}],multiple:false,custom:true}])";

const std::string expectedQuestionJson =
    R"({"questions":[{"question":"Pick one?","header":"Test","options":[{"label":"A","description":"Alpha"},{"label":"B","description":"Beta"}],"multiple":false,"custom":true}]})";
}  // namespace

TEST_F(Gemma4V2ContractTest, ParsesOpenCodeQuestionArrayOfObjectsRecursively) {
    auto parsed = parse("<|tool_call>call:question{" + nativeQuestionArgs + "}<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4V2ContractTest, PreservesNestedScalarTypes) {
    const std::string input =
        R"(<|tool_call>call:question{questions:[{question:<|"|>q<|"|>,header:<|"|>h<|"|>,options:[],multiple:false,custom:true}],meta:{count:2,score:22.8,missing:null,flags:[true,false,null,3]}}<tool_call|>)";
    auto parsed = parse(input);
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments,
        R"({"questions":[{"question":"q","header":"h","options":[],"multiple":false,"custom":true}],"meta":{"count":2,"score":22.8,"missing":null,"flags":[true,false,null,3]}})");
}

TEST_F(Gemma4V2ContractTest, AcceptsParenthesizedArgumentsWhenAnchoredByToolMarker) {
    auto parsed = parse("<|tool_call>call:question(" + nativeQuestionArgs + ")<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4V2ContractTest, AcceptsColonNameVariantWhenAnchoredByToolMarker) {
    auto parsed = parse("<|tool_call>:question{" + nativeQuestionArgs + "}<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4V2ContractTest, AcceptsDirectCallImmediatelyAfterReasoningEnd) {
    const std::string input = "<|channel>thought\nNeed user input<channel|>call:question{" + nativeQuestionArgs + "}<tool_call|>";
    auto parsed = parse(input);
    EXPECT_EQ(parsed.reasoning, "Need user input");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4V2ContractTest, DoesNotEmitUnknownToolAsExecutableCall) {
    auto parsed = parse(R"(<|tool_call>call:not_in_request{x:1}<tool_call|>)");
    EXPECT_TRUE(parsed.toolCalls.empty());
}

TEST_F(Gemma4V2ContractTest, MalformedCallIsBoundedAndLaterValidCallSurvives) {
    const std::string input =
        R"(<|tool_call>call:question{questions:[{question:<|"|>broken<|"|>}<tool_call|><|tool_call>call:question{questions:[]}<tool_call|>)";
    auto parsed = parse(input);
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"questions":[]})");
}
