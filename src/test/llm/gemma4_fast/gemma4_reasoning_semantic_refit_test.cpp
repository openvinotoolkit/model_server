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

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../../../llm/io_processing/output_parser.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;

namespace {
#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\OpenVINO\\gemma-4-E4B-it-int4-ov";
#else
const std::string tokenizerPath = "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov";
#endif

const std::string questionSchema = R"({"type":"object","properties":{"questions":{"type":"array"}}})";
const std::string questionCall = "<|tool_call>call:question{questions:[]}<tool_call|>";
const std::string toolStart = "<|tool_call>";

ToolsSchemas_t questionTools() {
    ToolsSchemas_t tools;
    tools.emplace("question", ToolSchemaWrapper{nullptr, questionSchema});
    return tools;
}

std::optional<ToolCallDelta> driveUntilToolCall(OutputParser& parser, const std::string& firstChunk) {
    for (int step = 0; step < 8; ++step) {
        const auto delta = parser.parseChunk(
            step == 0 ? firstChunk : std::string{},
            {},
            true,
            step == 7 ? ov::genai::GenerationFinishReason::STOP : ov::genai::GenerationFinishReason::NONE);
        if (delta && std::holds_alternative<ToolCallDelta>(*delta)) {
            return std::get<ToolCallDelta>(*delta);
        }
    }
    return std::nullopt;
}

void expectQuestionCall(const std::optional<ToolCallDelta>& call) {
    ASSERT_TRUE(call.has_value());
    EXPECT_EQ(call->name.value_or(""), "question");
    EXPECT_EQ(call->arguments, R"({"questions":[]})");
}

class Gemma4ReasoningSemanticRefitTest : public ::testing::Test {
protected:
    static std::unique_ptr<ov::genai::Tokenizer> tokenizer;

    static void SetUpTestSuite() {
        const char* configured = std::getenv("GEMMA4_TOKENIZER_PATH");
        tokenizer = std::make_unique<ov::genai::Tokenizer>(configured ? configured : tokenizerPath);
    }

    static void TearDownTestSuite() {
        tokenizer.reset();
    }
};

std::unique_ptr<ov::genai::Tokenizer> Gemma4ReasoningSemanticRefitTest::tokenizer;
}  // namespace

TEST_F(Gemma4ReasoningSemanticRefitTest, CanonicalReasoningCloseTransitionsToTool) {
    OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());

    auto reasoning = parser.parseChunk(
        "<|channel>thought\nNeed another tool<channel|>" + questionCall,
        {},
        true,
        ov::genai::GenerationFinishReason::NONE);

    ASSERT_TRUE(reasoning.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*reasoning));
    EXPECT_EQ(std::get<ReasoningDelta>(*reasoning).text, "Need another tool");
    expectQuestionCall(driveUntilToolCall(parser, ""));
}

TEST_F(Gemma4ReasoningSemanticRefitTest, RecoveryToolStartEndsOpenReasoningWithoutCanonicalCloser) {
    OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());

    auto first = parser.parseChunk(
        "<|channel>thought\nNeed another tool",
        {},
        true,
        ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*first));
    EXPECT_EQ(std::get<ReasoningDelta>(*first).text, "Need another tool");

    expectQuestionCall(driveUntilToolCall(parser, questionCall));
}

TEST_F(Gemma4ReasoningSemanticRefitTest, RecoveryImplicitPromptReasoningCanTransitionDirectlyToTool) {
    OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());
    parser.detectAndSetImplicitReasoningStart("prompt<|channel>thought\n");

    expectQuestionCall(driveUntilToolCall(parser, questionCall));
}

TEST_F(Gemma4ReasoningSemanticRefitTest, RecoverySameChunkReasoningPrefixIsPreservedBeforeToolHandoff) {
    OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());
    parser.detectAndSetImplicitReasoningStart("prompt<|channel>thought\n");

    auto reasoning = parser.parseChunk(
        "Need another tool" + questionCall,
        {},
        true,
        ov::genai::GenerationFinishReason::NONE);

    ASSERT_TRUE(reasoning.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*reasoning));
    EXPECT_EQ(std::get<ReasoningDelta>(*reasoning).text, "Need another tool");
    expectQuestionCall(driveUntilToolCall(parser, ""));
}

TEST_F(Gemma4ReasoningSemanticRefitTest, RecoveryPartialToolMarkerIsHeldBackInsteadOfLeakingIntoReasoning) {
    OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());
    parser.detectAndSetImplicitReasoningStart("prompt<|channel>thought\n");

    auto partial = parser.parseChunk(
        "Need another tool<|tool_",
        {},
        true,
        ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(partial.has_value());

    auto reasoning = parser.parseChunk(
        "call>call:question{questions:[]}<tool_call|>",
        {},
        true,
        ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(reasoning.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*reasoning));
    EXPECT_EQ(std::get<ReasoningDelta>(*reasoning).text, "Need another tool");

    expectQuestionCall(driveUntilToolCall(parser, ""));
}

TEST_F(Gemma4ReasoningSemanticRefitTest, RecoveryToolOpenerSurvivesEveryByteSplitWhileReasoningOwnsStream) {
    const std::string suffix = "call:question{questions:[]}<tool_call|>";

    for (size_t split = 0; split <= toolStart.size(); ++split) {
        SCOPED_TRACE(split);
        OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());
        parser.detectAndSetImplicitReasoningStart("prompt<|channel>thought\n");

        std::string reasoningText;
        std::optional<ToolCallDelta> toolCall;

        auto consume = [&](const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
            auto delta = parser.parseChunk(chunk, {}, true, finishReason);
            if (!delta.has_value())
                return;
            if (std::holds_alternative<ReasoningDelta>(*delta))
                reasoningText += std::get<ReasoningDelta>(*delta).text;
            else if (std::holds_alternative<ToolCallDelta>(*delta))
                toolCall = std::get<ToolCallDelta>(*delta);
        };

        consume("Need another tool" + toolStart.substr(0, split), ov::genai::GenerationFinishReason::NONE);
        consume(toolStart.substr(split) + suffix, ov::genai::GenerationFinishReason::NONE);
        for (int step = 0; step < 8 && !toolCall.has_value(); ++step) {
            consume("", step == 7 ? ov::genai::GenerationFinishReason::STOP : ov::genai::GenerationFinishReason::NONE);
        }

        EXPECT_EQ(reasoningText, "Need another tool");
        expectQuestionCall(toolCall);
    }
}
