// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <openvino/genai/llm_pipeline.hpp>

#include <memory>
#include <string>
#include <variant>

#include "src/llm/io_processing/generation_config_builder.hpp"
#include "src/llm/io_processing/input_processors/chat_template_processor.hpp"

using namespace ovms;
using Structured = ov::genai::StructuredOutputConfig;

namespace {
const std::string emptySchema = R"({"type":"object","properties":{},"additionalProperties":false})";

OpenAIRequest requestWithTools(const std::string& choice) {
    OpenAIRequest request;
    request.toolChoice = choice;
    request.toolNameSchemaMap.emplace("first", ToolSchemaWrapper{nullptr, emptySchema});
    request.toolNameSchemaMap.emplace("second", ToolSchemaWrapper{nullptr, emptySchema});
    return request;
}

const Structured::StructuralTag& rootGrammar(const ov::genai::GenerationConfig& config) {
    return std::get<Structured::StructuralTag>(
        config.structured_output_config.value().structural_tags_config.value());
}
}  // namespace

TEST(Gemma4PromptStateGenerationContractTest, OpenPromptReasoningUsesRequiredTriggeredNativeToolGrammar) {
    for (const std::string choice : {"required", "second"}) {
        SCOPED_TRACE(choice);
        auto request = requestWithTools(choice);
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        builder.parseConfigFromRequest(request);

        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Structured::Union>>(rootGrammar(builder.getConfig())));

        const bool changed = adaptGemma4HardToolGrammarForRenderedPrompt(
            builder.getConfig(), "<|turn>model\n<|channel>thought\n");
        EXPECT_TRUE(changed);

        const auto& root = rootGrammar(builder.getConfig());
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Structured::TriggeredTags>>(root));
        const auto& triggered = *std::get<std::shared_ptr<Structured::TriggeredTags>>(root);
        ASSERT_EQ(triggered.triggers.size(), 1u);
        EXPECT_EQ(triggered.triggers[0], "<|tool_call>");
        EXPECT_TRUE(triggered.at_least_one);
        EXPECT_FALSE(triggered.stop_after_first);
        ASSERT_EQ(triggered.tags.size(), choice == "second" ? 1u : 2u);
        if (choice == "second")
            EXPECT_EQ(triggered.tags[0].begin, "<|tool_call>call:second");
    }
}

TEST(Gemma4PromptStateGenerationContractTest, OrdinaryPromptKeepsImmediateHardGrammar) {
    auto request = requestWithTools("required");
    GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
    builder.parseConfigFromRequest(request);

    const bool changed = adaptGemma4HardToolGrammarForRenderedPrompt(
        builder.getConfig(), "<|turn>user\nUse a tool<turn|>\n<|turn>model\n");
    EXPECT_FALSE(changed);
    EXPECT_TRUE(std::holds_alternative<std::shared_ptr<Structured::Union>>(rootGrammar(builder.getConfig())));
}

TEST(Gemma4PromptStateGenerationContractTest, OpenPromptReasoningPreservesSingleCallPolicy) {
    auto request = requestWithTools("required");
    request.parallelToolCalls = false;
    GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
    builder.parseConfigFromRequest(request);

    ASSERT_TRUE(adaptGemma4HardToolGrammarForRenderedPrompt(
        builder.getConfig(), "prefix<|channel>thought\n"));
    const auto& triggered = *std::get<std::shared_ptr<Structured::TriggeredTags>>(rootGrammar(builder.getConfig()));
    EXPECT_TRUE(triggered.at_least_one);
    EXPECT_TRUE(triggered.stop_after_first);
}
