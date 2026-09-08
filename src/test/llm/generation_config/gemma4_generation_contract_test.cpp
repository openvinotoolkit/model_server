// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <openvino/genai/llm_pipeline.hpp>

#include <string>
#include <variant>

#include "src/llm/io_processing/generation_config_builder.hpp"

using namespace ovms;
using Structured = ov::genai::StructuredOutputConfig;

namespace {
const std::string emptySchema = R"({"type":"object","properties":{},"additionalProperties":false})";
const std::string responseSchema = R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object","properties":{"answer":{"type":"string"}}}}})";
const std::string openCodeQuestionSchema = R"({"type":"object","properties":{"questions":{"type":"array","items":{"type":"object","properties":{"question":{"type":"string"},"header":{"type":"string"},"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}},"required":["label","description"]}},"multiple":{"type":"boolean"},"custom":{"type":"boolean"}},"required":["question","header","options"]}}},"required":["questions"]})";

OpenAIRequest requestWithTools(const std::string& choice) {
    OpenAIRequest request;
    request.toolChoice = choice;
    request.toolNameSchemaMap.emplace("first", ToolSchemaWrapper{nullptr, emptySchema});
    request.toolNameSchemaMap.emplace("second", ToolSchemaWrapper{nullptr, emptySchema});
    return request;
}

template <typename T>
const T& grammar(const ov::genai::GenerationConfig& config) {
    return *std::get<std::shared_ptr<T>>(std::get<Structured::StructuralTag>(
        config.structured_output_config.value().structural_tags_config.value()));
}
}  // namespace

TEST(Gemma4GenerationContractTest, AbsentToolsAndNonePreserveResponseFormat) {
    for (bool guided : {false, true}) {
        for (bool response : {false, true}) {
            for (bool tools : {false, true}) {
                auto request = tools ? requestWithTools("none") : OpenAIRequest{};
                if (response)
                    request.responseFormat = responseSchema;
                GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
                builder.parseConfigFromRequest(request);
                EXPECT_EQ(builder.getConfig().structured_output_config.has_value(), response);
            }
        }
    }
}

TEST(Gemma4GenerationContractTest, ValidationFallbackPolicyIsGemmaSpecific) {
    auto request = requestWithTools("required");

    GenerationConfigBuilder gemma({}, "gemma4", false, STANDARD);
    gemma.parseConfigFromRequest(request);
    ASSERT_TRUE(gemma.getConfig().structured_output_config.has_value());
    EXPECT_NO_THROW(gemma.unsetStructuredOutputConfig());
    EXPECT_TRUE(gemma.getConfig().structured_output_config.has_value());

    GenerationConfigBuilder hermes({}, "hermes3", false, STANDARD);
    hermes.parseConfigFromRequest(request);
    ASSERT_TRUE(hermes.getConfig().structured_output_config.has_value());
    EXPECT_NO_THROW(hermes.unsetStructuredOutputConfig());
    EXPECT_FALSE(hermes.getConfig().structured_output_config.has_value());
}

TEST(Gemma4GenerationContractTest, AutoUsesNativeGemmaGenerationAndHardChoicesStayImmediateAndRepeatable) {
    for (bool guided : {false, true}) {
        for (const std::string choice : {"auto", "required", "second"}) {
            auto request = requestWithTools(choice);
            GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
            builder.parseConfigFromRequest(request);
            if (choice == "auto") {
                EXPECT_FALSE(builder.getConfig().structured_output_config)
                    << "Gemma4 auto tool choice must remain native/unconstrained";
            } else {
                const auto& tags = grammar<Structured::TagsWithSeparator>(builder.getConfig());
                EXPECT_TRUE(tags.at_least_one);
                EXPECT_FALSE(tags.stop_after_first);
                EXPECT_TRUE(tags.separator.empty());
                ASSERT_EQ(tags.tags.size(), choice == "second" ? 1u : 2u);
                if (choice == "second")
                    EXPECT_EQ(tags.tags[0].begin, "<|tool_call>call:second");
            }
        }
    }
}

TEST(Gemma4GenerationContractTest, OpenCodeQuestionSchemaDoesNotConstrainAutoButRemainsEnforcedForRequired) {
    for (const std::string choice : {"auto", "required"}) {
        OpenAIRequest request;
        request.toolChoice = choice;
        request.toolNameSchemaMap.emplace("question", ToolSchemaWrapper{nullptr, openCodeQuestionSchema});
        GenerationConfigBuilder builder({}, "gemma4", true, STANDARD);
        builder.parseConfigFromRequest(request);

        if (choice == "auto") {
            EXPECT_FALSE(builder.getConfig().structured_output_config);
        } else {
            const auto& tags = grammar<Structured::TagsWithSeparator>(builder.getConfig());
            ASSERT_EQ(tags.tags.size(), 1u);
            EXPECT_EQ(tags.tags[0].begin, "<|tool_call>call:question");
            EXPECT_EQ(std::get<Structured::JSONSchema>(tags.tags[0].content).value, openCodeQuestionSchema);
        }
    }
}

TEST(Gemma4GenerationContractTest, ImpossibleHardChoiceIsRejected) {
    for (bool guided : {false, true}) {
        for (const std::string choice : {"required", "missing"}) {
            OpenAIRequest request;
            request.toolChoice = choice;
            GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
            EXPECT_THROW(builder.parseConfigFromRequest(request), std::invalid_argument);
        }
        auto request = requestWithTools("missing");
        GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
        EXPECT_THROW(builder.parseConfigFromRequest(request), std::invalid_argument);
    }
}

TEST(Gemma4GenerationContractTest, ResponseFormatAndActiveToolsCannotSilentlyReplaceConstraints) {
    for (bool guided : {false, true}) {
        for (const std::string choice : {"auto", "required", "second"}) {
            auto request = requestWithTools(choice);
            request.responseFormat = responseSchema;
            GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
            EXPECT_THROW(builder.parseConfigFromRequest(request), std::invalid_argument);
        }
    }
}

TEST(Gemma4GenerationContractTest, ObjectSchemaIsPassedWithoutLosingNestedConstraints) {
    const std::string schema = R"({"type":"object","properties":{"nested":{"type":"object","properties":{"x":{"enum":[1,2]}}},"array":{"type":"array","items":{"type":["number","boolean","null"]}},"optional":{"type":"string"}},"required":["nested"],"additionalProperties":false})";
    auto request = requestWithTools("first");
    request.toolNameSchemaMap.erase("second");
    request.toolNameSchemaMap["first"].stringRepr = schema;
    GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
    builder.parseConfigFromRequest(request);
    const auto& tags = grammar<Structured::TagsWithSeparator>(builder.getConfig());
    ASSERT_EQ(tags.tags.size(), 1u);
    EXPECT_EQ(std::get<Structured::JSONSchema>(tags.tags[0].content).value, schema);
}
