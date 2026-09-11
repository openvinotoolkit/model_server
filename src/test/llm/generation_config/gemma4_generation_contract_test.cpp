// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <openvino/genai/llm_pipeline.hpp>
#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "src/llm/io_processing/generation_config_builder.hpp"
#include "src/test/platform_utils.hpp"

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

const Structured::StructuralTag& rootGrammar(const ov::genai::GenerationConfig& config) {
    return std::get<Structured::StructuralTag>(
        config.structured_output_config.value().structural_tags_config.value());
}

template <typename T>
const T& grammar(const ov::genai::GenerationConfig& config) {
    const auto& root = rootGrammar(config);
    if (const auto* sequence = std::get_if<std::shared_ptr<Structured::Concat>>(&root))
        return *std::get<std::shared_ptr<T>>((*sequence)->elements.back());
    if (const auto* alternatives = std::get_if<std::shared_ptr<Structured::Union>>(&root)) {
        for (const auto& element : (*alternatives)->elements) {
            if (const auto* tags = std::get_if<std::shared_ptr<T>>(&element))
                return **tags;
            if (const auto* sequence = std::get_if<std::shared_ptr<Structured::Concat>>(&element))
                return *std::get<std::shared_ptr<T>>((*sequence)->elements.back());
        }
    }
    return *std::get<std::shared_ptr<T>>(root);
}

const Structured::TriggeredTags& autoGrammar(const ov::genai::GenerationConfig& config) {
    return *std::get<std::shared_ptr<Structured::TriggeredTags>>(rootGrammar(config));
}

std::string grammarString(const ov::genai::GenerationConfig& config) {
    return std::visit([](const auto& value) {
        return Structured::structural_tag_to_string(value);
    }, rootGrammar(config));
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
    gemma.unsetStructuredOutputConfig();
    EXPECT_TRUE(gemma.getConfig().structured_output_config.has_value());

    GenerationConfigBuilder hermes({}, "hermes3", false, STANDARD);
    hermes.parseConfigFromRequest(request);
    ASSERT_TRUE(hermes.getConfig().structured_output_config.has_value());
    hermes.unsetStructuredOutputConfig();
    EXPECT_FALSE(hermes.getConfig().structured_output_config.has_value());
}

TEST(Gemma4GenerationContractTest, HardChoicesAllowReasoningBeforeMandatoryToolSelection) {
    for (const std::string choice : {"required", "second"}) {
        SCOPED_TRACE(choice);
        auto request = requestWithTools(choice);
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        builder.parseConfigFromRequest(request);
        const auto& root = rootGrammar(builder.getConfig());
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Structured::Union>>(root));
        const auto& alternatives = *std::get<std::shared_ptr<Structured::Union>>(root);
        ASSERT_EQ(alternatives.elements.size(), 2u);
        const auto& toolsOnly = *std::get<std::shared_ptr<Structured::TagsWithSeparator>>(alternatives.elements[0]);
        EXPECT_TRUE(toolsOnly.at_least_one);
        EXPECT_EQ(toolsOnly.tags.size(), choice == "second" ? 1u : 2u);
        if (choice == "second")
            EXPECT_EQ(toolsOnly.tags[0].begin, "<|tool_call>call:second");
        const auto& thoughtThenTools = *std::get<std::shared_ptr<Structured::Concat>>(alternatives.elements[1]);
        ASSERT_EQ(thoughtThenTools.elements.size(), 2u);
        const auto& thought = *std::get<std::shared_ptr<Structured::Tag>>(thoughtThenTools.elements[0]);
        EXPECT_EQ(thought.begin, "<|channel>thought\n");
        EXPECT_EQ(thought.end, "<channel|>");
        const auto& toolsAfterThought = *std::get<std::shared_ptr<Structured::TagsWithSeparator>>(thoughtThenTools.elements[1]);
        EXPECT_TRUE(toolsAfterThought.at_least_one);
        EXPECT_EQ(toolsAfterThought.tags.size(), choice == "second" ? 1u : 2u);
        if (choice == "second")
            EXPECT_EQ(toolsAfterThought.tags[0].begin, "<|tool_call>call:second");

        ov::genai::Tokenizer tokenizer(getGenericFullPathForSrcTest(
            "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov"));
        EXPECT_NO_THROW(builder.validateStructuredOutputConfig(tokenizer));
    }
}

TEST(Gemma4GenerationContractTest, HardChoiceCannotBeClearedButAutoMayFallbackAfterValidationFailure) {
    for (const std::string choice : {"required", "second"}) {
        auto request = requestWithTools(choice);
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        builder.parseConfigFromRequest(request);
        ASSERT_TRUE(builder.getConfig().structured_output_config.has_value());
        EXPECT_NO_THROW(builder.unsetStructuredOutputConfig()) << choice;
        EXPECT_TRUE(builder.getConfig().structured_output_config.has_value()) << choice;
    }

    auto autoRequest = requestWithTools("auto");
    GenerationConfigBuilder autoBuilder({}, "gemma4", false, STANDARD);
    autoBuilder.parseConfigFromRequest(autoRequest);
    ASSERT_TRUE(autoBuilder.getConfig().structured_output_config.has_value());
    EXPECT_NO_THROW(autoBuilder.unsetStructuredOutputConfig());
    EXPECT_FALSE(autoBuilder.getConfig().structured_output_config.has_value());
}

TEST(Gemma4GenerationContractTest, AutoUsesTriggeredToolGrammarAndHardChoicesStayImmediateAndRepeatable) {
    for (bool guided : {false, true}) {
        for (const std::string choice : {"auto", "required", "second"}) {
            SCOPED_TRACE(choice);
            auto request = requestWithTools(choice);
            GenerationConfigBuilder builder({}, "gemma4", guided, STANDARD);
            builder.parseConfigFromRequest(request);
            ASSERT_TRUE(builder.getConfig().structured_output_config.has_value());
            if (choice == "auto") {
                const auto& triggered = autoGrammar(builder.getConfig());
                ASSERT_EQ(triggered.triggers.size(), 1u);
                EXPECT_EQ(triggered.triggers[0], "<|tool_call>");
                EXPECT_FALSE(triggered.at_least_one)
                    << "Gemma4 auto must permit ordinary prose without a tool call";
                EXPECT_FALSE(triggered.stop_after_first);
                ASSERT_EQ(triggered.tags.size(), 2u);

                std::set<std::string> begins;
                for (const auto& tag : triggered.tags) {
                    begins.insert(tag.begin);
                    EXPECT_EQ(tag.end, "<tool_call|>");
                    ASSERT_TRUE(std::holds_alternative<Structured::JSONSchema>(tag.content));
                    EXPECT_EQ(std::get<Structured::JSONSchema>(tag.content).value, emptySchema);
                }
                EXPECT_EQ(begins, (std::set<std::string>{"<|tool_call>call:first", "<|tool_call>call:second"}));
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

TEST(Gemma4GenerationContractTest, ParallelToolCallsControlsGrammarRepeatability) {
    for (const std::string choice : {"auto", "required", "second"}) {
        for (bool parallel : {false, true}) {
            SCOPED_TRACE(choice + std::string(parallel ? ":parallel" : ":single"));
            auto request = requestWithTools(choice);
            request.parallelToolCalls = parallel;
            GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
            builder.parseConfigFromRequest(request);

            if (choice == "auto") {
                const auto& triggered = autoGrammar(builder.getConfig());
                EXPECT_EQ(triggered.stop_after_first, !parallel);
            } else {
                const auto& root = rootGrammar(builder.getConfig());
                ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Structured::Union>>(root));
                const auto& alternatives = *std::get<std::shared_ptr<Structured::Union>>(root);
                ASSERT_EQ(alternatives.elements.size(), 2u);
                const auto& toolsOnly = *std::get<std::shared_ptr<Structured::TagsWithSeparator>>(alternatives.elements[0]);
                EXPECT_EQ(toolsOnly.stop_after_first, !parallel);
                const auto& thoughtThenTools = *std::get<std::shared_ptr<Structured::Concat>>(alternatives.elements[1]);
                const auto& toolsAfterThought = *std::get<std::shared_ptr<Structured::TagsWithSeparator>>(thoughtThenTools.elements[1]);
                EXPECT_EQ(toolsAfterThought.stop_after_first, !parallel);
            }
        }
    }
}

TEST(Gemma4GenerationContractTest, AutoTriggeredGrammarValidatesWithGemmaTokenizer) {
    auto request = requestWithTools("auto");
    GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
    builder.parseConfigFromRequest(request);
    ASSERT_TRUE(builder.getConfig().structured_output_config.has_value());

    ov::genai::Tokenizer tokenizer(getGenericFullPathForSrcTest(
        "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov"));
    EXPECT_NO_THROW(builder.validateStructuredOutputConfig(tokenizer));
}

TEST(Gemma4GenerationContractTest, OpenCodeQuestionSchemaIsEnforcedForAutoAndRequired) {
    for (const std::string choice : {"auto", "required"}) {
        OpenAIRequest request;
        request.toolChoice = choice;
        request.toolNameSchemaMap.emplace("question", ToolSchemaWrapper{nullptr, openCodeQuestionSchema});
        GenerationConfigBuilder builder({}, "gemma4", true, STANDARD);
        builder.parseConfigFromRequest(request);

        if (choice == "auto") {
            const auto& triggered = autoGrammar(builder.getConfig());
            ASSERT_EQ(triggered.tags.size(), 1u);
            EXPECT_EQ(triggered.tags[0].begin, "<|tool_call>call:question");
            EXPECT_EQ(std::get<Structured::JSONSchema>(triggered.tags[0].content).value, openCodeQuestionSchema);
            EXPECT_FALSE(triggered.at_least_one);
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
    for (const std::string choice : {"auto", "first"}) {
        auto request = requestWithTools(choice);
        request.toolNameSchemaMap.erase("second");
        request.toolNameSchemaMap["first"].stringRepr = schema;
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        builder.parseConfigFromRequest(request);
        if (choice == "auto") {
            const auto& triggered = autoGrammar(builder.getConfig());
            ASSERT_EQ(triggered.tags.size(), 1u);
            EXPECT_EQ(std::get<Structured::JSONSchema>(triggered.tags[0].content).value, schema);
        } else {
            const auto& tags = grammar<Structured::TagsWithSeparator>(builder.getConfig());
            ASSERT_EQ(tags.tags.size(), 1u);
            EXPECT_EQ(std::get<Structured::JSONSchema>(tags.tags[0].content).value, schema);
        }
    }
}

TEST(Gemma4GenerationContractTest, RejectsToolNamesThatItsParserCannotExecute) {
    for (const std::string choice : {"auto", "required"}) {
        for (const std::string name : {"bad name", "", "bad:name"}) {
            OpenAIRequest request;
            request.toolChoice = choice;
            request.toolNameSchemaMap.emplace(name, ToolSchemaWrapper{nullptr, emptySchema});
            GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
            EXPECT_THROW(builder.parseConfigFromRequest(request), std::invalid_argument) << choice << ":" << name;
        }
    }
}

TEST(Gemma4GenerationContractTest, RejectsEmptyToolSchemasForAutoAndHardChoices) {
    for (const std::string choice : {"auto", "required", "first"}) {
        auto request = requestWithTools(choice);
        request.toolNameSchemaMap["first"].stringRepr.clear();
        if (choice == "first")
            request.toolNameSchemaMap.erase("second");
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        EXPECT_THROW(builder.parseConfigFromRequest(request), std::invalid_argument) << choice;
    }
}

TEST(Gemma4GenerationContractTest, GeneratedToolGrammarsNeverUseEmptyConstString) {
    for (const std::string choice : {"auto", "required", "second"}) {
        auto request = requestWithTools(choice);
        GenerationConfigBuilder builder({}, "gemma4", false, STANDARD);
        builder.parseConfigFromRequest(request);
        ASSERT_TRUE(builder.getConfig().structured_output_config.has_value());
        EXPECT_EQ(grammarString(builder.getConfig()).find("ConstString(\"\")"), std::string::npos) << choice;
    }
}
