// Copyright 2026 Intel Corporation. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "src/llm/io_processing/generation_config_builder.hpp"

using namespace ovms;
namespace {
using Structured = ov::genai::StructuredOutputConfig;

OpenAIRequest makeRequest(const std::string& choice) {
    OpenAIRequest req;
    req.toolChoice = choice;
    req.toolNameSchemaMap.emplace(
        "weather", ToolSchemaWrapper{nullptr,
            R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"});
    req.toolNameSchemaMap.emplace(
        "clock", ToolSchemaWrapper{nullptr,
            R"({"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]})"});
    return req;
}

const Structured::StructuralTag* tag(ov::genai::GenerationConfig& cfg) {
    if (!cfg.structured_output_config ||
        !cfg.structured_output_config->structural_tags_config)
        return nullptr;
    return std::get_if<Structured::StructuralTag>(
        &*cfg.structured_output_config->structural_tags_config);
}

const Structured::Concat* asOpenThought(const Structured::StructuralTag* g) {
    EXPECT_NE(g, nullptr);
    if (!g)
        return nullptr;
    const auto* held = std::get_if<std::shared_ptr<Structured::Concat>>(g);
    EXPECT_NE(held, nullptr);
    if (!held || !*held)
        return nullptr;
    EXPECT_EQ((*held)->elements.size(), 2u);
    if ((*held)->elements.size() != 2)
        return nullptr;
    const auto* thought = std::get_if<std::shared_ptr<Structured::Tag>>(&(*held)->elements[0]);
    EXPECT_NE(thought, nullptr);
    if (!thought || !*thought)
        return nullptr;
    EXPECT_TRUE((*thought)->begin.empty());
    EXPECT_EQ((*thought)->end, "</think>\n\n");
    EXPECT_TRUE(std::holds_alternative<Structured::AnyText>((*thought)->content));
    return held->get();
}

const Structured::TagsWithSeparator* tagsAfterThought(const Structured::StructuralTag* g) {
    const auto* concat = asOpenThought(g);
    if (!concat)
        return nullptr;
    const auto* list = std::get_if<std::shared_ptr<Structured::TagsWithSeparator>>(&concat->elements[1]);
    EXPECT_NE(list, nullptr);
    if (!list || !*list)
        return nullptr;
    return list->get();
}

const Structured::TriggeredTags* triggeredAfterThought(const Structured::StructuralTag* g) {
    const auto* concat = asOpenThought(g);
    if (!concat)
        return nullptr;
    const auto* trig = std::get_if<std::shared_ptr<Structured::TriggeredTags>>(&concat->elements[1]);
    EXPECT_NE(trig, nullptr);
    if (!trig || !*trig)
        return nullptr;
    return trig->get();
}

void expectXmlTags(const Structured::TagsWithSeparator& list, size_t count, bool stopAfterFirst) {
    EXPECT_EQ(list.separator, "\n");
    EXPECT_TRUE(list.at_least_one);
    EXPECT_EQ(list.stop_after_first, stopAfterFirst);
    ASSERT_EQ(list.tags.size(), count);
    for (const auto& t : list.tags) {
        EXPECT_NE(t.begin.find("<tool_call>"), std::string::npos);
        EXPECT_NE(t.begin.find("<function="), std::string::npos);
        EXPECT_EQ(t.end, "</function>\n</tool_call>");
        const auto* args = std::get_if<Structured::QwenXMLParametersFormat>(&t.content);
        ASSERT_NE(args, nullptr);
        EXPECT_NE(args->json_schema.find(R"("type":"object")"), std::string::npos);
    }
}
} // namespace

TEST(QwenmonsterGenerationPolicy, RequiredUsesNativeXMLSchemas) {
    for (const auto enableThinking : {std::optional<bool>{}, std::optional<bool>{true}}) {
        auto req = makeRequest("required");
        req.enableThinking = enableThinking;
        GenerationConfigBuilder builder({}, "qwen3coder", false, DecodingMethod::STANDARD);
        builder.parseConfigFromRequest(req);
        EXPECT_TRUE(builder.requiresValidStructuredOutput());
        const auto* list = tagsAfterThought(tag(builder.getConfig()));
        ASSERT_NE(list, nullptr);
        expectXmlTags(*list, 2u, false);
    }
}

TEST(QwenmonsterGenerationPolicy, NamedRestrictsToOneTool) {
    auto req = makeRequest("clock");
    req.parallelToolCalls = false;
    GenerationConfigBuilder builder({}, "qwen3coder", false, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(req);
    EXPECT_TRUE(builder.requiresValidStructuredOutput());
    const auto* list = tagsAfterThought(tag(builder.getConfig()));
    ASSERT_NE(list, nullptr);
    expectXmlTags(*list, 1u, true);
    EXPECT_NE(list->tags[0].begin.find("<function=clock>"), std::string::npos);
    EXPECT_EQ(list->tags[0].begin.find("<function=weather>"), std::string::npos);
}

TEST(QwenmonsterGenerationPolicy, ThinkingOffRequiredUsesImmediateTags) {
    auto req = makeRequest("required");
    req.enableThinking = false;
    GenerationConfigBuilder builder({}, "qwen3coder", false, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(req);
    EXPECT_TRUE(builder.requiresValidStructuredOutput());
    const auto* g = tag(builder.getConfig());
    ASSERT_NE(g, nullptr);
    EXPECT_FALSE(std::holds_alternative<std::shared_ptr<Structured::Concat>>(*g));
    const auto* list = std::get_if<std::shared_ptr<Structured::TagsWithSeparator>>(g);
    ASSERT_NE(list, nullptr);
    ASSERT_TRUE(*list);
    expectXmlTags(**list, 2u, false);
}

TEST(QwenmonsterGenerationPolicy, AutoUnconstrainedByDefault) {
    GenerationConfigBuilder builder({}, "qwen3coder", false, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(makeRequest("auto"));
    EXPECT_FALSE(builder.requiresValidStructuredOutput());
    EXPECT_FALSE(builder.getConfig().structured_output_config.has_value());
}

TEST(QwenmonsterGenerationPolicy, AutoGuidedIsLazyNotForced) {
    auto req = makeRequest("auto");
    req.parallelToolCalls = false;
    GenerationConfigBuilder builder({}, "qwen3coder", true, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(req);
    EXPECT_FALSE(builder.requiresValidStructuredOutput());
    const auto* trig = triggeredAfterThought(tag(builder.getConfig()));
    ASSERT_NE(trig, nullptr);
    ASSERT_EQ(trig->triggers.size(), 1u);
    EXPECT_EQ(trig->triggers[0], "<tool_call>");
    EXPECT_FALSE(trig->at_least_one);
    EXPECT_TRUE(trig->stop_after_first);
    ASSERT_EQ(trig->tags.size(), 2u);
}

TEST(QwenmonsterGenerationPolicy, ThinkingOffGuidedUsesImmediateTrigger) {
    auto req = makeRequest("auto");
    req.enableThinking = false;
    GenerationConfigBuilder builder({}, "qwen3coder", true, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(req);
    const auto* g = tag(builder.getConfig());
    ASSERT_NE(g, nullptr);
    EXPECT_FALSE(std::holds_alternative<std::shared_ptr<Structured::Concat>>(*g));
    const auto* trig = std::get_if<std::shared_ptr<Structured::TriggeredTags>>(g);
    ASSERT_NE(trig, nullptr);
    ASSERT_TRUE(*trig);
    ASSERT_EQ((*trig)->triggers.size(), 1u);
    EXPECT_EQ((*trig)->triggers[0], "<tool_call>");
    EXPECT_FALSE((*trig)->at_least_one);
    EXPECT_FALSE((*trig)->stop_after_first);
}

TEST(QwenmonsterGenerationPolicy, NoneDoesNotInstallGrammar) {
    GenerationConfigBuilder builder({}, "qwen3coder", true, DecodingMethod::STANDARD);
    builder.parseConfigFromRequest(makeRequest("none"));
    EXPECT_FALSE(builder.getConfig().structured_output_config.has_value());
}

TEST(QwenmonsterGenerationPolicy, UnknownNamedAndEmptyRequiredFailClosed) {
    auto req = makeRequest("missing");
    GenerationConfigBuilder named({}, "qwen3coder", false, DecodingMethod::STANDARD);
    EXPECT_THROW(named.parseConfigFromRequest(req), std::invalid_argument);
    OpenAIRequest empty;
    empty.toolChoice = "required";
    GenerationConfigBuilder required({}, "qwen3coder", true, DecodingMethod::STANDARD);
    EXPECT_THROW(required.parseConfigFromRequest(empty), std::invalid_argument);
}

TEST(QwenmonsterGenerationPolicy, SchemaAndNameValidation) {
    auto req = makeRequest("required");
    req.toolNameSchemaMap.emplace(
        "bad<name", ToolSchemaWrapper{nullptr, R"({"type":"object","properties":{}})"});
    GenerationConfigBuilder badName({}, "qwen3coder", false, DecodingMethod::STANDARD);
    EXPECT_THROW(badName.parseConfigFromRequest(req), std::invalid_argument);
    req = makeRequest("required");
    req.toolNameSchemaMap["weather"] = ToolSchemaWrapper{nullptr, ""};
    GenerationConfigBuilder badSchema({}, "qwen3coder", false, DecodingMethod::STANDARD);
    EXPECT_THROW(badSchema.parseConfigFromRequest(req), std::invalid_argument);
}

TEST(QwenmonsterGenerationPolicy, RejectsResponseFormatConflict) {
    auto req = makeRequest("required");
    req.responseFormat = R"({"type":"structural_tag","format":{"type":"json_object"}})";
    GenerationConfigBuilder builder({}, "qwen3coder", true, DecodingMethod::STANDARD);
    EXPECT_THROW(builder.parseConfigFromRequest(req), std::invalid_argument);
}

TEST(QwenmonsterGenerationPolicy, ParserFamilyIsolation) {
    auto req = makeRequest("required");
    GenerationConfigBuilder generic({}, "", false, DecodingMethod::STANDARD);
    EXPECT_NO_THROW(generic.parseConfigFromRequest(req));
    EXPECT_FALSE(generic.requiresValidStructuredOutput());
    GenerationConfigBuilder hermes({}, "hermes3", false, DecodingMethod::STANDARD);
    EXPECT_NO_THROW(hermes.parseConfigFromRequest(req));
    GenerationConfigBuilder gemma({}, "gemma4", false, DecodingMethod::STANDARD);
    EXPECT_NO_THROW(gemma.parseConfigFromRequest(req));
    EXPECT_TRUE(gemma.requiresValidStructuredOutput());
    const auto* g = tag(gemma.getConfig());
    ASSERT_NE(g, nullptr);
    const auto* list = std::get_if<std::shared_ptr<Structured::TagsWithSeparator>>(g);
    ASSERT_NE(list, nullptr);
    ASSERT_TRUE(*list);
    EXPECT_TRUE((*list)->at_least_one);
    ASSERT_FALSE((*list)->tags.empty());
    const auto* grammar = std::get_if<Structured::EBNF>(&(*list)->tags[0].content);
    ASSERT_NE(grammar, nullptr);
    EXPECT_NE(grammar->value.find("[ \\n\\t]{0,2}"), std::string::npos);
    EXPECT_EQ(grammar->value.find("[ \\n\\t]*"), std::string::npos);
    EXPECT_EQ(grammar->value.find("max_whitespace_cnt"), std::string::npos);
    const std::string json = (*list)->to_json();
    EXPECT_EQ(json.find("\"type\": \"json_schema\""), std::string::npos);
    EXPECT_NE(json.find("<|tool_call>call:weather"), std::string::npos);
    EXPECT_NE(json.find("<|tool_call>call:clock"), std::string::npos);
}
