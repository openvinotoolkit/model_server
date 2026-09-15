// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.

#include <string>

#include <gtest/gtest.h>

#include "src/llm/io_processing/chat_template/analyzer.hpp"

using namespace ovms;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Semantic oracle, not a vendored copy of the full Google template.
// Provenance: google/gemma-4-31B-it chat_template.jinja @ 68abe48
// Visible HF title at verification time:
//   fix: chat template — null handling, reasoning preservation,
//   turn-tag balance, input validation (#118)
const std::string kGoogleGemma4CanonicalSemanticSnippet = R"JINJA(
{{ '<|tool_call>call:' }}
{% set ns = namespace(prev_message_type=None, prev_non_tool_role=None) %}
{% set enable_thinking = enable_thinking | default(false) %}
{% set preserve_thinking = preserve_thinking | default(false) %}
{% if 'response' in tool_data['function'] %},response:{type:<|"|>OBJECT<|"|>}{% endif %}
{% if argument is none %}{{ 'null' }}{% endif %}
{% set thinking_gate = preserve_thinking and message.get('tool_calls') %}
{% if function['arguments'] is mapping %}{{ function['arguments'] }}
{% elif function['arguments'] is none %}
{% else %}{{ raise_exception('tool_calls[].function.arguments must be a JSON object (mapping), not a string') }}{% endif %}
{% if ns.prev_message_type == 'tool_response' and enable_thinking %}{{ '<|channel>thought\n' }}{% endif %}
)JINJA";

// Local OVMS compatibility oracle. This intentionally models the checked-in
// fixture after #4365, not the canonical Google template byte-for-byte.
const std::string kOvmsGemma4CompatibilitySnippet = R"JINJA(
{# Modifications to original chat template: ignore response field from tool definition #}
{{ '<|tool_call>call:' }}
{% if function['arguments'] is mapping %}{{ function['arguments'] }}
{% elif function['arguments'] is string %}{{ function['arguments'] }}{% endif %}
{% for k in range(loop.index0 + 1, loop_messages | length) %}
    {% if loop_messages[k]['role'] == 'tool' %}<|tool_response>response:{{ name }}{}<tool_response|>{% endif %}
{% endfor %}
)JINJA";

}  // namespace

TEST(Gemma4GoogleJinjaContractTest, PinnedGoogleSemanticOracleCarriesCanonicalWireFeatures) {
    const auto result = ChatTemplateAnalyzer::analyze(kGoogleGemma4CanonicalSemanticSnippet);

    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "gemma4");
    EXPECT_TRUE(result.caps.supportsToolCalls);
    EXPECT_TRUE(result.caps.requiresObjectArguments);

    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "prev_non_tool_role"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "preserve_thinking"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "argument is none"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "'null'"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "'response' in tool_data['function']"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "raise_exception"));
    EXPECT_TRUE(contains(kGoogleGemma4CanonicalSemanticSnippet, "<|channel>thought\\n"));
}

TEST(Gemma4GoogleJinjaContractTest, LocalCompatibilityOracleIsNotMistakenForCanonicalGoogleTemplate) {
    const auto result = ChatTemplateAnalyzer::analyze(kOvmsGemma4CompatibilitySnippet);

    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "gemma4");
    EXPECT_TRUE(result.caps.supportsToolCalls);

    // This is deliberate local compatibility: OpenAI string arguments can be
    // accepted here because OVMS also has adapter tests for string->object
    // conversion. The Google canonical oracle above is stricter.
    EXPECT_FALSE(result.caps.requiresObjectArguments);
    EXPECT_TRUE(contains(kOvmsGemma4CompatibilitySnippet, "function['arguments'] is string"));
    EXPECT_FALSE(contains(kOvmsGemma4CompatibilitySnippet, "argument is none"));
    EXPECT_FALSE(contains(kOvmsGemma4CompatibilitySnippet, "prev_non_tool_role"));
    EXPECT_FALSE(contains(kOvmsGemma4CompatibilitySnippet, "preserve_thinking"));
}
