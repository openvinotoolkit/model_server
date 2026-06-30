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

#include <string>

#include <gtest/gtest.h>

#include "../../llm/chat_template_analyzer.hpp"

using namespace ovms;

class ChatTemplateAnalyzerTest : public ::testing::Test {};

// --- Empty template ---

TEST_F(ChatTemplateAnalyzerTest, emptyTemplateReturnsDefaults) {
    auto result = ChatTemplateAnalyzer::analyze("");
    EXPECT_TRUE(result.detectedModelFamily.empty());
    EXPECT_FALSE(result.detectedToolParser.has_value());
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_FALSE(result.caps.supportsToolCalls);
    EXPECT_FALSE(result.caps.requiresObjectArguments);
}

// --- GPT-OSS ---

TEST_F(ChatTemplateAnalyzerTest, detectsGptOss) {
    std::string tmpl = R"({% if message.role == 'assistant' %}<|channel|>{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "gptoss");
    EXPECT_EQ(result.detectedToolParser.value(), "gptoss");
    EXPECT_EQ(result.detectedReasoningParser.value(), "gptoss");
    EXPECT_TRUE(result.caps.supportsToolCalls);
    EXPECT_TRUE(result.caps.supportsTools);
    EXPECT_TRUE(result.caps.supportsToolResponses);
}

// --- Gemma4 ---

TEST_F(ChatTemplateAnalyzerTest, detectsGemma4SingleQuote) {
    std::string tmpl = R"({% if tool_call %}'<|tool_call>call:'{{ tool_call.name }}{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "gemma4");
    EXPECT_EQ(result.detectedToolParser.value(), "gemma4");
    EXPECT_EQ(result.detectedReasoningParser.value(), "gemma4");
    EXPECT_TRUE(result.caps.requiresObjectArguments);
}

TEST_F(ChatTemplateAnalyzerTest, detectsGemma4NoQuote) {
    std::string tmpl = R"({% if tool_call %}<|tool_call>call:{{ tool_call.name }}{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "gemma4");
}

// --- Qwen3-Coder ---

TEST_F(ChatTemplateAnalyzerTest, detectsQwen3Coder) {
    std::string tmpl = R"(<function={{ func.name }}>{% for param in func.params %}<parameter={{ param.name }}>{{ param.value }}</parameter>{% endfor %}</function>)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "qwen3coder");
    EXPECT_EQ(result.detectedToolParser.value(), "qwen3coder");
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

TEST_F(ChatTemplateAnalyzerTest, detectsQwen3CoderWithThinkTags) {
    std::string tmpl = R"(<function={{ func.name }}><parameter={{ p.name }}>{{ p.value }}</parameter></function><think>reasoning</think>)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "qwen3coder");
    EXPECT_EQ(result.detectedToolParser.value(), "qwen3coder");
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

// --- LFM2 ---

TEST_F(ChatTemplateAnalyzerTest, detectsLfm2AssistantToolCall) {
    std::string tmpl = R"({% if role == 'assistant' %}<|assistant_tool_call|>{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "lfm2");
    EXPECT_EQ(result.detectedToolParser.value(), "lfm2");
}

TEST_F(ChatTemplateAnalyzerTest, detectsLfm2ToolCallStart) {
    std::string tmpl = R"(<|tool_call_start|>{{ tool_calls }}<|tool_call_end|>)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "lfm2");
    EXPECT_EQ(result.detectedToolParser.value(), "lfm2");
}

// --- Phi-4 ---

TEST_F(ChatTemplateAnalyzerTest, detectsPhi4Functools) {
    std::string tmpl = R"(prefix function calls with functools marker functools[{"name": "fn"}])";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "phi4");
    EXPECT_EQ(result.detectedToolParser.value(), "phi4");
}

// --- Devstral ---

TEST_F(ChatTemplateAnalyzerTest, detectsDevstral) {
    std::string tmpl = R"({% if tool_calls %}[TOOL_CALLS]{{ name }}[ARGS]{{ args }}{% endif %}{% if tool_result %}[TOOL_RESULTS]{{ result }}{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "devstral");
    EXPECT_EQ(result.detectedToolParser.value(), "devstral");
}

// --- Mistral ---

TEST_F(ChatTemplateAnalyzerTest, detectsMistralWithToolCalls) {
    std::string tmpl = R"({% if tool_calls %}[TOOL_CALLS]{{ tool_calls }}{% endif %} some other stuff)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "mistral");
    EXPECT_EQ(result.detectedToolParser.value(), "mistral");
}

TEST_F(ChatTemplateAnalyzerTest, detectsMistralWithAvailableTools) {
    std::string tmpl = R"([AVAILABLE_TOOLS]{{ tools }}[/AVAILABLE_TOOLS] template body)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "mistral");
    EXPECT_EQ(result.detectedToolParser.value(), "mistral");
}

// --- Llama3 ---

TEST_F(ChatTemplateAnalyzerTest, detectsLlama3) {
    std::string tmpl = R"({% if tool_calls %}<|python_tag|>{{ tool_calls }}{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "llama3");
    EXPECT_EQ(result.detectedToolParser.value(), "llama3");
    EXPECT_TRUE(result.caps.requiresNonNullContent);
}

// --- Hermes3/Qwen ---

TEST_F(ChatTemplateAnalyzerTest, detectsHermes3) {
    std::string tmpl = R"({% if tool_call %}<tool_call>{{ tool_call }}</tool_call>{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "hermes3");
    EXPECT_EQ(result.detectedToolParser.value(), "hermes3");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
}

TEST_F(ChatTemplateAnalyzerTest, detectsHermes3WithQwen3Reasoning) {
    std::string tmpl = R"(<tool_call>{{ tool_call }}</tool_call> and also <think>reasoning</think>)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "hermes3");
    EXPECT_EQ(result.detectedToolParser.value(), "hermes3");
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

TEST_F(ChatTemplateAnalyzerTest, detectsHermes3WithContentSplitThink) {
    std::string tmpl = R"(<tool_call>{{ tool_call }}</tool_call> and content.split('</think>') logic)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "hermes3");
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

// --- Reasoning-only ---

TEST_F(ChatTemplateAnalyzerTest, detectsReasoningOnlyWithThinkTags) {
    std::string tmpl = R"({% if reasoning %}<think>{{ reasoning }}</think>{% endif %} no tool call markers)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_TRUE(result.detectedModelFamily.empty());
    EXPECT_FALSE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

TEST_F(ChatTemplateAnalyzerTest, detectsReasoningOnlyWithContentSplit) {
    std::string tmpl = R"(some template with content.split('</think>') logic)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_TRUE(result.detectedModelFamily.empty());
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

// --- No detection ---

TEST_F(ChatTemplateAnalyzerTest, unknownTemplateReturnsEmpty) {
    std::string tmpl = R"({% for message in messages %}{{ message.content }}{% endfor %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_TRUE(result.detectedModelFamily.empty());
    EXPECT_FALSE(result.detectedToolParser.has_value());
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_FALSE(result.caps.supportsToolCalls);
}

// --- Priority: Devstral over Mistral ---

TEST_F(ChatTemplateAnalyzerTest, devstralTakesPriorityOverMistral) {
    // Both have [TOOL_CALLS] but Devstral also has [ARGS]
    std::string tmpl = R"([TOOL_CALLS]{{ name }}[ARGS]{{ args }}[TOOL_RESULTS]{{ results }})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_EQ(result.detectedModelFamily, "devstral");
    EXPECT_EQ(result.detectedToolParser.value(), "devstral");
}

// --- Capabilities struct defaults ---

TEST_F(ChatTemplateAnalyzerTest, defaultCapsValues) {
    ChatTemplateCaps caps;
    EXPECT_TRUE(caps.supportsSystemRole);
    EXPECT_FALSE(caps.supportsTools);
    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.supportsToolResponses);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);
}
