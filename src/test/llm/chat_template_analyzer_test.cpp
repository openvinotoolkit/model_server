//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "../../llm/io_processing/chat_template/analyzer.hpp"
#include "../platform_utils.hpp"

using namespace ovms;

class ChatTemplateAnalyzerTest : public ::testing::Test {
protected:
    const std::string& chatTemplatesPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/chat_templates", false);

    std::string loadTemplate(const std::string& filename) {
        std::ifstream file(chatTemplatesPath + "/" + filename);
        if (!file.is_open()) {
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    }
};

// --- Empty template ---

TEST_F(ChatTemplateAnalyzerTest, emptyTemplateReturnsDefaults) {
    auto result = ChatTemplateAnalyzer::analyze("");
    EXPECT_FALSE(result.detectedToolParser.has_value());
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_FALSE(result.caps.supportsToolCalls);
    EXPECT_FALSE(result.caps.requiresObjectArguments);
}

// --- GPT-OSS ---

TEST_F(ChatTemplateAnalyzerTest, detectsGptOss) {
    std::string tmpl = loadTemplate("chat_template_gpt_oss.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "gptoss");
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "gptoss");
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Gemma4 ---

TEST_F(ChatTemplateAnalyzerTest, detectsGemma4) {
    std::string tmpl = loadTemplate("chat_template_gemma.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "gemma4");
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "gemma4");
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Qwen3-Coder ---

TEST_F(ChatTemplateAnalyzerTest, detectsQwen3Coder) {
    std::string tmpl = loadTemplate("chat_template_qwen3coder_instruct.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "qwen3coder");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- LFM2 ---

TEST_F(ChatTemplateAnalyzerTest, detectsLfm25) {
    std::string tmpl = loadTemplate("chat_template_lfm25.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "lfm2");
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "lfm2");
    EXPECT_TRUE(result.caps.supportsToolCalls);
    EXPECT_EQ(result.caps.missnamedReasoningField, "thinking");
}

// --- Phi-4 ---

TEST_F(ChatTemplateAnalyzerTest, detectsPhi4) {
    std::string tmpl = loadTemplate("chat_template_phi4_mini.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "phi4");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Devstral ---

TEST_F(ChatTemplateAnalyzerTest, detectsDevstral) {
    std::string tmpl = loadTemplate("chat_template_devstral.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "devstral");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Mistral ---

TEST_F(ChatTemplateAnalyzerTest, detectsMistral) {
    std::string tmpl = loadTemplate("chat_template_mistral7b_v03.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "mistral");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Llama3 (no template file available — inline: TODO: Where to take it from?) ---

TEST_F(ChatTemplateAnalyzerTest, detectsLlama3) {
    std::string tmpl = R"({% if tool_calls %}<|python_tag|>{{ tool_calls }}{% endif %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "llama3");
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Hermes3/Qwen ---

TEST_F(ChatTemplateAnalyzerTest, detectsQwen3AsHermes3WithReasoning) {
    std::string tmpl = loadTemplate("chat_template_qwen3.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "hermes3");
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

TEST_F(ChatTemplateAnalyzerTest, detectsQwen36AsQwen3CoderWithReasoning) {
    std::string tmpl = loadTemplate("chat_template_qwen36.jinja");
    ASSERT_FALSE(tmpl.empty());
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "qwen3coder");
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
    EXPECT_TRUE(result.caps.supportsToolCalls);
}

// --- Reasoning-only (inline — no matching file) ---

TEST_F(ChatTemplateAnalyzerTest, detectsReasoningOnlyWithThinkTags) {
    std::string tmpl = R"({% if reasoning %}<think>{{ reasoning }}</think>{% endif %} no tool call markers)";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_FALSE(result.detectedToolParser.has_value());
    ASSERT_TRUE(result.detectedReasoningParser.has_value());
    EXPECT_EQ(result.detectedReasoningParser.value(), "qwen3");
}

// --- No detection ---

TEST_F(ChatTemplateAnalyzerTest, unknownTemplateReturnsEmpty) {
    std::string tmpl = R"({% for message in messages %}{{ message.content }}{% endfor %})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    EXPECT_FALSE(result.detectedToolParser.has_value());
    EXPECT_FALSE(result.detectedReasoningParser.has_value());
    EXPECT_FALSE(result.caps.supportsToolCalls);
}

// --- Priority: Devstral over Mistral (inline — tests specific precedence logic) ---

TEST_F(ChatTemplateAnalyzerTest, devstralTakesPriorityOverMistral) {
    std::string tmpl = R"([TOOL_CALLS]{{ name }}[ARGS]{{ args }}[TOOL_RESULTS]{{ results }})";
    auto result = ChatTemplateAnalyzer::analyze(tmpl);
    ASSERT_TRUE(result.detectedToolParser.has_value());
    EXPECT_EQ(result.detectedToolParser.value(), "devstral");
}

// --- Capabilities struct defaults ---

TEST_F(ChatTemplateAnalyzerTest, defaultCapsValues) {
    ChatTemplateCaps caps;
    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_TRUE(caps.missnamedReasoningField.empty());
}
