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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

#include "../../llm/chat_template_analyzer.hpp"
#include "../../llm/chat_template_caps.hpp"
#include "../../llm/chat_template_probe.hpp"
#include "../../llm/input_workarounds.hpp"
#include "../platform_utils.hpp"

using namespace ovms;

// Chat template applicator type
enum class TemplateApplicator {
    MINJA,
    JINJA  // Not implemented yet
};

// Test fixture providing end-to-end: analyze → probe → apply workarounds → apply template
class ChatTemplateEndToEndTest : public ::testing::Test {
protected:
    const std::string& tokenizerPath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m", false);
    const std::string& chatTemplatesPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/chat_templates", false);

    std::string savedLogLevel;

    void SetUp() override {
        const char* prev = std::getenv("OPENVINO_LOG_LEVEL");
        savedLogLevel = prev ? prev : "";
        setenv("OPENVINO_LOG_LEVEL", "0", 1);
    }

    void TearDown() override {
        if (savedLogLevel.empty()) {
            unsetenv("OPENVINO_LOG_LEVEL");
        } else {
            setenv("OPENVINO_LOG_LEVEL", savedLogLevel.c_str(), 1);
        }
    }

    // --- Inputs (set by each test) ---
    std::string chatTemplate;
    TemplateApplicator applicator = TemplateApplicator::MINJA;
    ov::genai::ChatHistory chatHistory;

    // --- Derived state (populated by run()) ---
    ChatTemplateAnalysisResult analysisResult;
    ChatTemplateCaps caps;
    std::string appliedOutput;
    bool applySuccess = false;

    // Load template from file
    static std::string loadTemplateFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    }

    // Run the full pipeline: analyze → probe → workarounds → apply
    void run(bool addGenerationPrompt = true) {
        ASSERT_FALSE(chatTemplate.empty()) << "chatTemplate must be set before calling run()";
        ASSERT_FALSE(chatHistory.empty()) << "chatHistory must be set before calling run()";

        // Step 1: Static analysis
        analysisResult = ChatTemplateAnalyzer::analyze(chatTemplate);
        caps = analysisResult.caps;

        std::cout << "=== Analysis ===" << std::endl;
        std::cout << "  modelFamily: " << analysisResult.detectedModelFamily << std::endl;
        std::cout << "  toolParser: " << analysisResult.detectedToolParser.value_or("(none)") << std::endl;
        std::cout << "  reasoningParser: " << analysisResult.detectedReasoningParser.value_or("(none)") << std::endl;
        std::cout << "  supportsToolCalls: " << caps.supportsToolCalls << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;
        std::cout << "  requiresNonNullContent: " << caps.requiresNonNullContent << std::endl;

        // Step 2: Probe (only if template supports tools)
        if (caps.supportsToolCalls) {
            ov::genai::Tokenizer probeTokenizer(tokenizerPath);
            probeTokenizer.set_chat_template(chatTemplate);
            probeChatTemplateCaps(probeTokenizer, caps);
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 3: Apply workarounds to the chat history
        if (applicator == TemplateApplicator::MINJA) {
            input_workarounds::applyToHistory(caps, analysisResult.detectedModelFamily, chatHistory);
        } else {
            GTEST_SKIP() << "JINJA applicator not implemented yet";
        }

        // Step 4: Apply chat template
        if (applicator == TemplateApplicator::MINJA) {
            ov::genai::Tokenizer tokenizer(tokenizerPath);
            tokenizer.set_chat_template(chatTemplate);
            try {
                appliedOutput = tokenizer.apply_chat_template(chatHistory, addGenerationPrompt);
                applySuccess = true;
            } catch (const std::exception& e) {
                std::cout << "apply_chat_template FAILED: " << e.what() << std::endl;
                applySuccess = false;
            }
        }

        std::cout << "=== Result ===" << std::endl;
        std::cout << appliedOutput << std::endl;
    }
};

// =============================================================================
// Example: gpt-oss-20b with tool call containing string arguments
// The probe should detect requiresObjectArguments=true, workaround should convert
// string args to object, and the final template should render them natively.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, GptOss_ToolCallWithStringArgs) {
    // Load the real gpt-oss chat template
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gpt_oss.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load gpt-oss template";

    // Simulate a request with tool_calls where arguments are a JSON string
    // (as sent by most OpenAI-compatible clients)
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);

    std::string expectedOutput = R"(<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-06-25

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>What's the weather in Paris?<|end|><|start|>assistant to=functions.get_weather <|channel|>commentary json<|message|>{"location": "Paris", "unit": "celsius"}<|end|>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Example: Qwen3.6-35B-A3B-int4-ov with tool call containing string arguments
// The probe should detect requiresObjectArguments=true, workaround should convert
// string args to object, and the final template should render them natively.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Qwen36_ToolCallWithStringArgs) {
    // Load the real qwen chat template
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen36.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen36 template";

    // Simulate a request with tool_calls where arguments are a JSON string
    // (as sent by most OpenAI-compatible clients)
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);

    std::string expectedOutput = R"(<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call><|im_end|>
)";
     EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Example: Gemma4 with tool call containing string arguments
// The probe should detect requiresObjectArguments=true via the needle:<| pattern,
// workaround should convert string args to object, and template should render
// them in Gemma's native key:<|"|>value<|"|> format.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Gemma4_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gemma.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load gemma template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);

    // FIXME: Why is </s> here? because of facebook-opt125?
    std::string expectedOutput = R"(</s><|turn>user
What's the weather in Paris?<turn|>
<|turn>model
<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)";
     EXPECT_EQ(appliedOutput, expectedOutput);
}
