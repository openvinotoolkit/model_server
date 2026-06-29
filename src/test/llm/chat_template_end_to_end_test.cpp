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
            bool probeOk = probeChatTemplateCaps(probeTokenizer, caps);
            if (!probeOk) {
                std::cout << "=== Probe FAILED: minja cannot render tool calls ===" << std::endl;
                caps.supportsToolCalls = false;
                caps.supportsTools = false;
            }
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
    EXPECT_EQ(analysisResult.detectedModelFamily, "gptoss");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    // Note: template embeds current date, so we check key fragments instead of full match
    EXPECT_NE(appliedOutput.find("to=functions.get_weather"), std::string::npos);
    EXPECT_NE(appliedOutput.find("{\"location\": \"Paris\", \"unit\": \"celsius\"}"), std::string::npos);
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
    EXPECT_EQ(analysisResult.detectedModelFamily, "qwen3coder");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

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
    EXPECT_EQ(analysisResult.detectedModelFamily, "gemma4");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    // FIXME: Why is </s> here? because of facebook-opt125?
    std::string expectedOutput = R"(</s><|turn>user
What's the weather in Paris?<turn|>
<|turn>model
<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Example: Qwen3 Coder with tool call containing string arguments
// Minja silently fails for this template because it uses from_json filter
// which is not supported by minja. The probe detects this (raw JSON dump in
// output containing "tool_calls": [) and disables tool call support.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Qwen3Coder_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3coder_instruct.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3 coder instruct template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    // Probe should detect minja silent failure (from_json not supported)
    // and disable tool call support
    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.supportsTools);
}

// =============================================================================
// Example: Phi-4-mini-instruct with tool call containing string arguments
// This template uses message.tool_calls with direct {{ call.function.arguments }}
// rendering. The static analyzer does NOT detect tool support (no known markers),
// so the probe is never invoked. Tool calls still render correctly if manually
// enabled, but the current pipeline won't auto-detect tool support for this variant.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Phi4Mini_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_phi4_mini.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load phi4-mini template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);
    // Analyzer does not detect phi4-mini tool support (no <|tool▁call▁begin|> marker)
    EXPECT_FALSE(caps.supportsToolCalls);
}

// =============================================================================
// Example: Qwen3-4B with tool call containing string arguments
// Uses <tool_call></tool_call> format with tojson for object args.
// Probe detects requiresObjectArguments=true (tojson adds spaces, string args don't).
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Qwen3_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);
    EXPECT_EQ(analysisResult.detectedModelFamily, "hermes3");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Example: Mistral-7B-Instruct-v0.3 with tool call containing string arguments
// Uses [TOOL_CALLS] + [TOOL_RESULTS] format (detected as "devstral" by analyzer).
// Template uses tool_call.function|tojson so object args render natively.
// Probe detects requiresObjectArguments=true and workaround converts args.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Mistral7B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_mistral7b_v03.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load mistral7b-v0.3 template";

    // Mistral requires 9-char alphanumeric tool_call IDs
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"abc123def","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(false);

    ASSERT_TRUE(applySuccess);
    EXPECT_EQ(analysisResult.detectedModelFamily, "devstral");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s>[INST] What's the weather in Paris?[/INST][TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}, "id": "abc123def"}]</s>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}
