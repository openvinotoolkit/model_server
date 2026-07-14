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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

#include "../../llm/io_processing/chat_template/analyzer.hpp"
#include "../../llm/io_processing/chat_template/caps.hpp"
#include "../../llm/io_processing/chat_template/probe.hpp"
#include "../../llm/io_processing/input_processors/chat_template_adapter.hpp"
#include "../../utils/env_guard.hpp"
#include "../platform_utils.hpp"

using namespace ovms;

// Test fixture providing end-to-end: analyze → probe → apply workarounds → apply template
class ChatTemplateEndToEndMinjaTest : public ::testing::Test {
protected:
    // Any tokenizer will do the job, the only thing we need to do is to change chat template content before use
    const std::string& tokenizerPath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m", false);
    const std::string& chatTemplatesPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/chat_templates", false);

    std::string savedLogLevel;

    void SetUp() override {
        const char* prev = std::getenv("OPENVINO_LOG_LEVEL");
        savedLogLevel = prev ? prev : "";
        SetEnvironmentVar("OPENVINO_LOG_LEVEL", "0");
    }

    void TearDown() override {
        if (savedLogLevel.empty()) {
            UnSetEnvironmentVar("OPENVINO_LOG_LEVEL");
        } else {
            SetEnvironmentVar("OPENVINO_LOG_LEVEL", savedLogLevel);
        }
    }

    // --- Inputs (set by each test) ---
    std::string chatTemplate;
    ov::genai::ChatHistory chatHistory;

    // --- Derived state (populated by run()) ---
    ChatTemplateAnalysisResult analysisResult;
    ChatTemplateCaps caps;
    std::string appliedOutput;
    bool exceptionThrownDuringApplication = false;
    bool basicRenderOk = false;

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
        std::cout << "  toolParser: " << analysisResult.detectedToolParser.value_or("(none)") << std::endl;
        std::cout << "  reasoningParser: " << analysisResult.detectedReasoningParser.value_or("(none)") << std::endl;
        std::cout << "  supportsToolCalls: " << caps.supportsToolCalls << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 2: Basic render probe (can minja render this template at all?)
        {
            ov::genai::Tokenizer basicTokenizer(tokenizerPath);
            basicTokenizer.set_chat_template(chatTemplate);
            basicRenderOk = probeChatTemplateBasicRenderMinja(basicTokenizer);
            if (!basicRenderOk) {
                std::cout << "=== Basic Render Probe FAILED: template incompatible with minja ===" << std::endl;
                return;
            }
        }

        // Step 3a: Tool probe (only if template supports tools)
        if (caps.supportsToolCalls) {
            ov::genai::Tokenizer probeTokenizer(tokenizerPath);
            probeTokenizer.set_chat_template(chatTemplate);
            bool probeOk = probeChatTemplateCapsMinja(probeTokenizer, caps);
            if (!probeOk) {
                std::cout << "=== Probe FAILED: minja cannot render tool calls ===" << std::endl;
            }
        }

        // Step 3b: Probe reasoning caps using Python Jinja (same function used in production)
        {
            ov::genai::Tokenizer probeTokenizer(tokenizerPath);
            probeTokenizer.set_chat_template(chatTemplate);
            bool probeOk = probeChatTemplateReasoning(probeTokenizer, caps);
            if (!probeOk) {
                std::cout << "=== Reasoning Probe FAILED: minja cannot render reasoning ===" << std::endl;
            }
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;
        std::cout << "  missnamedReasoningField: " << (caps.missnamedReasoningField.empty() ? "(none)" : caps.missnamedReasoningField) << std::endl;

        // Step 4: Apply workarounds to the chat history
        chat_template_adapter::applyToHistory(caps, chatHistory);

        // Step 5: Apply chat template
        ov::genai::Tokenizer tokenizer(tokenizerPath);
        tokenizer.set_chat_template(chatTemplate);
        try {
            appliedOutput = tokenizer.apply_chat_template(chatHistory, addGenerationPrompt);
            exceptionThrownDuringApplication = false;
        } catch (const std::exception& e) {
            std::cout << "apply_chat_template FAILED: " << e.what() << std::endl;
            exceptionThrownDuringApplication = true;
        }

        std::cout << "=== Result ===" << std::endl;
        std::cout << appliedOutput << std::endl;
    }
};

// =============================================================================
// The chat template we use here contains multiple patches, including one that relates to `string2obj`.
// Without the workaround, it translates to {"key":"val"}.
// With, it would translate to {'key': 'val'} which is not correct.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, GptOss_ToolCallWithStringArgs) {
    // Load the real gpt-oss chat template
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gpt_oss.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load gpt-oss template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "gptoss");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "gptoss");

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|start|>user<|message|>What's the weather in Paris?<|end|><|start|>assistant to=functions.get_weather <|channel|>commentary json<|message|>{"location":"Paris","unit":"celsius"}<|end|><|start|>assistant)";
    EXPECT_NE(appliedOutput.find(expectedOutput), std::string::npos) << appliedOutput;
}

// =============================================================================
// Since minja automatically detects str2obj is needed, both: string and object format works.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Qwen36_ToolCallWithStringArgs) {
    // Load the real qwen chat template
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen36.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen36 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "qwen3");

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
<|im_start|>assistant
<think>
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Ovms detects str2obj workaround is needed and applies workaround.
// Minja does not do it automatically, their probe is broken with gemma.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Gemma4_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gemma.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load gemma template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "gemma4");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "gemma4");

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
// This test is running against chat template which contains our patch with `from_json` filter.
// This filter is unsupported by minja, therefore the test is expected to fail.
// Model stays unsupported with minja chat template mode.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Qwen3Coder_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3coder_instruct.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3 coder instruct template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    // No exception is thrown by minja even though there is unsupported filter
    ASSERT_FALSE(exceptionThrownDuringApplication);
    // Basic probing does not reach the unsupported filter, so it is ok to use it without agentic capabilities
    ASSERT_TRUE(basicRenderOk);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);
}

// =============================================================================
// Chat template taken from Ovms extras, original chat template does not render tools at all.
// Model does not need str2obj workaround.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Phi4Mini_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_phi4_mini.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load phi4-mini template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "phi4");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|system|>
You are a helpful assistant.<|end|><|user|>What's the weather in Paris?<|end|><|assistant|>{"name": "get_weather", "arguments": {"location":"Paris","unit":"celsius"}}<|end|><|assistant|>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// It works either way, with str2obj conversion or not - does not matter.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Qwen3_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "hermes3");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "qwen3");

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
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// It works either way, with str2obj conversion or not - does not matter.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Mistral7B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_mistral7b_v03.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load mistral7b-v0.3 template";

    // Mistral requires 9-char alphanumeric tool_call IDs
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"abc123def","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "mistral");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s>[INST] What's the weather in Paris?[/INST][TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}, "id": "abc123def"}]</s>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// // LFM2 does not render tool calls in current chat template, minja inserts them as stringified JSON, which is not correct.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, LFM2_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm2.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

run(true);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_TRUE(caps.missnamedReasoningField.empty());

    std::string expectedOutput = R"(</s><|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
{
  "tool_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "location": "Paris",
        "unit": "celsius"
      },
      "id": "call_abc123"
    }
  ],
  "content": ""
}<|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

TEST_F(ChatTemplateEndToEndMinjaTest, LFM25_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2.5 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "lfm2");

    EXPECT_TRUE(caps.supportsToolCalls);   
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s><|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_weather(location='Paris', unit='celsius')]<|tool_call_end|><|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

TEST_F(ChatTemplateEndToEndMinjaTest, LFM25_ToolCallWithStringArgsAndReasoning) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2.5 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant", "reasoning_content":"Here is some reasoning content","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "lfm2");

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s><|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>Here is some reasoning content</think><|tool_call_start|>[get_weather(location='Paris', unit='celsius')]<|tool_call_end|><|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Same story as Qwen3-8B, but with image tags.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Qwen3VL_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3vl.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3-vl template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "hermes3");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Same story as Qwen3-8B, but without reasoning.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Qwen3_30B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3_30b.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3-30b template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "hermes3");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Synthetic test: template that throws on basic rendering (e.g. uses undefined
// filter). The basic render probe should catch this and return false.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, BrokenTemplate_BasicRenderFails) {
    // Template uses an undefined filter that causes minja to throw
    chatTemplate = R"({%- for message in messages -%}{{ message.content | undefined_filter }}{%- endfor -%})";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"Hi"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"Hello"})"));

    run(true);

    // Minja silently fails (without exception), but our basic render check should catch it by parsing results.
    ASSERT_FALSE(exceptionThrownDuringApplication);
    EXPECT_FALSE(basicRenderOk);
}
