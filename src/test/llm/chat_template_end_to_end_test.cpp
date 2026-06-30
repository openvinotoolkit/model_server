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
    JINJA  // Implemented in separate translation unit
};

// Test fixture providing end-to-end: analyze → probe → apply workarounds → apply template
class ChatTemplateEndToEndTest : public ::testing::Test {
protected:

    // Any tokenizer will do the job, the only thing we need to do is to do is to change chat template content before use
    // TODO: Check if should dump custom template into tokenizer directory before test runs
    // Minja likes to do some work during initialization and we need to ensure it does that with the correct template content
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
        std::cout << "  modelFamily: " << analysisResult.detectedModelFamily << std::endl;
        std::cout << "  toolParser: " << analysisResult.detectedToolParser.value_or("(none)") << std::endl;
        std::cout << "  reasoningParser: " << analysisResult.detectedReasoningParser.value_or("(none)") << std::endl;
        std::cout << "  supportsToolCalls: " << caps.supportsToolCalls << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;
        std::cout << "  requiresNonNullContent: " << caps.requiresNonNullContent << std::endl;

        // Step 2: Basic render probe (can minja render this template at all?)
        {
            ov::genai::Tokenizer basicTokenizer(tokenizerPath);
            basicTokenizer.set_chat_template(chatTemplate);
            basicRenderOk = probeChatTemplateBasicRender(basicTokenizer);
            if (!basicRenderOk) {
                std::cout << "=== Basic Render Probe FAILED: template incompatible with minja ===" << std::endl;
                return;
            }
        }

        // Step 3: Tool probe (only if template supports tools)
        if (caps.supportsToolCalls) {
            ov::genai::Tokenizer probeTokenizer(tokenizerPath);
            probeTokenizer.set_chat_template(chatTemplate);
            bool probeOk = probeChatTemplateCaps(probeTokenizer, caps);
            if (!probeOk) {
                std::cout << "=== Probe FAILED: minja cannot render tool calls ===" << std::endl;
            }
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 4: Apply workarounds to the chat history
        if (applicator == TemplateApplicator::MINJA) {
            input_workarounds::applyToHistory(caps, analysisResult.detectedModelFamily, chatHistory);
        } else {
            GTEST_SKIP() << "JINJA applicator not implemented yet";
        }

        // Step 5: Apply chat template
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
// The probe should detect requiresObjectArguments=false, workaround should not be applied
// applied, and the final template should render them natively.
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

    run(true);

    ASSERT_TRUE(applySuccess);

    EXPECT_EQ(analysisResult.detectedModelFamily, "gptoss");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "gptoss");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "gptoss");

    EXPECT_TRUE(caps.supportsTools);
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);

    std::string expectedOutput = R"(<|start|>user<|message|>What's the weather in Paris?<|end|><|start|>assistant to=functions.get_weather <|channel|>commentary json<|message|>{"location":"Paris","unit":"celsius"}<|end|><|start|>assistant)";
    EXPECT_NE(appliedOutput.find(expectedOutput), std::string::npos) << appliedOutput;
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

    run(true);

    ASSERT_TRUE(applySuccess);

    EXPECT_EQ(analysisResult.detectedModelFamily, "qwen3coder");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "qwen3");

    EXPECT_TRUE(caps.supportsTools);
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_TRUE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);

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

    run(true);

    ASSERT_TRUE(applySuccess);

    EXPECT_EQ(analysisResult.detectedModelFamily, "gemma4");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "gemma4");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "gemma4");

    EXPECT_TRUE(caps.supportsTools);
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_TRUE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);

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

    run(true);

    
    ASSERT_TRUE(applySuccess);  // here we dont block people from using such templates that mis-render requests
    ASSERT_TRUE(basicRenderOk);  // only tool rendering is broken

    EXPECT_EQ(analysisResult.detectedModelFamily, "qwen3coder");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());
    
    // This is bug, how to fix?
    // TODO: "| from_json" patching?
    EXPECT_FALSE(caps.supportsTools);
    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);
}

// =============================================================================
// Example: Phi-4-mini-instruct with tool call containing string arguments
// This template uses message.tool_calls with direct {{ call.function.arguments }}
// rendering. The static analyzer detects phi4 via the "functools" marker.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Phi4Mini_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_phi4_mini.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load phi4-mini template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);
    // FIXME:
    // The model doesnt render available tools
    // The model renders tool calls
    // However we have it in agentic demo so I keep it here for documentation 

    // It only works when chat template is taken from our extras

    ASSERT_TRUE(applySuccess);

    EXPECT_EQ(analysisResult.detectedModelFamily, "phi4");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "phi4");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsTools);
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);

    std::string expectedOutput = R"(<|system|>
You are a helpful assistant.<|end|><|user|>What's the weather in Paris?<|end|><|assistant|>{"name": "get_weather", "arguments": {"location":"Paris","unit":"celsius"}}<|end|><|assistant|>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
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

    run(true);

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
<|im_start|>assistant
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

    run(true);

    ASSERT_TRUE(applySuccess);
    EXPECT_EQ(analysisResult.detectedModelFamily, "devstral");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s>[INST] What's the weather in Paris?[/INST][TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}, "id": "abc123def"}]</s>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Example: LFM2-24B-A2B with tool call containing string arguments
// Template only injects tools into the system prompt ("List of tools: [...]")
// and has no tool_call rendering logic — assistant messages render content only.
// Analyzer does not detect tool support, so probe is never invoked.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, LFM2_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm2.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_TRUE(applySuccess);
    // Analyzer does not detect tool support (no tool_call markers in template)
    EXPECT_FALSE(caps.supportsToolCalls);
}

// =============================================================================
// Example: LFM2.5 with tool call containing string arguments
// Uses <|tool_call_start|>[func(arg=val)]<|tool_call_end|> format.
// Template uses {%- generation -%} blocks. Basic render works (simple messages
// are fine), but tool_calls rendering silently fails — minja dumps raw JSON
// instead of calling render_tool_calls macro. Tool probe catches this.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, LFM25_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2.5 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    // Basic render works, but tool probe detects minja silent failure
    EXPECT_TRUE(basicRenderOk);
    EXPECT_EQ(analysisResult.detectedModelFamily, "lfm2");
    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.supportsTools);
}

// =============================================================================
// Example: Qwen3-VL-8B-Instruct with tool call containing string arguments
// Uses <tool_call>{"name": ..., "arguments": ...}</tool_call> format.
// Template handles both string and object arguments natively (has is_string check).
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Qwen3VL_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3vl.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3-vl template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_TRUE(applySuccess);
    EXPECT_TRUE(caps.supportsToolCalls);

    EXPECT_NE(appliedOutput.find("<tool_call>"), std::string::npos);
    EXPECT_NE(appliedOutput.find("get_weather"), std::string::npos);
    EXPECT_NE(appliedOutput.find("</tool_call>"), std::string::npos);
}

// =============================================================================
// Example: Qwen3-30B-A3B-Instruct-2507 with tool call containing string arguments
// Uses <tool_call>{"name": ..., "arguments": ...}</tool_call> format.
// Template handles both string and object arguments natively (has is_string check).
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, Qwen3_30B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3_30b.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load qwen3-30b template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    ASSERT_TRUE(applySuccess);
    EXPECT_TRUE(caps.supportsToolCalls);

    EXPECT_NE(appliedOutput.find("<tool_call>"), std::string::npos);
    EXPECT_NE(appliedOutput.find("get_weather"), std::string::npos);
    EXPECT_NE(appliedOutput.find("</tool_call>"), std::string::npos);
}

// =============================================================================
// Synthetic test: template that throws on basic rendering (e.g. uses undefined
// filter). The basic render probe should catch this and return false.
// =============================================================================
TEST_F(ChatTemplateEndToEndTest, BrokenTemplate_BasicRenderFails) {
    // Template uses an undefined filter that causes minja to throw
    chatTemplate = R"({%- for message in messages -%}{{ message.content | undefined_filter }}{%- endfor -%})";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"Hi"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"Hello"})"));

    run(true);

    EXPECT_FALSE(basicRenderOk);
}
