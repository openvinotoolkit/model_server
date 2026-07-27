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

        // Step 3: Tool probe (only if template supports tools)
        if (caps.supportsToolCalls) {
            ov::genai::Tokenizer probeTokenizer(tokenizerPath);
            probeTokenizer.set_chat_template(chatTemplate);
            bool probeOk = probeChatTemplateCapsMinja(probeTokenizer, caps);
            if (!probeOk) {
                std::cout << "=== Probe FAILED: minja cannot render tool calls ===" << std::endl;
            }
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

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
// // TODO: Implement assertions, where to take lfm2 deployments steps from?
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, LFM2_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm2.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run(true);

    // TODO: Implement assertions, where to take lfm2 deployments steps from?
}

// =============================================================================
// Minja can't handle this chat template for some reason.
// TODO(przepeck): ensure this tests the same template as we will publish to HF
// =============================================================================
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
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    // TODO: It just does not work for now, documented with assertion

    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);  // TODO(przepeck): change once we have it working

    // TODO: Expect appliedOutput once fixed
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
// Onyx (early preview model) chat template. Unlike every other template in this
// suite, Onyx's own Jinja template does not consume the standard OpenAI
// "tool_calls" list at all -- it only reads message['content'] (a plain string)
// and an Onyx-specific message['recipient'] field (e.g. "functions.get_weather",
// "self", "user"). Feeding it a standard tool_calls-shaped assistant message
// therefore renders an effectively empty assistant turn.
// ChatTemplateAnalyzer now recognizes Onyx's control tokens (see analyzer.cpp) and
// sets detectedToolParser/detectedReasoningParser, but deliberately leaves
// caps.supportsToolCalls false since the template can't natively round-trip an
// OpenAI tool_calls history (demonstrated by this very test), so no input-side
// workaround is applied either -- detectedToolParser only affects output parsing.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Onyx_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_onyx.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load onyx template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "onyx");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "onyx");

    EXPECT_FALSE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);

    // The template itself never reads "tool_calls" (it only looks at
    // message['content'] and message['recipient']). Because caps.supportsToolCalls
    // is false here, OVMS does not apply its own tool-call workaround either.
    // Minja's own generic fallback (used for templates it detects have no native
    // tool-call rendering) kicks in instead and serializes the whole message
    // (tool_calls + content) as a JSON blob into message['content'] -- the
    // function name/args are NOT lost, but they end up as raw, unparsed JSON text
    // rather than in Onyx's native " to=functions.<name>" / <|eom|> framing.
    std::string expectedOutput = R"(</s><|start|>system<|message|>You are a helpful assistant.<|eot|><|start|>user<|message|>What's the weather in Paris?<|eot|><|start|>assistant<|message|>{
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
}<|eot|><|start|>assistant)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Onyx's own message shape: instead of the OpenAI "tool_calls" array, the
// assistant turn carries a "recipient" field (here "functions.get_weather")
// and a plain-string content holding the raw JSON arguments. This is the shape
// Onyx's template actually understands, ending the turn with "<|eom|>" (a
// continuation marker) rather than "<|eot|>".
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Onyx_ToolCallWithRecipientField) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_onyx.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load onyx template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","recipient":"functions.get_weather","content":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    std::string expectedOutput = R"(</s><|start|>system<|message|>You are a helpful assistant.<|eot|><|start|>user<|message|>What's the weather in Paris?<|eot|><|start|>assistant to=functions.get_weather<|message|>{"location":"Paris","unit":"celsius"}<|eom|><|start|>assistant)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Full-scope round trip: exercises every message shape the Onyx template
// natively understands in a single history, not just one shape in isolation --
// user prompt -> assistant tool call (recipient=functions.<name>, continuation
// "<|eom|>") -> tool call response (role="tool") -> assistant final answer
// (recipient="user", "<|eot|>").
//
// This surfaces a second, previously undocumented gap alongside the tool_calls
// one above (see muse/chat_template_issues.md): because caps.supportsToolCalls
// is false for Onyx, ChatTemplateAdapter's generic fallback also intercepts
// plain role="tool" messages -- not just assistant tool_calls -- and rewrites
// them into a synthetic role="user" message serializing {tool, content} as a
// JSON blob, rather than passing them through to the template's own native
// "tool"-role branch (which expects message['name'] + message['content'] and
// would render "<|start|>tool <name><|message|>...<|eot|>"). So even a chat
// history built entirely out of Onyx's own native fields (recipient) still
// does not round-trip once a plain OpenAI-shaped tool response message is
// mixed in.
// =============================================================================
TEST_F(ChatTemplateEndToEndMinjaTest, Onyx_FullMultiTurnToolCallRoundTrip) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_onyx.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load onyx template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","recipient":"functions.get_weather","content":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"tool","name":"get_weather","content":"{\"temperature\":15,\"unit\":\"celsius\"}"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","recipient":"user","content":"It's 15C in Paris."})"));

    run(true);

    ASSERT_FALSE(exceptionThrownDuringApplication);

    EXPECT_FALSE(caps.supportsToolCalls);

    // Known gap (see class comment above): the "tool" message is NOT rendered via
    // the template's native "<|start|>tool <name><|message|>...<|eot|>" branch --
    // ChatTemplateAdapter's fallback rewrites it into a synthetic user message
    // carrying a JSON blob first.
    std::string expectedOutput =
        R"(</s><|start|>system<|message|>You are a helpful assistant.<|eot|>)"
        R"(<|start|>user<|message|>What's the weather in Paris?<|eot|>)"
        R"(<|start|>assistant to=functions.get_weather<|message|>{"location":"Paris","unit":"celsius"}<|eom|>)"
        "<|start|>user<|message|>{\n"
        "  \"tool_response\": {\n"
        "    \"tool\": \"get_weather\",\n"
        "    \"content\": \"{\\\"temperature\\\":15,\\\"unit\\\":\\\"celsius\\\"}\"\n"
        "  }\n"
        "}<|eot|>"
        R"(<|start|>assistant to=user<|message|>It's 15C in Paris.<|eot|>)"
        R"(<|start|>assistant)";
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
