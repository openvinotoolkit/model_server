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
#include <memory>
#include <string>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

#include <openvino/genai/chat_history.hpp>

#include "../../llm/io_processing/chat_template/analyzer.hpp"
#include "../../llm/io_processing/chat_template/caps.hpp"
#include "../../llm/io_processing/chat_template/probe.hpp"
#include "../../llm/io_processing/input_processors/chat_template_adapter.hpp"
#include "../../llm/py_jinja_template_processor.hpp"
#include "../../utils/env_guard.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/servable_initializer.hpp"
#include "../platform_utils.hpp"
#include "../test_with_temp_dir.hpp"

#include "src/filesystem/filesystem.hpp"

using namespace ovms;

// End-to-end test using Python Jinja for template rendering.
// This tests the same templates as ChatTemplateEndToEndTest (minja path)
// but uses the real Python Jinja2 engine with full extension support.
class ChatTemplateEndToEndJinjaTest : public TestWithTempDir {
protected:
    const std::string& chatTemplatesPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/chat_templates", false);
    const std::string& tokenizerModelPath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m", false);

    std::shared_ptr<GenAiServable> servable;
    std::string savedLogLevel;

    void SetUp() override {
        TestWithTempDir::SetUp();
        const char* prev = std::getenv("OPENVINO_LOG_LEVEL");
        savedLogLevel = prev ? prev : "";
        SetEnvironmentVar("OPENVINO_LOG_LEVEL", "0");
        // Copy tokenizer model files to temp dir (required by PyJinjaTemplateProcessor)
        for (const auto& filename : {"openvino_tokenizer.xml", "openvino_tokenizer.bin",
                 "openvino_detokenizer.xml", "openvino_detokenizer.bin"}) {
            std::filesystem::copy_file(
                tokenizerModelPath + "/" + filename,
                directoryPath + "/" + filename,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    void TearDown() override {
        servable.reset();
        if (savedLogLevel.empty()) {
            UnSetEnvironmentVar("OPENVINO_LOG_LEVEL");
        } else {
            SetEnvironmentVar("OPENVINO_LOG_LEVEL", savedLogLevel);
        }
        TestWithTempDir::TearDown();
    }

    // --- Inputs (set by each test) ---
    std::string chatTemplate;
    ov::genai::ChatHistory chatHistory;

    // --- Derived state (populated by run()) ---
    ChatTemplateAnalysisResult analysisResult;
    ChatTemplateCaps caps;
    std::string appliedOutput;
    bool exceptionThrownDuringApplication = false;

    // Load template from file
    static std::string loadTemplateFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    }

    // Initialize the Python Jinja template processor with the current chatTemplate
    void initJinjaProcessor() {
        // Write chat_template.jinja to temp dir
        std::ofstream jinjaFile(directoryPath + "/chat_template.jinja");
        jinjaFile << chatTemplate;
        jinjaFile.close();

        servable = std::make_shared<ContinuousBatchingServable>();
        servable->getProperties()->modelsPath = directoryPath;
        servable->getProperties()->tokenizer = ov::genai::Tokenizer(directoryPath);

        ExtraGenerationInfo extraGenInfo = GenAiServableInitializer::readExtraGenerationInfo(
            servable->getProperties(), directoryPath);
        GenAiServableInitializer::loadPyTemplateProcessor(servable->getProperties(), extraGenInfo);
    }

    // Run the full Jinja pipeline: analyze → probe → workarounds → apply via Python Jinja
    void run() {
        ASSERT_FALSE(chatTemplate.empty()) << "chatTemplate must be set before calling run()";
        ASSERT_FALSE(chatHistory.empty()) << "chatHistory must be set before calling run()";

        // Step 1: Static analysis
        analysisResult = ChatTemplateAnalyzer::analyze(chatTemplate);
        caps = analysisResult.caps;

        std::cout << "=== Analysis (Jinja) ===" << std::endl;
        std::cout << "  toolParser: " << analysisResult.detectedToolParser.value_or("(none)") << std::endl;
        std::cout << "  reasoningParser: " << analysisResult.detectedReasoningParser.value_or("(none)") << std::endl;
        std::cout << "  supportsToolCalls: " << caps.supportsToolCalls << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 2: Initialize Jinja processor (needed for probe and rendering)
        initJinjaProcessor();
        ASSERT_NE(servable->getProperties()->templateProcessor.chatTemplate, nullptr)
            << "Failed to load Python Jinja template processor";

        // Step 3a: Probe tool caps using Python Jinja (same function used in production)
        if (!probeChatTemplateCapsJinja(servable->getProperties()->templateProcessor, caps)) {
            std::cout << "=== Jinja Probe FAILED: silent failure detected ===" << std::endl;
        }

        // Step 3b: Probe reasoning caps using Python Jinja (same function used in production)
        {
            ov::genai::Tokenizer probeTokenizer(tokenizerModelPath);
            probeTokenizer.set_chat_template(chatTemplate);
            if (!probeChatTemplateReasoning(probeTokenizer, caps)) {
                std::cout << "=== Jinja Reasoning Probe FAILED: silent failure detected ===" << std::endl;
            }
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;
        std::cout << "  missnamedReasoningField: " << (caps.missnamedReasoningField.empty() ? "(none)" : caps.missnamedReasoningField) << std::endl;

        // Step 4: Apply workarounds to chat history
        chat_template_adapter::applyToHistory(caps, chatHistory);

        // Step 5: Serialize and render via Python Jinja (same as production ChatTemplateProcessor)
        std::string requestBody = "{\"messages\":" + chatHistory.get_messages().to_json_string() + "}";
        std::string renderOutput;
        bool success = PyJinjaTemplateProcessor::applyChatTemplate(
            servable->getProperties()->templateProcessor,
            requestBody, renderOutput);
        exceptionThrownDuringApplication = !success;
        appliedOutput = renderOutput;

        std::cout << "=== Result (Jinja) ===" << std::endl;
        if (!exceptionThrownDuringApplication) {
            std::cout << appliedOutput << std::endl;
        } else {
            std::cout << "FAILED: " << appliedOutput << std::endl;
        }
    }
};

// =============================================================================
// The chat template we use here contains multiple patches, including one that relates to `string2obj`.
// Without the workaround, it translates to {"key":"val"}.
// With, it would translate to {'key': 'val'} which is not correct.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, GptOss_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gpt_oss.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
// Ovms automatically detects str2obj is needed and applies workaround.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen36_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen36.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Gemma4_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gemma.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "gemma4");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "gemma4");

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s><|turn>user
What's the weather in Paris?<turn|>
<|turn>model
<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// This test is running against chat template which contains our patch with `from_json` filter.
// This filter is supported by Jinja, therefore the test is expected to pass.
// Ovms must not apply str2obj, because chat template does it already.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3Coder_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3coder_instruct.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
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
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Chat template taken from Ovms extras, original chat template does not render tools at all.
// Model does not need str2obj workaround.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Phi4Mini_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_phi4_mini.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
TEST_F(ChatTemplateEndToEndJinjaTest, Mistral7B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_mistral7b_v03.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"abc123def","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
// LFM2 does not render tool calls in current chat template, therefore the output is the same as if there were no tool calls.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, LFM2_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm2.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_TRUE(caps.missnamedReasoningField.empty());

    std::string expectedOutput = R"(</s><|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<|im_end|>
<|im_start|>assistant
)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

TEST_F(ChatTemplateEndToEndJinjaTest, LFM25_ToolCallWithStringArgsAndReasoning) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty()) << "Failed to load lfm2.5 template";

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant", "reasoning_content":"Here is some reasoning content","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "lfm2.5");

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
// str2obj is required for this template, Ovms detects it and applies the workaround.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, LFM25_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

    ASSERT_FALSE(exceptionThrownDuringApplication);

    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "lfm2");
    ASSERT_TRUE(analysisResult.detectedReasoningParser.has_value());
    EXPECT_EQ(analysisResult.detectedReasoningParser.value(), "lfm2.5");

    EXPECT_EQ(caps.missnamedReasoningField, "thinking");
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

// =============================================================================
// Same story as Qwen3-8B, but with image tags.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3VL_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3vl.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3_30B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3_30b.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's the weather in Paris?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]})"));

    run();

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
