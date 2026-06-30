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

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../../llm/chat_template_analyzer.hpp"
#include "../../llm/chat_template_caps.hpp"
#include "../../llm/chat_template_probe.hpp"
#include "../../llm/input_workarounds.hpp"
#include "../../llm/py_jinja_template_processor.hpp"
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
        setenv("OPENVINO_LOG_LEVEL", "0", 1);
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
            unsetenv("OPENVINO_LOG_LEVEL");
        } else {
            setenv("OPENVINO_LOG_LEVEL", savedLogLevel.c_str(), 1);
        }
        TestWithTempDir::TearDown();
    }

    // --- Inputs (set by each test) ---
    std::string chatTemplate;

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

    // Initialize the Python Jinja template processor with the current chatTemplate
    void initJinjaProcessor() {
        // Write chat_template.jinja to temp dir
        std::ofstream jinjaFile(directoryPath + "/chat_template.jinja");
        jinjaFile << chatTemplate;
        jinjaFile.close();

        servable = std::make_shared<ContinuousBatchingServable>();
        servable->getProperties()->modelsPath = directoryPath;
        servable->getProperties()->tokenizer = ov::genai::Tokenizer(directoryPath);
        // Override the tokenizer's template with our test template
        servable->getProperties()->tokenizer.set_chat_template(chatTemplate);

        ExtraGenerationInfo extraGenInfo = GenAiServableInitializer::readExtraGenerationInfo(
            servable->getProperties(), directoryPath);
        GenAiServableInitializer::loadPyTemplateProcessor(servable->getProperties(), extraGenInfo);
    }

    // Probe tool caps using Python Jinja (same engine as production Jinja path)
    // Apply workarounds to the JSON request body
    std::string applyWorkarounds(const std::string& jsonBody) {
        if (!caps.requiresObjectArguments && !caps.requiresNonNullContent) {
            return jsonBody;
        }
        rapidjson::Document doc;
        doc.Parse(jsonBody.c_str());
        if (doc.HasParseError()) {
            return jsonBody;
        }
        input_workarounds::applyToJson(caps, analysisResult.detectedModelFamily, doc);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        return buffer.GetString();
    }

    // Run the full Jinja pipeline: analyze → probe → workarounds → apply via Python Jinja
    void run(const std::string& requestJson) {
        ASSERT_FALSE(chatTemplate.empty()) << "chatTemplate must be set before calling run()";
        ASSERT_FALSE(requestJson.empty()) << "requestJson must be provided";

        // Step 1: Static analysis
        analysisResult = ChatTemplateAnalyzer::analyze(chatTemplate);
        caps = analysisResult.caps;

        std::cout << "=== Analysis (Jinja) ===" << std::endl;
        std::cout << "  modelFamily: " << analysisResult.detectedModelFamily << std::endl;
        std::cout << "  toolParser: " << analysisResult.detectedToolParser.value_or("(none)") << std::endl;
        std::cout << "  reasoningParser: " << analysisResult.detectedReasoningParser.value_or("(none)") << std::endl;
        std::cout << "  supportsToolCalls: " << caps.supportsToolCalls << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 2: Initialize Jinja processor (needed for probe and rendering)
        initJinjaProcessor();
        ASSERT_NE(servable->getProperties()->templateProcessor.chatTemplate, nullptr)
            << "Failed to load Python Jinja template processor";

        // Step 3: Probe tool caps using Python Jinja (same function used in production)
        if (!probeChatTemplateCapsJinja(servable->getProperties()->templateProcessor,
                servable->getProperties()->modelsPath, caps)) {
            std::cout << "=== Jinja Probe FAILED: silent failure detected ===" << std::endl;
        }

        std::cout << "=== After Probe ===" << std::endl;
        std::cout << "  requiresObjectArguments: " << caps.requiresObjectArguments << std::endl;

        // Step 4: Apply workarounds to JSON
        std::string modifiedJson = applyWorkarounds(requestJson);

        // Step 5: Render via Python Jinja
        applySuccess = PyJinjaTemplateProcessor::applyChatTemplate(
            servable->getProperties()->templateProcessor,
            servable->getProperties()->modelsPath,
            modifiedJson, appliedOutput);

        std::cout << "=== Result (Jinja) ===" << std::endl;
        if (applySuccess) {
            std::cout << appliedOutput << std::endl;
        } else {
            std::cout << "FAILED: " << appliedOutput << std::endl;
        }
    }
};

// =============================================================================
// Jinja: gpt-oss-20b with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, GptOss_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gpt_oss.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

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
// Jinja: Qwen3.6-35B with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen36_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen36.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

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
// Jinja: Gemma4 with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Gemma4_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_gemma.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

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

    std::string expectedOutput = R"(</s><|turn>user
What's the weather in Paris?<turn|>
<|turn>model
<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Jinja: Qwen3 Coder — Python Jinja supports from_json, so this WORKS
// unlike the minja path which silently fails.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3Coder_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3coder_instruct.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);

    EXPECT_EQ(analysisResult.detectedModelFamily, "qwen3coder");
    ASSERT_TRUE(analysisResult.detectedToolParser.has_value());
    EXPECT_EQ(analysisResult.detectedToolParser.value(), "qwen3coder");
    ASSERT_FALSE(analysisResult.detectedReasoningParser.has_value());

    EXPECT_TRUE(caps.supportsTools);
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.supportsToolResponses);
    EXPECT_FALSE(caps.requiresObjectArguments);
    EXPECT_FALSE(caps.requiresNonNullContent);
}

// =============================================================================
// Jinja: Phi4Mini with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Phi4Mini_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_phi4_mini.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);
    EXPECT_FALSE(caps.supportsToolCalls);
}

// =============================================================================
// Jinja: Qwen3-4B with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

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
// Jinja: Mistral-7B-Instruct-v0.3 with tool call containing string arguments
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Mistral7B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_mistral7b_v03.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"abc123def","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);
    EXPECT_EQ(analysisResult.detectedModelFamily, "devstral");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    std::string expectedOutput = R"(</s>[INST] What's the weather in Paris?[/INST][TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}, "id": "abc123def"}]</s>)";
    EXPECT_EQ(appliedOutput, expectedOutput);
}

// =============================================================================
// Jinja: LFM2-24B — no tool_call rendering, analyzer doesn't detect tool support
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, LFM2_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm2.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);
    EXPECT_FALSE(caps.supportsToolCalls);
}

// =============================================================================
// Jinja: LFM2.5 — Python Jinja supports {%- generation -%}. Without from_json,
// string args fail (.items() on string), but Jinja probe detects this and sets
// requiresObjectArguments=true. Workaround converts args → template renders OK.
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, LFM25_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_lfm25.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    // Jinja probe detects requiresObjectArguments=true, workaround converts args
    ASSERT_TRUE(applySuccess);
    EXPECT_EQ(analysisResult.detectedModelFamily, "lfm2");
    EXPECT_TRUE(caps.supportsToolCalls);
    EXPECT_TRUE(caps.requiresObjectArguments);

    EXPECT_NE(appliedOutput.find("<|tool_call_start|>"), std::string::npos);
    EXPECT_NE(appliedOutput.find("get_weather("), std::string::npos);
    EXPECT_NE(appliedOutput.find("<|tool_call_end|>"), std::string::npos);
}

// =============================================================================
// Jinja: Qwen3-VL-8B-Instruct with tool call containing string arguments
// Template handles both string and object arguments natively (has is_string check).
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3VL_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3vl.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);
    EXPECT_TRUE(caps.supportsToolCalls);

    EXPECT_NE(appliedOutput.find("<tool_call>"), std::string::npos);
    EXPECT_NE(appliedOutput.find("get_weather"), std::string::npos);
    EXPECT_NE(appliedOutput.find("</tool_call>"), std::string::npos);
}

// =============================================================================
// Jinja: Qwen3-30B-A3B-Instruct-2507 with tool call containing string arguments
// Template handles both string and object arguments natively (has is_string check).
// =============================================================================
TEST_F(ChatTemplateEndToEndJinjaTest, Qwen3_30B_ToolCallWithStringArgs) {
    chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_qwen3_30b.jinja");
    ASSERT_FALSE(chatTemplate.empty());

    std::string requestJson = R"({"messages":[
        {"role":"user","content":"What's the weather in Paris?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}
    ]})";

    run(requestJson);

    ASSERT_TRUE(applySuccess);
    EXPECT_TRUE(caps.supportsToolCalls);

    EXPECT_NE(appliedOutput.find("<tool_call>"), std::string::npos);
    EXPECT_NE(appliedOutput.find("get_weather"), std::string::npos);
    EXPECT_NE(appliedOutput.find("</tool_call>"), std::string::npos);
}
