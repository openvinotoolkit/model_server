//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <rapidjson/error/en.h>

#include "src/filesystem/filesystem.hpp"
#include "../../http_payload.hpp"
#include "../../http_rest_api_handler.hpp"
#include "../../httpservermodule.hpp"
#include "../../llm/io_processing/output_parser.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/language_model/continuous_batching/servable_initializer.hpp"
#include "../../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../../server.hpp"
#include "../../servable_management/modelmanager.hpp"
#include "../../servable_management/servable_group_manager.hpp"
#include "../../servable_management/servablemanagermodule.hpp"
#include "../platform_utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

#include "src/test/test_http_utils.hpp"
#include "src/test/test_utils.hpp"
#include "src/test/light_test_utils.hpp"
#include "src/test/test_with_temp_dir.hpp"

using namespace ovms;

class LLMChatTemplateTest : public TestWithTempDir {
protected:
    std::string tokenizerConfigFilePath;
    std::string jinjaConfigFilePath;
    std::shared_ptr<GenAiServable> servable;

    void LoadTemplateProcessor() {
        servable = std::make_shared<ContinuousBatchingServable>();
        servable->getProperties()->modelsPath = directoryPath;
        servable->getProperties()->tokenizer = ov::genai::Tokenizer(directoryPath);
        std::cout << "Chat template to be used: \n"
                  << servable->getProperties()->tokenizer.get_original_chat_template() << std::endl;
        auto& properties = *servable->getProperties();
        properties.chatTemplateEngine = ChatTemplateEngine::JINJA;
        const auto status = GenAiServableInitializer::loadChatTemplate(servable->getProperties(), directoryPath);
        ASSERT_TRUE(status.ok()) << status.string();
        ASSERT_EQ(properties.chatTemplateStatus, ChatTemplateStatus::AVAILABLE);
        GenAiServableInitializer::analyzeChatTemplate(servable->getProperties());
        ASSERT_TRUE(properties.preparedChatTemplate.isPrepared());
    }

    bool render(const std::string& body, std::string& output) {
        return servable->getProperties()->preparedChatTemplate.apply(body, output);
    }

    void SetUp() {
        TestWithTempDir::SetUp();
        tokenizerConfigFilePath = directoryPath + "/tokenizer_config.json";
        jinjaConfigFilePath = directoryPath + "/chat_template.jinja";

        // We need real model tokenizer and detokenizer as we rely on them to load chat template properly
        std::string realModelPath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m");

        std::string srcTokenizerPath = ovms::FileSystem::joinPath({realModelPath, "openvino_tokenizer.xml"});
        std::string dstTokenizerPath = ovms::FileSystem::joinPath({directoryPath, "openvino_tokenizer.xml"});
        std::filesystem::copy_file(srcTokenizerPath, dstTokenizerPath, std::filesystem::copy_options::overwrite_existing);

        std::string srcTokenizerBinPath = ovms::FileSystem::joinPath({realModelPath, "openvino_tokenizer.bin"});
        std::string dstTokenizerBinPath = ovms::FileSystem::joinPath({directoryPath, "openvino_tokenizer.bin"});
        std::filesystem::copy_file(srcTokenizerBinPath, dstTokenizerBinPath, std::filesystem::copy_options::overwrite_existing);

        std::string srcDetokenizerPath = ovms::FileSystem::joinPath({realModelPath, "openvino_detokenizer.xml"});
        std::string dstDetokenizerPath = ovms::FileSystem::joinPath({directoryPath, "openvino_detokenizer.xml"});
        std::filesystem::copy_file(srcDetokenizerPath, dstDetokenizerPath, std::filesystem::copy_options::overwrite_existing);

        std::string srcDetokenizerBinPath = ovms::FileSystem::joinPath({realModelPath, "openvino_detokenizer.bin"});
        std::string dstDetokenizerBinPath = ovms::FileSystem::joinPath({directoryPath, "openvino_detokenizer.bin"});
        std::filesystem::copy_file(srcDetokenizerBinPath, dstDetokenizerBinPath, std::filesystem::copy_options::overwrite_existing);
    }

    void TearDown() {
        servable.reset();
        std::filesystem::remove_all(directoryPath);
        TestWithTempDir::TearDown();
    }

public:
    bool CreateTokenizerConfig(std::string& fileContents) {
        return createConfigFileWithContent(fileContents, tokenizerConfigFilePath);
    }
    bool CreateJinjaConfig(std::string& fileContents) {
        return createConfigFileWithContent(fileContents, jinjaConfigFilePath);
    }
    void CopyDefaultChatTemplate() {
        std::string srcFilePath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m/chat_template.jinja");
        std::string dstFilePath = ovms::FileSystem::joinPath({directoryPath, "chat_template.jinja"});
        std::filesystem::copy_file(srcFilePath, dstFilePath, std::filesystem::copy_options::overwrite_existing);
    }
};

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyBody) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = "";
    ASSERT_EQ(render(payloadBody, finalPrompt), false);
    std::string errorOutput = "Expecting value: line 1 column 1 (char 0)";
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ExplicitJinjaRejectsInvalidTemplate) {
    std::string invalidJinja = "{% invalid_tag %}";
    ASSERT_TRUE(CreateJinjaConfig(invalidJinja));
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    mediapipe::LLMCalculatorOptions nodeOptions;
    nodeOptions.set_chat_template_engine(mediapipe::LLMCalculatorOptions::JINJA);
    properties->chatTemplateEngine = GenAiServableInitializer::determineChatTemplateEngine(nodeOptions);

    auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateEngine, ChatTemplateEngine::JINJA);
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::LOAD_FAILED);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::NONE);
    EXPECT_FALSE(properties->preparedChatTemplate.isPrepared());
    const auto validationStatus = GenAiServableInitializer::validateChatTemplateAvailability(properties);
    EXPECT_FALSE(validationStatus.ok());
    EXPECT_NE(validationStatus.string().find("could not be loaded"), std::string::npos);
}

TEST_F(LLMChatTemplateTest, EmptyJinjaFileFallsBackToGenAiTokenizerTemplate) {
    std::string emptyJinja;
    ASSERT_TRUE(CreateJinjaConfig(emptyJinja));
    std::filesystem::resize_file(jinjaConfigFilePath, 0);
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->tokenizer.set_chat_template("{{ messages[0]['content'] }}");
    properties->chatTemplateEngine = ChatTemplateEngine::JINJA;

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::AVAILABLE);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::GENAI_TOKENIZER);
    EXPECT_TRUE(properties->preparedChatTemplate.isPrepared());
}

TEST_F(LLMChatTemplateTest, JinjaReadFailureFallsBackToGenAiTokenizerTemplate) {
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->tokenizer.set_chat_template("{{ messages[0]['content'] }}");
    properties->chatTemplateEngine = ChatTemplateEngine::JINJA;
    ASSERT_TRUE(std::filesystem::create_directory(jinjaConfigFilePath));

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::AVAILABLE);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::GENAI_TOKENIZER);
    EXPECT_TRUE(properties->preparedChatTemplate.isPrepared());
}

TEST_F(LLMChatTemplateTest, JinjaFileWinsOverTokenizerConfigTemplate) {
    std::string fileTemplate = "file:{{ messages[0]['content'] }}";
    ASSERT_TRUE(CreateJinjaConfig(fileTemplate));
    std::string tokenizerConfig = R"({"chat_template":"config:{{ messages[0]['content'] }}"})";
    ASSERT_TRUE(CreateTokenizerConfig(tokenizerConfig));
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->chatTemplateEngine = ChatTemplateEngine::JINJA;

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::AVAILABLE);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::CHAT_TEMPLATE_JINJA);

    std::string output;
    ASSERT_TRUE(properties->preparedChatTemplate.apply(R"({"messages":[{"content":"selected"}]})", output)) << output;
    EXPECT_EQ(output, "file:selected");
}

TEST_F(LLMChatTemplateTest, MinjaUsesChatTemplateJinjaSource) {
    std::string fileTemplate = "{% for message in messages %}{{ message['content'] }}{% endfor %}";
    ASSERT_TRUE(CreateJinjaConfig(fileTemplate));
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->chatTemplateEngine = ChatTemplateEngine::MINJA;

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    GenAiServableInitializer::analyzeChatTemplate(properties);
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::AVAILABLE);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::CHAT_TEMPLATE_JINJA);
    // CreateJinjaConfig adds newline at the end. To consider if this behavior is expected in all cases.
    EXPECT_EQ(properties->tokenizer.get_chat_template(), fileTemplate + "\n");
}

TEST_F(LLMChatTemplateTest, ImplicitJinjaLoadFailureKeepsServableAvailable) {
    std::string invalidJinja = "{% invalid_tag %}";
    ASSERT_TRUE(CreateJinjaConfig(invalidJinja));
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->chatTemplateEngine = GenAiServableInitializer::determineChatTemplateEngine(
        mediapipe::LLMCalculatorOptions{});
    ASSERT_EQ(properties->chatTemplateEngine, ChatTemplateEngine::JINJA);

    auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateEngine, ChatTemplateEngine::JINJA);
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::LOAD_FAILED);
    EXPECT_FALSE(properties->preparedChatTemplate.isPrepared());
}

TEST_F(LLMChatTemplateTest, MissingTemplateDisablesChatUnlessEngineWasExplicit) {
    for (const bool pythonRuntimeAvailable : {true, false}) {
        const mediapipe::LLMCalculatorOptions nodeOptions;
        servable = std::make_shared<ContinuousBatchingServable>();
        auto properties = servable->getProperties();
        properties->tokenizer = ov::genai::Tokenizer(directoryPath);
        properties->tokenizer.set_chat_template("");
        properties->chatTemplateEngine = GenAiServableInitializer::determineChatTemplateModeWithRuntimeContext(
            nodeOptions, pythonRuntimeAvailable);

        const auto loadStatus = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
        ASSERT_TRUE(loadStatus.ok()) << loadStatus.string();
        EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::MISSING);
        EXPECT_FALSE(properties->preparedChatTemplate.isPrepared());
    }

    for (const auto engine : {mediapipe::LLMCalculatorOptions::JINJA, mediapipe::LLMCalculatorOptions::MINJA}) {
        servable = std::make_shared<ContinuousBatchingServable>();
        auto properties = servable->getProperties();
        properties->tokenizer = ov::genai::Tokenizer(directoryPath);
        properties->tokenizer.set_chat_template("");
        properties->chatTemplateEngine = engine == mediapipe::LLMCalculatorOptions::JINJA ? ChatTemplateEngine::JINJA : ChatTemplateEngine::MINJA;

        const auto loadStatus = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
        ASSERT_TRUE(loadStatus.ok()) << loadStatus.string();
        EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::MISSING);
        const auto validationStatus = GenAiServableInitializer::validateChatTemplateAvailability(properties);
        EXPECT_FALSE(validationStatus.ok());
        EXPECT_NE(validationStatus.string().find("Explicit chat_template_engine"), std::string::npos);
    }
}

TEST_F(LLMChatTemplateTest, MissingTemplateRejectsChatInputButAllowsCompletionInput) {
    servable = std::make_shared<ContinuousBatchingServable>();
    servable->getProperties()->chatTemplateStatus = ChatTemplateStatus::MISSING;

    auto chatContext = servable->createExecutionContext();
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello"}});
    chatContext->inputRequest.input = std::move(history);
    const auto chatStatus = servable->validateInputCompatibility(chatContext);
    EXPECT_EQ(chatStatus.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(chatStatus.message(), "This servable supports only /v1/completions endpoint.");

    auto completionContext = servable->createExecutionContext();
    completionContext->inputRequest.input = std::string("Hello");
    EXPECT_TRUE(servable->validateInputCompatibility(completionContext).ok());
}

TEST_F(LLMChatTemplateTest, MissingTemplateClearsDisabledParsers) {
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->chatTemplateEngine = ChatTemplateEngine::MINJA;
    properties->toolParserName = "none";
    properties->reasoningParserName = "none";

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::MISSING);
    EXPECT_EQ(properties->toolParserName, "none");
    EXPECT_EQ(properties->reasoningParserName, "none");

    GenAiServableInitializer::clearDisabledParserNames(properties);
    EXPECT_TRUE(properties->toolParserName.empty());
    EXPECT_TRUE(properties->reasoningParserName.empty());

    const ToolsSchemas_t toolSchemas;
    OutputParser outputParser(properties->tokenizer, properties->toolParserName, properties->reasoningParserName, toolSchemas);
    EXPECT_FALSE(outputParser.isToolParserAvailable());
    EXPECT_FALSE(outputParser.isReasoningParserAvailable());
}

TEST_F(LLMChatTemplateTest, MinjaIncompatibilityIsRecordedAsUnavailable) {
    std::string invalidTemplate = "{% invalid_tag %}";
    ASSERT_TRUE(CreateJinjaConfig(invalidTemplate));
    servable = std::make_shared<ContinuousBatchingServable>();
    auto properties = servable->getProperties();
    properties->tokenizer = ov::genai::Tokenizer(directoryPath);
    properties->chatTemplateEngine = ChatTemplateEngine::MINJA;

    const auto status = GenAiServableInitializer::loadChatTemplate(properties, directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(properties->chatTemplateStatus, ChatTemplateStatus::INCOMPATIBLE);
    EXPECT_EQ(properties->chatTemplateSource, ChatTemplateSource::NONE);
    EXPECT_FALSE(GenAiServableInitializer::validateChatTemplateAvailability(properties).ok());
}

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyMessage) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_TRUE(finalPrompt.empty());
}

TEST_F(LLMChatTemplateTest, ChatTemplateMessageWithEmptyObject) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{}]
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, "");
}

TEST_F(LLMChatTemplateTest, ChatTemplateDefault) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "role": "user", "content": "How can I help you?" }]
        }
    )";
    std::string expectedOutput = "User: How can I help you?";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

// add_generation_prompt request field controls whether the trailing generation
// prompt is rendered (assistant prefill support, issue #3877).
TEST_F(LLMChatTemplateTest, ChatTemplateAddGenerationPromptDefaultsTrue) {
    std::string jinja = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<|GEN|>{% endif %}";
    ASSERT_TRUE(CreateJinjaConfig(jinja));
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "role": "user", "content": "hi" }]
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_NE(finalPrompt.find("<|GEN|>"), std::string::npos) << "default should add generation prompt, got: " << finalPrompt;
}

TEST_F(LLMChatTemplateTest, ChatTemplateAddGenerationPromptFalse) {
    std::string jinja = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<|GEN|>{% endif %}";
    ASSERT_TRUE(CreateJinjaConfig(jinja));
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    // add_generation_prompt is folded into chat_template_kwargs by the OpenAI API handler
    // before applyChatTemplate is ever called; reflect that contract here.
    std::string payloadBody = R"(
        {
            "messages": [{ "role": "user", "content": "hi" }, { "role": "assistant", "content": "partial" }],
            "chat_template_kwargs": { "add_generation_prompt": false }
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt.find("<|GEN|>"), std::string::npos) << "add_generation_prompt=false should omit generation prompt, got: " << finalPrompt;
    ASSERT_NE(finalPrompt.find("partial"), std::string::npos) << "assistant prefill content should be present, got: " << finalPrompt;
}

TEST_F(LLMChatTemplateTest, ChatTemplateMultiMessage) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "role": "user", "content": "How can I help you?" }, { "role": "user", "content": "2How can I help you?" }]
        }
    )";
    std::string expectedOutput = "User: How can I help you?User: 2How can I help you?";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateComplexMessage) {
    CopyDefaultChatTemplate();
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "User: hello";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaUppercase) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[0]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    LoadTemplateProcessor();

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = " Hi, HELLO ";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaException) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[3]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string errorOutput = "list object has no element 3";
    ASSERT_EQ(render(payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateComparePythonAndGenAiProcessors) {
    GTEST_SKIP() << "Skipping test due to GenAI template processor not being able to compare values of different types (no implicit conversion). Enable when resolved.";
    // Using modified Llama2 template to work with limited tokenizer object (with no models loaded)
    std::string tokenizerJson = R"({
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '<s>' + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' </s>' }}{% endif %}{% endfor %}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);

    std::string pythonProcessorOutput = "";
    std::string payloadBody = R"(
        {
            "model": "model",
            "stream": false,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is OpenVINO?"
                },
                {
                    "role": "assistant",
                    "content": "OpenVINO is a toolkit for optimizing and deploying deep learning models."
                },
                {
                    "role": "user",
                    "content": "Is it free to use?"
                }
            ]
        }
    )";
    ASSERT_EQ(render(payloadBody, pythonProcessorOutput), true);
    ov::genai::Tokenizer tokenizer(directoryPath);
    ov::genai::ChatHistory chatHistory;
    chatHistory.push_back({{"role", "system"}, {"content", "You are a helpful assistant."}});
    chatHistory.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});
    chatHistory.push_back({{"role", "assistant"}, {"content", "OpenVINO is a toolkit for optimizing and deploying deep learning models."}});
    chatHistory.push_back({{"role", "user"}, {"content", "Is it free to use?"}});
    // Issue with 'system_message != false' part of the template
    std::string chatTemplate = R"({% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '<s>' + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' </s>' }}{% endif %}{% endfor %})";
    tokenizer.set_chat_template(chatTemplate);
    std::cout << "GenAI chat template: " << tokenizer.get_chat_template() << std::endl;
    constexpr bool addGenerationPrompt = true;
    std::string genAiProcessorOutput = tokenizer.apply_chat_template(chatHistory, addGenerationPrompt);
    ASSERT_EQ(pythonProcessorOutput, genAiProcessorOutput);
}

std::string CreatePayloadBodyWithChatTemplateKwargs(const std::string& chat_template_kwargs) {
    std::string base = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
    )";
    if (!chat_template_kwargs.empty()) {
        base += R"(,)";
        base += R"("chat_template_kwargs": )";
        base += chat_template_kwargs;
    }
    base += R"(})";
    return base;
}

TEST_F(LLMChatTemplateTest, ChatTemplateKwargsPositive) {
    std::string templateJinja = R"({% if enable_thinking %}Thinking is on{% else %}Thinking is off{% endif %})";

    ASSERT_EQ(CreateJinjaConfig(templateJinja), true);
    LoadTemplateProcessor();
    std::string finalPrompt = "";
    std::string payloadBody;
    std::string expectedOutput;

    // Explicitly setting enable_thinking to true
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({"enable_thinking": true})");
    expectedOutput = "Thinking is on";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Explicitly setting enable_thinking to false
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({"enable_thinking": false})");
    expectedOutput = "Thinking is off";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Setting chat_template_kwargs to empty object
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({})");
    expectedOutput = "Thinking is off";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Explicitly setting chat_template_kwargs to null
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs("null");
    expectedOutput = "Thinking is off";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Setting chat_template_kwargs with multiple values including enable_thinking
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({"enable_thinking": true, "another_param": "value"})");
    expectedOutput = "Thinking is on";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Setting chat_template_kwargs with multiple values but without enable_thinking
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({"another_param": "value", "yet_another_param": [1,2,3]})");
    expectedOutput = "Thinking is off";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // Default setting
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs("");
    expectedOutput = "Thinking is off";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateKwargsNegative) {
    std::string templateJinja = R"({% if enable_thinking %}Thinking is on{% else %}Thinking is off{% endif %})";
    ASSERT_EQ(CreateJinjaConfig(templateJinja), true);
    LoadTemplateProcessor();
    // chat_template_kwargs must be an object
    // This is a negative test case to ensure that the template processor correctly handles invalid chat_template
    std::string finalPrompt = "";
    std::string payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"("string, not_an_object")");
    std::string expectedOutput = "chat_template_kwargs must be an object";
    ASSERT_EQ(render(payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);

    // chat_template_kwargs cannot contain keys that are natively provided to the template
    payloadBody = CreatePayloadBodyWithChatTemplateKwargs(R"({"messages": [{"role": "user", "content": "hello"}]})");
    expectedOutput = "jinja2.environment.Template.render() got multiple values for keyword argument 'messages'";
    ASSERT_EQ(render(payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTojsonNoHtmlEscaping) {
    // Simulates Granite/DeepSeek-R1 style tool_call templates that use tojson
    // The tojson filter must return plain JSON, not HTML-escaped Markup.
    // Without the override, Jinja2's default tojson wraps output in Markup,
    // causing < > & " to be escaped as &lt; &gt; &amp; &quot; when
    // concatenated with plain strings via + in {% set %} blocks.
    //
    // Key: the bug only manifests when tojson output (Markup) is concatenated
    // with a plain str inside {% set %}, because Markup.__radd__ escapes the
    // str operand. Direct {{ tojson }} rendering does NOT trigger the bug.
    std::string jinjaTemplate =
        "{%- set ns = namespace(tool_text='<tools>') %}"
        "{%- if tools %}"
        "  {%- for tool in tools %}"
        "    {%- set ns.tool_text = ns.tool_text + '\\n' + (tool | tojson) %}"
        "  {%- endfor %}"
        "  {%- set ns.tool_text = ns.tool_text + '\\n</tools>' %}"
        "{%- endif %}"
        "{{ ns.tool_text }}";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    LoadTemplateProcessor();

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    // Must contain literal <tools> tags, NOT &lt;tools&gt;
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("<tools>"));
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("</tools>"));
    // Must not contain any HTML-escaped entities
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&lt;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&gt;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&quot;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&amp;")));
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("get_weather"));
}

TEST_F(LLMChatTemplateTest, ChatTemplateTojsonIndentWorks) {
    // Verifies that tojson(indent=2) still produces indented JSON output
    // after the tojson override that prevents HTML escaping.
    std::string jinjaTemplate =
        "{%- set ns = namespace(tool_text='<tools>') %}"
        "{%- if tools %}"
        "  {%- for tool in tools %}"
        "    {%- set ns.tool_text = ns.tool_text + '\\n' + (tool | tojson(indent=2)) %}"
        "  {%- endfor %}"
        "  {%- set ns.tool_text = ns.tool_text + '\\n</tools>' %}"
        "{%- endif %}"
        "{{ ns.tool_text }}";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    LoadTemplateProcessor();

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        }
    )";
    ASSERT_EQ(render(payloadBody, finalPrompt), true);
    // Must contain literal <tools> tags, NOT &lt;tools&gt;
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("<tools>"));
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("</tools>"));
    // Must not contain any HTML-escaped entities
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&lt;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&gt;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&quot;")));
    EXPECT_THAT(finalPrompt, ::testing::Not(::testing::HasSubstr("&amp;")));
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("get_weather"));
    // Verify indentation is present — indent=2 produces newlines and spaces inside JSON
    EXPECT_THAT(finalPrompt, ::testing::HasSubstr("  \"type\": \"function\""));
}

std::string configTemplate = R"(
        {
            "model_config_list": [],
            "mediapipe_config_list": [
            {
                "name":"lm_cb_regular",
                "graph_path":"<GRAPH_PATTERN>"
            }
            ]
        }
    )";

std::string graphTemplate = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node {
            name: "llmNode1"
            calculator: "HttpLLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
            }
            node_options: {
                [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
                models_path: "<MODELS_PATTERN>",
                }
            }
            input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
                [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                    tag_index: "LOOPBACK:0"
                }
                }
            }
            }
    })";

class CleanupFilesGuard {
    const std::string& pathToClean;

public:
    CleanupFilesGuard(const std::string& pathToClean) :
        pathToClean(pathToClean) {}
    ~CleanupFilesGuard() {
        std::filesystem::remove_all(pathToClean);
    }
};

const std::string GRAPH_PATTERN = "<GRAPH_PATTERN>";
const std::string WORKSPACE_PATTERN = "<MODELS_PATTERN>";
const std::string MODEL_PATH = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m");

class LLMChatTemplateHttpTest : public TestWithTempDir {
protected:
    static std::unique_ptr<std::thread> t;

    std::string tokenizerConfigFilePath;
    std::string jinjaConfigFilePath;
    std::string ovmsConfigFilePath;
    std::string graphConfigFilePath;

    static std::string GetFileNameFromPath(const std::string& parentDir, const std::string& fullPath) {
        std::string fileName = fullPath;
        fileName.replace(fileName.find(parentDir), std::string(parentDir).size(), "");
        return fileName;
    }

    static bool CreateConfigFile(const std::string& graphPath, const std::string& configFilePath) {
        std::string configContents = configTemplate;
        configContents.replace(configContents.find(GRAPH_PATTERN), std::string(GRAPH_PATTERN).size(), graphPath);
        return createConfigFileWithContent(configContents, configFilePath);
    }

    static bool CreatePipelineGraph(const std::string& workspacePath, const std::string& graphConfigFilePath) {
        std::string configContents = graphTemplate;
        configContents.replace(configContents.find(WORKSPACE_PATTERN), std::string(WORKSPACE_PATTERN).size(), workspacePath);
        return createConfigFileWithContent(configContents, graphConfigFilePath);
    }

    static void CreateSymbolicLinks(const std::string& toDirectory) {
        for (const auto& entry : fs::directory_iterator(MODEL_PATH)) {
            std::filesystem::path outFilename = entry.path();
            std::string outFilenameStr = outFilename.string();
            std::string fileName = GetFileNameFromPath(MODEL_PATH, outFilenameStr);
            SPDLOG_INFO("Filename to link {}\n", fileName);
            std::string symlinkPath = ovms::FileSystem::joinPath({toDirectory, fileName});
            SPDLOG_INFO("Creating symlink from: {}\n to:\n{}", outFilenameStr, symlinkPath);
            fs::create_symlink(outFilenameStr, symlinkPath);
            // TODO: Symlinks are never removed
        }
    }

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v1/chat/completions";
    const std::string endpointCompletions = "/v1/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        TestWithTempDir::SetUp();
        tokenizerConfigFilePath = directoryPath + "/tokenizer_config.json";
        jinjaConfigFilePath = directoryPath + "/chat_template.jinja";
        ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        graphConfigFilePath = directoryPath + "/graph_config.pbtxt";

        CreateConfigFile(graphConfigFilePath, ovmsConfigFilePath);
        CreatePipelineGraph(directoryPath, graphConfigFilePath);
        CreateSymbolicLinks(directoryPath);

        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, ovmsConfigFilePath.c_str());
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        TestWithTempDir::TearDown();
    }
};
std::unique_ptr<std::thread> LLMChatTemplateHttpTest::t;

std::string fullResponse;

class LLMIdleGroupWakeUpHttpTest : public LLMChatTemplateHttpTest {
protected:
    static constexpr const char* kGraphName = "lm_cb_regular";
    static constexpr const char* kGroupName = "lazy";

    void SetUp() override {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        TestWithTempDir::SetUp();
        ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        const std::string graphPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/lm_cb_regular.pbtxt");

        const std::string config = R"({
            "model_config_list": [],
            "mediapipe_config_list": [{
                "name": "lm_cb_regular",
                "graph_path": ")" + graphPath +
                                   R"(",
                "group_name": "lazy"
            }]
        })";
        ASSERT_TRUE(createConfigFileWithContent(config, ovmsConfigFilePath));

        std::string port = "9174";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServerWithExtraArgs(t, server, port, ovmsConfigFilePath.c_str(), {"--idle_unload_timeout_seconds", "30"});
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    ovms::ModelManager& manager() const {
        ovms::Server& server = ovms::Server::instance();
        return dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME))->getServableManager();
    }
};

TEST_F(LLMIdleGroupWakeUpHttpTest, V3UnaryRequestWakesSleepingGroupedGraph) {
    auto* definition = manager().getMediapipeFactory().findDefinitionByName(kGraphName);
    ASSERT_NE(definition, nullptr);
    EXPECT_EQ(definition->getStateCode(), ovms::PipelineDefinitionStateCode::SLEEPING);

    auto* groupManager = manager().getGroupManager();
    ASSERT_NE(groupManager, nullptr);
    EXPECT_TRUE(groupManager->getActiveGroupName().empty());

    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Hi"}]
        }
    )";

    const auto status = handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(groupManager->getActiveGroupName(), kGroupName);
}

class LLMJinjaChatTemplateHttpTest : public LLMChatTemplateHttpTest {
public:
    static std::unique_ptr<std::thread> t;

protected:
    static const std::string getDirectoryPath() {
        const std::string directoryName = "LLMJinjaChatTemplateHttpTest";
        std::string directoryPath = std::string{"/tmp/"} + directoryName;
        return getGenericFullPathForTmp(directoryPath);
    }
    static void SetUpTestSuite() {
        const auto directoryPath = getDirectoryPath();
        std::filesystem::remove_all(directoryPath);
        std::filesystem::create_directories(directoryPath);

        const std::string ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        CreateConfigFile(
            directoryPath + "/graph_config.pbtxt",
            ovmsConfigFilePath);
        CreatePipelineGraph(
            directoryPath,
            directoryPath + "/graph_config.pbtxt");
        CreateSymbolicLinks(directoryPath);
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, ovmsConfigFilePath.c_str());
    }

    void SetUp() override {
        writer = std::make_shared<MockedServerRequestInterface>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    void TearDown() override {
        handler.reset();
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        std::filesystem::remove_all(getDirectoryPath());
    }
};

std::unique_ptr<std::thread> LLMJinjaChatTemplateHttpTest::t;

TEST_F(LLMJinjaChatTemplateHttpTest, inferChatCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferChatCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": true,
            "seed" : 1,
            "max_tokens": 6,
            "prompt": "?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": true,
            "seed" : 1,
            "max_tokens": 6,
            "prompt": "?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferDefaultChatCompletionsUnary) {
    std::unique_ptr<CleanupFilesGuard> cleanupGuard = std::make_unique<CleanupFilesGuard>(directoryPath);
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}
