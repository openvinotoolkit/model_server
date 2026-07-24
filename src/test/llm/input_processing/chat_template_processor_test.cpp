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

// Unit tests for ChatTemplateProcessor covering three orthogonal concerns:
//
//   1. serializeForJinja                — pure C++, no Python required.
//      Locks down the OVMS-owned JSON shape passed to any Jinja engine.
//
//   2. Native tokenizer path            — useMinja=true, no Python required.
//      Uses GenAI's built-in minja (tokenizer.apply_chat_template).
//
//   3. Runtime PyJinja path             — useMinja=false with a prepared
//      runtime chat template. Loads libovmspython at test setup and drives
//      the same tokenizer/template through standard Python Jinja2.
//
// Each tokenizer-based group uses a different fixture:
//   - SmolLM2-360M-Instruct: a model WITH a chat template (positive tests).
//   - BAAI/bge-reranker-base: a model WITHOUT a chat template (negative
//                             test for the "Failed to apply chat template" path).
//
// Error path NOT covered here:
//   "Final prompt after applying chat template is empty"
//   — This safety net requires a model whose chat_template unconditionally
//     renders to an empty string, which is not present in our test fixtures.

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>

#include "../../../llm/io_processing/input_processors/chat_template_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"
#include "../../../llm/runtime_chat_template.hpp"
#include "../../../llm/runtime_chat_template_runtime_loader.hpp"
#include "../../platform_utils.hpp"

namespace ovms {
namespace {

// SmolLM2-360M-Instruct injects this default system turn when the first
// message role is not 'system' (from tokenizer_config.json chat_template).
constexpr const char* SMOL_DEFAULT_SYSTEM =
    "<|im_start|>system\n"
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    "<|im_end|>\n";

static InputRequest makeChatRequest(ov::genai::ChatHistory history) {
    InputRequest req;
    req.input = std::move(history);
    return req;
}

// ---------------------------------------------------------------------------
// SerializeForJinja — locks down the JSON shape the OVMS-owned serializer
// emits before handing off to any Jinja engine (in-process PyJinja or the
// runtime chat-template C ABI). Pure C++, no Python required.
// ---------------------------------------------------------------------------

TEST(ChatTemplateProcessorSerializeForJinjaTest, MessagesOnly_NoOptionalKeys) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});

    const std::string json = ChatTemplateProcessor::serializeForJinja(history);

    // No optional keys must be emitted when tools and extra_context are empty.
    EXPECT_EQ(json, R"({"messages":[{"content":"Hi.","role":"user"}]})");
}

TEST(ChatTemplateProcessorSerializeForJinjaTest, MessagesAndTools_ToolsKeyIncluded) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});
    history.set_tools(ov::genai::JsonContainer::from_json_string(R"([{"type":"function"}])"));

    const std::string json = ChatTemplateProcessor::serializeForJinja(history);

    EXPECT_NE(json.find(R"("messages":[{"content":"Hi.","role":"user"}])"), std::string::npos) << json;
    EXPECT_NE(json.find(R"("tools":[{"type":"function"}])"), std::string::npos) << json;
    EXPECT_EQ(json.find("chat_template_kwargs"), std::string::npos) << json;
}

TEST(ChatTemplateProcessorSerializeForJinjaTest, MessagesAndKwargs_KwargsKeyIncluded) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(R"({"enable_thinking":true})"));

    const std::string json = ChatTemplateProcessor::serializeForJinja(history);

    EXPECT_NE(json.find(R"("messages":[{"content":"Hi.","role":"user"}])"), std::string::npos) << json;
    EXPECT_EQ(json.find(R"("tools":)"), std::string::npos) << json;
    EXPECT_NE(json.find(R"("chat_template_kwargs":{"enable_thinking":true})"), std::string::npos) << json;
}

TEST(ChatTemplateProcessorSerializeForJinjaTest, MessagesToolsAndKwargs_AllKeysPresent) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});
    history.set_tools(ov::genai::JsonContainer::from_json_string(R"([{"type":"function"}])"));
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(R"({"add_generation_prompt":false})"));

    const std::string json = ChatTemplateProcessor::serializeForJinja(history);

    // Key order in the serialized envelope is fixed: messages, then tools, then kwargs.
    const auto messagesPos = json.find(R"("messages":)");
    const auto toolsPos = json.find(R"("tools":)");
    const auto kwargsPos = json.find(R"("chat_template_kwargs":)");
    ASSERT_NE(messagesPos, std::string::npos) << json;
    ASSERT_NE(toolsPos, std::string::npos) << json;
    ASSERT_NE(kwargsPos, std::string::npos) << json;
    EXPECT_LT(messagesPos, toolsPos) << json;
    EXPECT_LT(toolsPos, kwargsPos) << json;
}

TEST(ChatTemplateProcessorSerializeForJinjaTest, EmptyHistory_ProducesEmptyMessagesArray) {
    ov::genai::ChatHistory history;

    const std::string json = ChatTemplateProcessor::serializeForJinja(history);

    EXPECT_EQ(json, R"({"messages":[]})");
}

// ---------------------------------------------------------------------------
// Fixture: SmolLM2-360M-Instruct (model WITH a chat template)
// ---------------------------------------------------------------------------

static std::unique_ptr<ov::genai::Tokenizer> sharedTokenizer;

class ChatTemplateProcessorTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sharedTokenizer = std::make_unique<ov::genai::Tokenizer>(getGenericFullPathForSrcTest(
            "/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct"));
    }

    static void TearDownTestSuite() {
        sharedTokenizer.reset();
    }
};

// ---------------------------------------------------------------------------
// Positive tests
// ---------------------------------------------------------------------------

// The template injects a default system message when the first role is 'user',
// and appends the generation-prompt suffix ("<|im_start|>assistant\n").
TEST_F(ChatTemplateProcessorTest, TextMessage_DefaultSystemInjected_GenerationPromptAppended) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();

    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\nWhat is OpenVINO?<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(req.promptText, expected);
}

// An explicit 'system' message must suppress the template's default system
// injection and appear verbatim as the first turn.
TEST_F(ChatTemplateProcessorTest, ExplicitSystemMessage_SuppressesDefaultSystemInjection) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "You are a custom assistant."}});
    history.push_back({{"role", "user"}, {"content", "Hello."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();

    const std::string expected =
        "<|im_start|>system\nYou are a custom assistant.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(req.promptText, expected);
    EXPECT_EQ(req.promptText.find("SmolLM"), std::string::npos)
        << "Default system injection must be absent when an explicit system message is provided";
}

// Every turn in a multi-turn conversation must appear in promptText in order.
TEST_F(ChatTemplateProcessorTest, MultiTurnConversation_AllTurnsRendered) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "Be concise."}});
    history.push_back({{"role", "user"}, {"content", "First question."}});
    history.push_back({{"role", "assistant"}, {"content", "First answer."}});
    history.push_back({{"role", "user"}, {"content", "Second question."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();

    const std::string expected =
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nFirst question.<|im_end|>\n"
        "<|im_start|>assistant\nFirst answer.<|im_end|>\n"
        "<|im_start|>user\nSecond question.<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(req.promptText, expected);
}

// add_generation_prompt=false (read out of chat_template_kwargs) must omit the
// trailing generation-prompt suffix while otherwise rendering normally.
TEST_F(ChatTemplateProcessorTest, AddGenerationPromptFalse_OmitsGenerationPromptSuffix) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(R"({"add_generation_prompt": false})"));

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();

    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\nWhat is OpenVINO?<|im_end|>\n";
    EXPECT_EQ(req.promptText, expected);
    EXPECT_EQ(req.promptText.find("<|im_start|>assistant"), std::string::npos)
        << "add_generation_prompt=false must omit the trailing generation prompt";
}

// A non-boolean add_generation_prompt must be rejected with a clear error
// instead of throwing an unhandled JsonContainer type-mismatch exception.
TEST_F(ChatTemplateProcessorTest, AddGenerationPromptNonBoolean_ReturnsInvalidArgument) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(R"({"add_generation_prompt": "yes"})"));

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("add_generation_prompt"), std::string::npos) << status.message();
}

// The processor must populate req.promptText and leave the ChatHistory
// variant in req.input intact (it does not replace the input variant).
TEST_F(ChatTemplateProcessorTest, PromptTextPopulated_ChatHistoryVariantPreserved) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hi."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(req.promptText.empty());
    EXPECT_TRUE(std::holds_alternative<ov::genai::ChatHistory>(req.input));
}

// A message with empty string content must still produce a non-empty prompt —
// the SmolLM2 template wraps every turn with start/end tokens regardless of
// content length, so an empty content string does not trigger the empty-prompt
// safety net in ChatTemplateProcessor.
TEST_F(ChatTemplateProcessorTest, EmptyStringContent_TemplateStillProducesOutput) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", ""}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*sharedTokenizer, true, nullptr);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(req.promptText.empty());
}

// ---------------------------------------------------------------------------
// Negative test — model WITHOUT a chat template
// (BAAI/bge-reranker-base has no chat_template specified anywhere)
// ChatTemplateProcessor must catch that exception and return kInvalidArgument.
// ---------------------------------------------------------------------------

TEST(ChatTemplateProcessorNoChatTemplateTest, TokenizerWithoutChatTemplate_ReturnsError) {
    const std::string path = getGenericFullPathForSrcTest(
        "/ovms/src/test/llm_testing/BAAI/bge-reranker-base/ov");
    ov::genai::Tokenizer tokenizer(path);

    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(tokenizer, true, nullptr);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(),
        "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
    EXPECT_TRUE(req.promptText.empty());
}

// ---------------------------------------------------------------------------
// Fixture: SmolLM2-360M-Instruct via the runtime PyJinja path (useMinja=false)
//
// This exercises the same tokenizer/template through the prepared runtime
// chat-template API (dlopen'd from libovmspython), which routes through
// standard Python Jinja2 instead of GenAI's embedded minja.
//
// The unit-test binary is built with libovmspython alongside. The global
// PythonEnvironment (see gtest_main.cpp) initializes the interpreter for
// the whole process, so this fixture only prepares the tokenizer and the
// runtime chat template. If either preparation step fails, the tests fail
// loudly (they do not silently skip).
//
// Rendering asserts are token-substring based rather than exact-string, because
// PyJinja and minja can differ in incidental whitespace even when driven by the
// same chat template.
// ---------------------------------------------------------------------------

#if (PYTHON_DISABLE == 0)
static std::unique_ptr<ov::genai::Tokenizer> pyJinjaTokenizer;
static std::unique_ptr<PreparedRuntimeChatTemplate> sharedPreparedRuntimeTemplate;

class ChatTemplateProcessorPyJinjaTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ASSERT_NE(getRuntimeChatTemplateRuntimeApi(), nullptr)
            << "libovmspython is not loadable — the unit-test binary is expected to be built "
               "alongside //src/python:libovmspython.so.";

        const std::string modelsPath = getGenericFullPathForSrcTest(
            "/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct");
        pyJinjaTokenizer = std::make_unique<ov::genai::Tokenizer>(modelsPath);

        auto prepared = std::make_unique<PreparedRuntimeChatTemplate>();
        std::string runtimeOutput;
        RuntimeChatTemplateError runtimeError = RuntimeChatTemplateError::NONE;
        const auto status = prepareRuntimeChatTemplate(
            modelsPath,
            pyJinjaTokenizer->get_chat_template(),
            pyJinjaTokenizer->get_bos_token(),
            pyJinjaTokenizer->get_eos_token(),
            *prepared,
            runtimeOutput,
            &runtimeError);
        ASSERT_EQ(status, RuntimeChatTemplatePrepareStatus::PREPARED)
            << "prepareRuntimeChatTemplate failed (error=" << static_cast<int>(runtimeError)
            << "): " << runtimeOutput;
        sharedPreparedRuntimeTemplate = std::move(prepared);
    }

    static void TearDownTestSuite() {
        sharedPreparedRuntimeTemplate.reset();
        pyJinjaTokenizer.reset();
    }
};

TEST_F(ChatTemplateProcessorPyJinjaTest, TextMessage_DefaultSystemInjected_GenerationPromptAppended) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*pyJinjaTokenizer, /*useMinja=*/false, sharedPreparedRuntimeTemplate.get());
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_NE(req.promptText.find("SmolLM"), std::string::npos)
        << "Default system injection expected in: " << req.promptText;
    EXPECT_NE(req.promptText.find("<|im_start|>user"), std::string::npos) << req.promptText;
    EXPECT_NE(req.promptText.find("What is OpenVINO?"), std::string::npos) << req.promptText;
    EXPECT_NE(req.promptText.find("<|im_start|>assistant"), std::string::npos)
        << "Trailing generation prompt expected in: " << req.promptText;
}

TEST_F(ChatTemplateProcessorPyJinjaTest, ExplicitSystemMessage_SuppressesDefaultSystemInjection) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "You are a custom assistant."}});
    history.push_back({{"role", "user"}, {"content", "Hello."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*pyJinjaTokenizer, /*useMinja=*/false, sharedPreparedRuntimeTemplate.get());
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_NE(req.promptText.find("You are a custom assistant."), std::string::npos) << req.promptText;
    EXPECT_EQ(req.promptText.find("SmolLM"), std::string::npos)
        << "Default system injection must be absent when an explicit system message is provided: "
        << req.promptText;
}

TEST_F(ChatTemplateProcessorPyJinjaTest, MultiTurnConversation_AllTurnsRendered) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "Be concise."}});
    history.push_back({{"role", "user"}, {"content", "First question."}});
    history.push_back({{"role", "assistant"}, {"content", "First answer."}});
    history.push_back({{"role", "user"}, {"content", "Second question."}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*pyJinjaTokenizer, /*useMinja=*/false, sharedPreparedRuntimeTemplate.get());
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_NE(req.promptText.find("Be concise."), std::string::npos) << req.promptText;
    EXPECT_NE(req.promptText.find("First question."), std::string::npos) << req.promptText;
    EXPECT_NE(req.promptText.find("First answer."), std::string::npos) << req.promptText;
    EXPECT_NE(req.promptText.find("Second question."), std::string::npos) << req.promptText;

    // Turn order must be preserved.
    const auto firstQ = req.promptText.find("First question.");
    const auto firstA = req.promptText.find("First answer.");
    const auto secondQ = req.promptText.find("Second question.");
    EXPECT_LT(firstQ, firstA) << req.promptText;
    EXPECT_LT(firstA, secondQ) << req.promptText;
}

TEST_F(ChatTemplateProcessorPyJinjaTest, AddGenerationPromptFalse_OmitsGenerationPromptSuffix) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(R"({"add_generation_prompt": false})"));

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*pyJinjaTokenizer, /*useMinja=*/false, sharedPreparedRuntimeTemplate.get());
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_NE(req.promptText.find("What is OpenVINO?"), std::string::npos) << req.promptText;
    EXPECT_EQ(req.promptText.find("<|im_start|>assistant"), std::string::npos)
        << "add_generation_prompt=false must omit the trailing generation prompt: " << req.promptText;
}

TEST_F(ChatTemplateProcessorPyJinjaTest, EmptyStringContent_TemplateStillProducesOutput) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", ""}});

    InputRequest req = makeChatRequest(std::move(history));
    ChatTemplateProcessor processor(*pyJinjaTokenizer, /*useMinja=*/false, sharedPreparedRuntimeTemplate.get());
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(req.promptText.empty());
}
#endif  // PYTHON_DISABLE == 0

}  // namespace
}  // namespace ovms
