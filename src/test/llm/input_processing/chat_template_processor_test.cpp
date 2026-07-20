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

// Unit tests for ChatTemplateProcessor — native tokenizer path (useMinja=false).
//
// The PyJinja path is exercised by Python-enabled integration tests and is not
// covered here.
//
// Each test group uses a different tokenizer fixture:
//   - SmolLM2-360M-Instruct: a model WITH a chat template (positive tests).
//   - facebook/opt-125m     : a base model WITHOUT a chat template (negative
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(*sharedTokenizer);
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
    ChatTemplateProcessor processor(tokenizer);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(),
        "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
    EXPECT_TRUE(req.promptText.empty());
}

}  // namespace
}  // namespace ovms
