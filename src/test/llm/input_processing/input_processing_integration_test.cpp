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

// Integration tests for the full input processing pipeline:
//   JSON string
//   → OpenAIApiHandler::parseRequest()        (wire format → chatHistory + request fields)
//   → OpenAIApiHandler::extractInputRequest() (chatHistory → inputRequest + generationConfig)
//   → InputProcessor::process()              (ImageDecoding / TextNorm / ChatTemplate / Tokenize)
//
// The tokenizer is loaded once per test suite from a known test fixture
// (SmolLM2-360M-Instruct). No inference is performed.

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>

#include "../../../llm/apis/openai_completions.hpp"
#include "../../../llm/apis/openai_responses.hpp"
#include "../../../llm/io_processing/base_generation_config_builder.hpp"
#include "../../../llm/io_processing/generation_config_builder.hpp"
#include "../../../llm/io_processing/input_processor.hpp"
#include "../../../llm/io_processing/input_processor_context.hpp"
#include "../../../llm/io_processing/input_request.hpp"
#include "../../platform_utils.hpp"

namespace ovms {
namespace {

// SmolLM2-360M-Instruct injects this default system turn when the first message
// is not 'system' (verified from tokenizer_config.json chat_template).
constexpr const char* SMOL_DEFAULT_SYSTEM =
    "<|im_start|>system\n"
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    "<|im_end|>\n";

// Minimal 1×1 PNG (same fixture used throughout the handler test suite).
constexpr const char* TINY_PNG =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg==";

// ---------------------------------------------------------------------------
// Shared fixture
// ---------------------------------------------------------------------------

static std::unique_ptr<ov::genai::Tokenizer> sharedTokenizer;

class InputProcessingIntegrationTest : public testing::TestWithParam<Endpoint> {
protected:
    static void SetUpTestSuite() {
        sharedTokenizer = std::make_unique<ov::genai::Tokenizer>(getGenericFullPathForSrcTest(
            "/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct"));
    }

    static void TearDownTestSuite() {
        sharedTokenizer.reset();
    }

    // ------------------------------------------------------------------
    // Run result — owns the Document so the handler's reference stays valid.
    // ------------------------------------------------------------------
    struct RunResult {
        absl::Status parseStatus;
        absl::Status processStatus;
        InputRequest req;
        std::shared_ptr<OpenAIApiHandler> handler;
        std::shared_ptr<rapidjson::Document> doc;  // keeps handler's reference alive
    };

    RunResult runPipeline(const std::string& json, bool isVLM = false) {
        RunResult result;

        result.doc = std::make_shared<rapidjson::Document>();
        result.doc->Parse(json.c_str());
        if (result.doc->HasParseError()) {
            result.parseStatus = absl::InvalidArgumentError("JSON parse error");
            return result;
        }

        auto& doc = *result.doc;
        auto now = std::chrono::system_clock::now();
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            result.handler = std::make_shared<OpenAIChatCompletionsHandler>(
                doc, GetParam(), now, *sharedTokenizer);
        } else {
            result.handler = std::make_shared<OpenAIResponsesHandler>(
                doc, GetParam(), now, *sharedTokenizer);
        }

        result.parseStatus = result.handler->parseRequest(
            /*maxTokensLimit=*/std::nullopt,
            /*bestOfLimit=*/0,
            /*maxModelLength=*/std::nullopt);
        if (!result.parseStatus.ok()) {
            return result;
        }

        GenerationConfigBuilder configBuilder(
            ov::genai::GenerationConfig{}, /*toolParserName=*/"",
            /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD);
        auto reqResult = result.handler->extractInputRequest(configBuilder);
        if (!reqResult.ok()) {
            result.parseStatus = reqResult.status();
            return result;
        }
        result.req = std::move(*reqResult);

        InputProcessorContext ctx;
        ctx.config.isVLM = isVLM;
        ctx.config.useMinja = false;
        ctx.tokenizer = *sharedTokenizer;

        InputProcessor processor(ctx, result.req);
        result.processStatus = processor.process(result.req);
        return result;
    }

    // ------------------------------------------------------------------
    // JSON builders — produce the correct wire format per endpoint.
    // ------------------------------------------------------------------

    // Single text message, with optional extra top-level fields.
    std::string textJson(const std::string& text, const std::string& extraFields = "") const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            return R"({"model":"m","messages":[{"role":"user","content":")" +
                   text + R"("}])" + extraFields + "}";
        }
        return R"({"model":"m","input":[{"type":"message","role":"user","content":[)"
               R"({"type":"input_text","text":")" +
               text + R"("}]}])" + extraFields + "}";
    }

    // Single message: one text part + one image.
    std::string singleImageJson(const std::string& imageUrl) const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            return R"({"model":"m","messages":[{"role":"user","content":[)"
                   R"({"type":"text","text":"Describe this."},)"
                   R"({"type":"image_url","image_url":{"url":")" +
                   imageUrl + R"("}}]}]})";
        }
        return R"({"model":"m","input":[{"type":"message","role":"user","content":[)"
               R"({"type":"input_text","text":"Describe this."},)"
               R"({"type":"input_image","image_url":{"url":")" +
               imageUrl + R"("}}]}]})";
    }

    // Single message: text + N images.
    std::string multiImageJson(const std::vector<std::string>& urls) const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            std::string parts = R"({"type":"text","text":"Describe all."})";
            for (const auto& url : urls)
                parts += R"(,{"type":"image_url","image_url":{"url":")" + url + R"("}})";
            return R"({"model":"m","messages":[{"role":"user","content":[)" + parts + R"(]}]})";
        }
        std::string parts = R"({"type":"input_text","text":"Describe all."})";
        for (const auto& url : urls)
            parts += R"(,{"type":"input_image","image_url":{"url":")" + url + R"("}})";
        return R"({"model":"m","input":[{"type":"message","role":"user","content":[)" + parts + R"(]}]})";
    }

    // Single message: text — image — text (interleaved).
    std::string interleavedJson(const std::string& imageUrl) const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            return R"({"model":"m","messages":[{"role":"user","content":[)"
                   R"({"type":"text","text":"Before."},)"
                   R"({"type":"image_url","image_url":{"url":")" +
                   imageUrl + R"("}},)"
                              R"({"type":"text","text":"After."}]}]})";
        }
        return R"({"model":"m","input":[{"type":"message","role":"user","content":[)"
               R"({"type":"input_text","text":"Before."},)"
               R"({"type":"input_image","image_url":{"url":")" +
               imageUrl + R"("}},)"
                          R"({"type":"input_text","text":"After."}]}]})";
    }

    // Multi-turn: system + user + assistant + user-with-image.
    std::string multiTurnWithImageJson(const std::string& imageUrl) const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            return R"({"model":"m","messages":[)"
                   R"({"role":"system","content":"You are helpful."},)"
                   R"({"role":"user","content":"First question."},)"
                   R"({"role":"assistant","content":"That's a great question!"},)"
                   R"({"role":"user","content":[)"
                   R"({"type":"image_url","image_url":{"url":")" +
                   imageUrl + R"("}},)"
                              R"({"type":"text","text":"Describe it."}]}]})";
        }
        return R"({"model":"m","input":[)"
               R"({"type":"message","role":"system","content":[{"type":"input_text","text":"You are helpful."}]},)"
               R"({"type":"message","role":"user","content":[{"type":"input_text","text":"First question."}]},)"
               R"({"type":"message","role":"assistant","content":[{"type":"output_text","text":"That's a great question!"}]},)"
               R"({"type":"message","role":"user","content":[)"
               R"({"type":"input_image","image_url":{"url":")" +
               imageUrl + R"("}},)"
                          R"({"type":"input_text","text":"Describe it."}]}]})";
    }

    // Text-only content array (two text parts, no images).
    std::string textArrayJson() const {
        if (GetParam() == Endpoint::CHAT_COMPLETIONS) {
            return R"({"model":"m","messages":[{"role":"user","content":[)"
                   R"({"type":"text","text":"First part."},)"
                   R"({"type":"text","text":"Second part."}]}]})";
        }
        return R"({"model":"m","input":[{"type":"message","role":"user","content":[)"
               R"({"type":"input_text","text":"First part."},)"
               R"({"type":"input_text","text":"Second part."}]}]})";
    }

    // Extra fields string that works for both endpoints.
    std::string genConfigExtra() const {
        const std::string maxField = (GetParam() == Endpoint::RESPONSES)
                                         ? "\"max_output_tokens\""
                                         : "\"max_completion_tokens\"";
        return "," + maxField + R"(:42,"temperature":0.7,"top_p":0.85)";
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_P(InputProcessingIntegrationTest, TextOnly_PromptAndGenConfigPopulated) {
    auto result = runPipeline(textJson("What is OpenVINO?", genConfigExtra()));

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\nWhat is OpenVINO?<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(result.req.promptText, expected);
    EXPECT_GT(result.req.inputIds.get_size(), 0u);
    EXPECT_TRUE(result.req.inputImages.empty());
    EXPECT_EQ(result.handler->getMaxTokens(), 42);
    EXPECT_NEAR(result.req.generationConfig.temperature, 0.7f, 1e-5f);
    EXPECT_NEAR(result.req.generationConfig.top_p, 0.85f, 1e-5f);
}

TEST_P(InputProcessingIntegrationTest, SingleBase64Image_DecodedAndTagInPrompt) {
    auto result = runPipeline(singleImageJson(TINY_PNG), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    // ImageDecodingProcessor accumulates image tags before text:
    // content = "<ov_genai_image_0>\nDescribe this."
    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\n<ov_genai_image_0>\nDescribe this.<|im_end|>\n"
        "<|im_start|>assistant\n";
    ASSERT_EQ(result.req.inputImages.size(), 1u);
    EXPECT_EQ(result.req.promptText, expected);
    EXPECT_GT(result.req.inputIds.get_size(), 0u);
}

TEST_P(InputProcessingIntegrationTest, ThreeImages_AllDecodedWithCorrectTagIndices) {
    auto result = runPipeline(multiImageJson({TINY_PNG, TINY_PNG, TINY_PNG}), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    // All image tags precede the text part (processor accumulation order).
    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\n"
        "<ov_genai_image_0>\n<ov_genai_image_1>\n<ov_genai_image_2>\n"
        "Describe all.<|im_end|>\n"
        "<|im_start|>assistant\n";
    ASSERT_EQ(result.req.inputImages.size(), 3u);
    EXPECT_EQ(result.req.promptText, expected);
}

TEST_P(InputProcessingIntegrationTest, InterleavedTextAndImage_BothTextPartsInPrompt) {
    auto result = runPipeline(interleavedJson(TINY_PNG), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    // [text("Before."), image, text("After.")] → image tag before joined text.
    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\n<ov_genai_image_0>\nBefore.\nAfter.<|im_end|>\n"
        "<|im_start|>assistant\n";
    ASSERT_EQ(result.req.inputImages.size(), 1u);
    EXPECT_EQ(result.req.promptText, expected);
}

TEST_P(InputProcessingIntegrationTest, MultiTurnConversation_AllTurnsInPrompt) {
    auto result = runPipeline(multiTurnWithImageJson(TINY_PNG), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    // First message is 'system' → no default system injection by the template.
    // ImageDecodingProcessor also flattens the Responses endpoint's assistant
    // text array ([{type:text,text:"..."}]) to a plain string before ChatTemplateProcessor.
    const std::string expected =
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nFirst question.<|im_end|>\n"
        "<|im_start|>assistant\nThat's a great question!<|im_end|>\n"
        "<|im_start|>user\n<ov_genai_image_0>\nDescribe it.<|im_end|>\n"
        "<|im_start|>assistant\n";
    ASSERT_EQ(result.req.inputImages.size(), 1u);
    EXPECT_EQ(result.req.promptText, expected);
}

TEST_P(InputProcessingIntegrationTest, GenerationConfigFields_SurviveFullPipeline) {
    auto result = runPipeline(textJson("Hello.", genConfigExtra()));

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_TRUE(result.processStatus.ok()) << result.processStatus.message();

    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\nHello.<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(result.req.promptText, expected);
    EXPECT_EQ(result.handler->getMaxTokens(), 42);
    EXPECT_NEAR(result.req.generationConfig.temperature, 0.7f, 1e-5f);
    EXPECT_NEAR(result.req.generationConfig.top_p, 0.85f, 1e-5f);
}

TEST_P(InputProcessingIntegrationTest, TextArrayContent_FlattenedByNormalizationProcessor) {
    // A text-only content array must be flattened by TextContentNormalizationProcessor
    // so both strings appear in the final prompt. This holds on both the LM path and the
    // VLM path used without images (VLM is a superset of LM).
    // TextContentNormalizationProcessor joins the two parts with \n before the template runs.
    const std::string expected =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\nFirst part.\nSecond part.<|im_end|>\n"
        "<|im_start|>assistant\n";

    for (bool isVLM : {false, true}) {
        auto result = runPipeline(textArrayJson(), isVLM);

        ASSERT_TRUE(result.parseStatus.ok()) << "isVLM=" << isVLM << ": " << result.parseStatus.message();
        ASSERT_TRUE(result.processStatus.ok()) << "isVLM=" << isVLM << ": " << result.processStatus.message();
        EXPECT_EQ(result.req.promptText, expected) << "isVLM=" << isVLM;
        EXPECT_TRUE(result.req.inputImages.empty()) << "isVLM=" << isVLM;
    }
}

// ---------------------------------------------------------------------------
// Negative tests — verify that pipeline errors are surfaced correctly and
// that downstream stages are NOT executed when an upstream stage fails.
// ---------------------------------------------------------------------------

// When handler parsing fails (e.g. empty input), the InputProcessor chain
// must never run: promptText and inputImages stay at their zero-values.
TEST_P(InputProcessingIntegrationTest, HandlerParseError_ProcessorNeverRuns) {
    // Empty messages / input array is rejected by each endpoint's parseRequest().
    const std::string json = (GetParam() == Endpoint::CHAT_COMPLETIONS)
                                 ? R"({"model":"m","messages":[]})"
                                 : R"({"model":"m","input":[]})";

    auto result = runPipeline(json);

    ASSERT_FALSE(result.parseStatus.ok());
    // Processors never ran: promptText and inputImages remain empty.
    EXPECT_TRUE(result.req.promptText.empty());
    EXPECT_TRUE(result.req.inputImages.empty());
}

// When string content already contains an <ov_genai_image_N> tag,
// ImageDecodingProcessor's injection guard must reject the request.
// parseRequest() must have already succeeded (handler has no such restriction).
// Chat uses flat-string content; Responses uses an array text-part — both
// exercise different code paths in the injection guard.
TEST_P(InputProcessingIntegrationTest, InjectionGuardInContent_ProcessingFails) {
    auto result = runPipeline(textJson("look at <ov_genai_image_0> this"), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_FALSE(result.processStatus.ok());
    EXPECT_EQ(result.processStatus.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(result.processStatus.message(), "Message contains restricted <ov_genai_image> tag");
    EXPECT_TRUE(result.req.promptText.empty());
    EXPECT_TRUE(result.req.inputImages.empty());
}

// When the base64 payload of an image URL is malformed, ImageDecodingProcessor
// must fail after the handler has successfully accepted the request.
TEST_P(InputProcessingIntegrationTest, InvalidBase64Image_ImageDecodingFails) {
    auto result = runPipeline(singleImageJson("data:image/jpeg;base64,NOT_VALID_BASE64!!!"), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_FALSE(result.processStatus.ok());
    EXPECT_EQ(result.processStatus.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(result.processStatus.message(), "Invalid base64 string in request");
    EXPECT_TRUE(result.req.inputImages.empty());
    EXPECT_TRUE(result.req.promptText.empty());
}

// When an HTTP image URL is provided but no allowed domain list is configured,
// ImageDecodingProcessor must reject the URL after the handler accepts it.
// This tests the boundary between handler (wire-format acceptance) and
// ImageDecodingProcessor (policy enforcement).
TEST_P(InputProcessingIntegrationTest, HttpImageUrlNoAllowedDomains_ImageDecodingFails) {
    auto result = runPipeline(singleImageJson("http://example.com/photo.jpg"), /*isVLM=*/true);

    ASSERT_TRUE(result.parseStatus.ok()) << result.parseStatus.message();
    ASSERT_FALSE(result.processStatus.ok());
    EXPECT_EQ(result.processStatus.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(result.processStatus.message(),
        "Given url does not match any allowed domain from allowed_media_domains");
    EXPECT_TRUE(result.req.inputImages.empty());
    EXPECT_TRUE(result.req.promptText.empty());
}

INSTANTIATE_TEST_SUITE_P(BothEndpoints, InputProcessingIntegrationTest,
    testing::Values(Endpoint::CHAT_COMPLETIONS, Endpoint::RESPONSES),
    [](const testing::TestParamInfo<Endpoint>& info) {
        return info.param == Endpoint::CHAT_COMPLETIONS ? "ChatCompletions" : "Responses";
    });

// ---------------------------------------------------------------------------
// Endpoint equivalence — non-parameterized, runs both explicitly.
// ---------------------------------------------------------------------------

TEST(InputProcessingEquivalenceTest, ChatAndResponsesProduceSamePromptAndImages) {
    const std::string tokPath = getGenericFullPathForSrcTest(
        "/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct");
    // The tokenizer is shared between both endpoint runs in this one test.
    ov::genai::Tokenizer tokenizer(tokPath);

    const auto run = [&](Endpoint ep) -> std::pair<std::string, size_t> {
        const std::string chatJson =
            R"({"model":"m","messages":[{"role":"user","content":[)"
            R"({"type":"text","text":"Describe this."},)"
            R"({"type":"image_url","image_url":{"url":")" +
            std::string(TINY_PNG) + R"("}}]}]})";
        const std::string responsesJson =
            R"({"model":"m","input":[{"type":"message","role":"user","content":[)"
            R"({"type":"input_text","text":"Describe this."},)"
            R"({"type":"input_image","image_url":{"url":")" +
            std::string(TINY_PNG) + R"("}}]}]})";

        auto doc = std::make_shared<rapidjson::Document>();
        doc->Parse((ep == Endpoint::CHAT_COMPLETIONS ? chatJson : responsesJson).c_str());

        std::shared_ptr<OpenAIApiHandler> handler;
        auto now = std::chrono::system_clock::now();
        if (ep == Endpoint::CHAT_COMPLETIONS) {
            handler = std::make_shared<OpenAIChatCompletionsHandler>(*doc, ep, now, tokenizer);
        } else {
            handler = std::make_shared<OpenAIResponsesHandler>(*doc, ep, now, tokenizer);
        }

        EXPECT_TRUE(handler->parseRequest(std::nullopt, 0, std::nullopt).ok());

        GenerationConfigBuilder configBuilder(
            ov::genai::GenerationConfig{}, "", false, DecodingMethod::STANDARD);
        auto reqResult = handler->extractInputRequest(configBuilder);
        EXPECT_TRUE(reqResult.ok());
        auto req = std::move(*reqResult);

        InputProcessorContext ctx;
        ctx.config.isVLM = true;
        ctx.config.useMinja = false;
        ctx.tokenizer = tokenizer;

        InputProcessor processor(ctx, req);
        EXPECT_TRUE(processor.process(req).ok());
        return {req.promptText, req.inputImages.size()};
    };

    auto [chatPrompt, chatImages] = run(Endpoint::CHAT_COMPLETIONS);
    auto [responsesPrompt, responsesImages] = run(Endpoint::RESPONSES);

    const std::string expectedPrompt =
        std::string(SMOL_DEFAULT_SYSTEM) +
        "<|im_start|>user\n<ov_genai_image_0>\nDescribe this.<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(chatPrompt, expectedPrompt);
    EXPECT_EQ(responsesPrompt, expectedPrompt);
    EXPECT_EQ(chatImages, 1u);
    EXPECT_EQ(responsesImages, 1u);
}

}  // namespace
}  // namespace ovms
