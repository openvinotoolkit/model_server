//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../default_task.hpp"
#include "../default_task_detector.hpp"

class CLIParserDetermineTaskTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {
public:
    static const std::vector<std::pair<std::string, std::string>> modelTaskMapping;

    static std::string getModelPath(const std::string& modelName) {
        return "src/test/models_config_json/" + modelName;
    }
};

const std::vector<std::pair<std::string, std::string>> CLIParserDetermineTaskTest::modelTaskMapping = {
    {"llama", "text_generation"},
    {"qwen3", "text_generation"},
    {"phi3", "text_generation"},
    {"bge", "embeddings"},
    {"t5_encoder", "embeddings"},
    {"qwen2_embedding", "embeddings"},
    {"cross_encoder", "rerank"},
    {"xlm_roberta", "rerank"},
    {"qwen2_rerank", "rerank"},
    {"stable_diffusion", "image_generation"},
    {"flux", "image_generation"},
    {"speecht5", "text2speech"},
    {"parlertts", "text2speech"},
    {"whisper", "speech2text"},
    {"seamlessm4t", "speech2text"},
    {"qwen3_6", "text_generation"},
    {"qwen3_asr", "speech2text"},
    {"lfm", "text_generation"},
    {"trinity", "text_generation"},
    {"gemma4", "text_generation"},
    {"bge_reranker", "rerank"},
    {"sdxl", "image_generation"},
    {"flux_pipeline", "image_generation"},
    {"qwen3_multi_arch", "speech2text"},
    {"invalid_architecture", ""}  // This model has an unsupported architecture and should throw an exception
};

INSTANTIATE_TEST_SUITE_P(
    DetermineTaskFromModelCatalog,
    CLIParserDetermineTaskTest,
    ::testing::ValuesIn(CLIParserDetermineTaskTest::modelTaskMapping),
    [](const ::testing::TestParamInfo<std::pair<std::string, std::string>>& info) {
        return info.param.first;
    });

TEST_P(CLIParserDetermineTaskTest, DetermineTaskFromModelCatalog) {
    auto [modelName, expectedTask] = GetParam();
    std::string modelDirName = getModelPath(modelName);

    // Test executable is typically two levels below repository root in Bazel outputs.
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path modelPath = std::filesystem::weakly_canonical(currentPath / ".." / ".." / modelDirName);

    if (!std::filesystem::exists(modelPath)) {
        auto searchPath = currentPath;
        while (searchPath != searchPath.parent_path()) {
            std::filesystem::path candidate = searchPath / modelDirName;
            if (std::filesystem::exists(candidate)) {
                modelPath = candidate;
                break;
            }
            searchPath = searchPath.parent_path();
        }
    }

    ASSERT_TRUE(std::filesystem::exists(modelPath))
        << "Model directory not found: " << modelDirName
        << " (tried: " << (currentPath / ".." / ".." / modelDirName).string()
        << ", current_path: " << currentPath.string() << ")";
    ASSERT_TRUE(std::filesystem::exists(modelPath / "config.json") || std::filesystem::exists(modelPath / "model_index.json"))
        << "Neither config.json nor model_index.json found in: " << modelPath.string();

    if (expectedTask.empty()) {
        EXPECT_THROW(
            ovms::determineDefaultTaskParameter(
                std::make_optional(modelPath.string()),
                std::nullopt,
                std::nullopt),
            std::logic_error);
        return;
    }

    std::string result = ovms::determineDefaultTaskParameter(
        std::make_optional(modelPath.string()),
        std::nullopt,
        std::nullopt);

    EXPECT_EQ(result, expectedTask)
        << "Model: " << modelName << ", Expected: " << expectedTask << ", Got: " << result;
}

// ─── Per-detector unit tests (in-memory, no filesystem) ───────────────────────

static ovms::ModelCatalogContext makeCtx(const std::string& identifier,
    const std::string& configJson = "",
    const std::string& modulesJson = "",
    const std::string& modelIndexJson = "") {
    ovms::ModelCatalogContext ctx(std::filesystem::path{}, identifier);
    if (!configJson.empty())
        ctx.addContent("config.json", configJson);
    if (!modulesJson.empty())
        ctx.addContent("modules.json", modulesJson);
    if (!modelIndexJson.empty())
        ctx.addContent("model_index.json", modelIndexJson);
    return ctx;
}

// ── Speech2Text ────────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, Speech2TextDetectsWhisper) {
    auto ctx = makeCtx("whisper", R"({"architectures":["WhisperForConditionalGeneration"]})");
    EXPECT_TRUE(ovms::Speech2TextDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, Speech2TextDetectsSeamlessM4TPrefix) {
    auto ctx = makeCtx("seamless", R"({"architectures":["SeamlessM4Tv2Model"]})");
    EXPECT_TRUE(ovms::Speech2TextDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, Speech2TextDetectsQwen3ASR) {
    auto ctx = makeCtx("qwen3-asr", R"({"architectures":["Qwen3ASRForConditionalGeneration"]})");
    EXPECT_TRUE(ovms::Speech2TextDetector{}.scan(ctx));
}

// ── Text2Speech ────────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, Text2SpeechDetectsParlerTTS) {
    auto ctx = makeCtx("parlertts", R"({"architectures":["ParlerTTSForConditionalGeneration"]})");
    EXPECT_TRUE(ovms::Text2SpeechDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, Text2SpeechDetectsNullArchWithNMels) {
    auto ctx = makeCtx("kokoro", R"({"architectures":null,"n_mels":80})");
    EXPECT_TRUE(ovms::Text2SpeechDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, Text2SpeechDoesNotMatchNullArchWithoutNMels) {
    auto ctx = makeCtx("unknown", R"({"architectures":null})");
    EXPECT_FALSE(ovms::Text2SpeechDetector{}.scan(ctx));
}

// ── Rerank ─────────────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, RerankDetectsViaModulesJson) {
    auto ctx = makeCtx("reranker",
        R"({"architectures":["Qwen3ForCausalLM"]})",
        R"([{"type":"sentence_transformers.cross_encoder.LogitScore"}])");
    EXPECT_TRUE(ovms::RerankDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, RerankDetectsForSequenceClassificationSuffix) {
    auto ctx = makeCtx("bge-reranker", R"({"architectures":["BertForSequenceClassification"]})");
    EXPECT_TRUE(ovms::RerankDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, RerankDetectsQwen3ByNameKeyword) {
    auto ctx = makeCtx("Qwen3-Reranker-0.6B", R"({"architectures":["Qwen3ForCausalLM"]})");
    EXPECT_TRUE(ovms::RerankDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, RerankDoesNotMatchQwen3WithEmbedKeyword) {
    auto ctx = makeCtx("Qwen3-Embedding-0.6B", R"({"architectures":["Qwen3ForCausalLM"]})");
    EXPECT_FALSE(ovms::RerankDetector{}.scan(ctx));
}

// ── ImageGeneration ────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, ImageGenDetectsTransformer2DModelSuffix) {
    auto ctx = makeCtx("flux", R"({"architectures":["FluxTransformer2DModel"]})");
    EXPECT_TRUE(ovms::ImageGenerationDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, ImageGenDetectsCLIPTextModel) {
    auto ctx = makeCtx("clip", R"({"architectures":["CLIPTextModel"]})");
    EXPECT_TRUE(ovms::ImageGenerationDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, ImageGenDetectsModelIndexStableDiffusion) {
    ovms::ModelCatalogContext ctx(std::filesystem::path{}, "sdxl");
    ctx.addContent("model_index.json", R"({"_class_name":"StableDiffusionXLPipeline"})");
    EXPECT_TRUE(ovms::ImageGenerationDetector{}.scan(ctx));
}

// ── Embeddings ─────────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, EmbeddingsDetectsViaModulesJson) {
    auto ctx = makeCtx("Qwen3-Embedding-0.6B",
        R"({"architectures":["Qwen3ForCausalLM"]})",
        R"([{"type":"sentence_transformers.models.Pooling"}])");
    EXPECT_TRUE(ovms::EmbeddingsDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, EmbeddingsDetectsBertModel) {
    auto ctx = makeCtx("bge", R"({"architectures":["BertModel"]})");
    EXPECT_TRUE(ovms::EmbeddingsDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, EmbeddingsDetectsQwen3ByNameKeyword) {
    auto ctx = makeCtx("Qwen3-Embedding-0.6B", R"({"architectures":["Qwen3ForCausalLM"]})");
    EXPECT_TRUE(ovms::EmbeddingsDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, EmbeddingsDoesNotMatchInternVLChatModel) {
    auto ctx = makeCtx("internvl", R"({"architectures":["InternVLChatModel"]})");
    EXPECT_FALSE(ovms::EmbeddingsDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, EmbeddingsDoesNotMatchCLIPTextModel) {
    auto ctx = makeCtx("clip", R"({"architectures":["CLIPTextModel"]})");
    EXPECT_FALSE(ovms::EmbeddingsDetector{}.scan(ctx));
}

// ── TextGeneration ─────────────────────────────────────────────────────────────
TEST(TaskDetectorTest, TextGenDetectsForCausalLMSuffix) {
    auto ctx = makeCtx("llama", R"({"architectures":["LlamaForCausalLM"]})");
    EXPECT_TRUE(ovms::TextGenerationDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, TextGenDetectsInternVLChatModel) {
    auto ctx = makeCtx("internvl", R"({"architectures":["InternVLChatModel"]})");
    EXPECT_TRUE(ovms::TextGenerationDetector{}.scan(ctx));
}
TEST(TaskDetectorTest, TextGenDetectsQwen3WithoutKeyword) {
    auto ctx = makeCtx("Qwen3-8B", R"({"architectures":["Qwen3ForCausalLM"]})");
    EXPECT_TRUE(ovms::TextGenerationDetector{}.scan(ctx));
}

// ── DefaultTaskDetector ordering ──────────────────────────────────────────────
TEST(TaskDetectorTest, DefaultDetectorReturnsTextGenForQwen3NoKeyword) {
    auto ctx = makeCtx("Qwen3-8B", R"({"architectures":["Qwen3ForCausalLM"]})");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "text_generation");
}
TEST(TaskDetectorTest, DefaultDetectorReturnsEmbeddingsForQwen3EmbedKeyword) {
    auto ctx = makeCtx("Qwen3-Embedding-0.6B", R"({"architectures":["Qwen3ForCausalLM"]})");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "embeddings");
}
TEST(TaskDetectorTest, DefaultDetectorReturnsRerankForQwen3RerankKeyword) {
    auto ctx = makeCtx("Qwen3-Reranker-0.6B", R"({"architectures":["Qwen3ForCausalLM"]})");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "rerank");
}
TEST(TaskDetectorTest, DefaultDetectorModulesJsonTakesPrecedenceOverNameKeyword) {
    // modules.json says embeddings → overrides any name-based guess
    auto ctx = makeCtx("some-reranker-model",
        R"({"architectures":["Qwen3ForCausalLM"]})",
        R"([{"type":"sentence_transformers.models.Pooling"}])");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "embeddings");
}
TEST(TaskDetectorTest, DefaultDetectorReturnsEmptyForUnrecognizedArchitecture) {
    auto ctx = makeCtx("unknown-model", R"({"architectures":["SomeBrandNewUnknownArchitecture"]})");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "");
}
TEST(TaskDetectorTest, DefaultDetectorReturnsImageGenForModelIndexOnly) {
    // Diffusers pipeline models (SDXL, Flux) have only model_index.json — no config.json.
    auto ctx = makeCtx("sdxl", "", "", R"({"_class_name":"StableDiffusionXLPipeline"})");
    ovms::DefaultTaskDetector det;
    EXPECT_EQ(det.detect(ctx), "image_generation");
}
