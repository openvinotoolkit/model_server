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
    DetermineTaskFromConfigStream,
    CLIParserDetermineTaskTest,
    ::testing::ValuesIn(CLIParserDetermineTaskTest::modelTaskMapping),
    [](const ::testing::TestParamInfo<std::pair<std::string, std::string>>& info) {
        return info.param.first;
    });

TEST_P(CLIParserDetermineTaskTest, DetermineTaskFromConfigStream) {
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
