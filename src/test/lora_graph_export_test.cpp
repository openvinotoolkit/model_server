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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "light_test_utils.hpp"
#include "test_with_temp_dir.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../graph_export/graph_export.hpp"
#include "../graph_export/image_generation_graph_cli_parser.hpp"
#include "src/filesystem/filesystem.hpp"
#include "../status.hpp"

class LoraGraphCreationTest : public TestWithTempDir {};

// ===================== LoRA Graph Export Tests =====================

const std::string expectedImageGenWithOneLora = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
          lora_adapters { alias: "pokemon" path: "loras/juliensimon/sd-pokemon-lora/pytorch_lora_weights.safetensors" mode: DYNAMIC }
      }
  }
}

)";

const std::string expectedImageGenWithTwoLoras = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: "GPU"
          max_resolution: "1024x1024"
          lora_adapters { alias: "pokemon" path: "loras/juliensimon/sd-pokemon-lora/model.safetensors" mode: DYNAMIC }
          lora_adapters { alias: "anime-style" path: "loras/org2/anime-lora/weights.safetensors" mode: DYNAMIC }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationWithOneLora) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "juliensimon/sd-pokemon-lora", "pytorch_lora_weights.safetensors"});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenWithOneLora, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

TEST_F(LoraGraphCreationTest, imageGenerationWithTwoLoras) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    imageGenerationGraphSettings.maxResolution = "1024x1024";
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "juliensimon/sd-pokemon-lora", "model.safetensors"});
    imageGenerationGraphSettings.loraAdapters.push_back({"anime-style", "org2/anime-lora", "weights.safetensors"});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenWithTwoLoras, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

const std::string expectedImageGenerationGraphContentsDefault = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationNoLorasRemainsUnchanged) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenerationGraphContentsDefault, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

// ===================== LoRA CLI-to-Settings Tests =====================

TEST(ImageGenCLILoraParsingTest, SingleLoraWithAlias) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=juliensimon/sd-pokemon-lora";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "juliensimon/sd-pokemon-lora");
    EXPECT_FALSE(graphSettings.loraAdapters[0].safetensorsFile.has_value());
}

TEST(ImageGenCLILoraParsingTest, MissingAliasThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "juliensimon/sd-pokemon-lora";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, SingleLoraWithAliasAndFilename) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=juliensimon/sd-pokemon-lora@custom_lora.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "juliensimon/sd-pokemon-lora");
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "custom_lora.safetensors");
}

TEST(ImageGenCLILoraParsingTest, MultipleLoras) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org1/repo1,anime=org2/repo2@weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 2);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "org1/repo1");
    EXPECT_FALSE(graphSettings.loraAdapters[0].safetensorsFile.has_value());
    EXPECT_EQ(graphSettings.loraAdapters[1].alias, "anime");
    EXPECT_EQ(graphSettings.loraAdapters[1].sourceLora, "org2/repo2");
    EXPECT_EQ(graphSettings.loraAdapters[1].safetensorsFile.value(), "weights.safetensors");
}

TEST(ImageGenCLILoraParsingTest, EmptySourceLorasProducesNoAdapters) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 0);
}

TEST(ImageGenCLILoraParsingTest, InvalidEmptyAlias) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "=org/repo";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidEmptyFilenameAfterAt) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/repo@";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, MissingAliasWithFilenameThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "org1/repo1@special.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

// ===================== LoRA Source Type Tests =====================

TEST(ImageGenCLILoraParsingTest, DirectUrlWithAlias) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=https://huggingface.co/juliensimon/sd-pokemon-lora/resolve/main/pytorch_lora_weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "https://huggingface.co/juliensimon/sd-pokemon-lora/resolve/main/pytorch_lora_weights.safetensors");
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "pytorch_lora_weights.safetensors");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::DIRECT_URL);
}

TEST(ImageGenCLILoraParsingTest, DirectUrlHttpWithAlias) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=http://example.com/weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::DIRECT_URL);
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "weights.safetensors");
}

TEST(ImageGenCLILoraParsingTest, DirectUrlMissingAliasThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "https://example.com/weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, DirectUrlNotSafetensorsThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=https://example.com/model.bin";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

class ImageGenCLILoraParsingWithTempDir : public TestWithTempDir {};

TEST_F(ImageGenCLILoraParsingWithTempDir, LocalFileWithAlias) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "test_weights.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    hfSettings.sourceLoras = "pokemon=" + tmpFile;
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, tmpFile);
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "test_weights.safetensors");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::LOCAL_FILE);
}

TEST_F(ImageGenCLILoraParsingWithTempDir, MixedSourceTypes) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "local.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    hfSettings.sourceLoras = "hf=org/repo,url=https://example.com/remote.safetensors,local=" + tmpFile;
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 3);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::HF_REPO);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "hf");
    EXPECT_EQ(graphSettings.loraAdapters[1].sourceType, ovms::LoraSourceType::DIRECT_URL);
    EXPECT_EQ(graphSettings.loraAdapters[1].alias, "url");
    EXPECT_EQ(graphSettings.loraAdapters[1].safetensorsFile.value(), "remote.safetensors");
    EXPECT_EQ(graphSettings.loraAdapters[2].sourceType, ovms::LoraSourceType::LOCAL_FILE);
    EXPECT_EQ(graphSettings.loraAdapters[2].alias, "local");
    EXPECT_EQ(graphSettings.loraAdapters[2].safetensorsFile.value(), "local.safetensors");
}

TEST(ImageGenCLILoraParsingTest, LocalFileMissingAliasThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "/tmp/some_weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, LocalFileNotSafetensorsThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=/tmp/model.bin";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, LocalFileDoesNotExistThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=/nonexistent/path/weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

// ===================== Graph Export with Different Source Types =====================

const std::string expectedImageGenWithUrlLora = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
          lora_adapters { alias: "pokemon" path: "loras/pokemon/pytorch_lora_weights.safetensors" mode: DYNAMIC }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationWithUrlLora) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "https://huggingface.co/juliensimon/sd-pokemon-lora/resolve/main/pytorch_lora_weights.safetensors", "pytorch_lora_weights.safetensors", ovms::LoraSourceType::DIRECT_URL});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenWithUrlLora, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

const std::string expectedImageGenWithLocalLora = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
          lora_adapters { alias: "pokemon" path: "/path/to/weights.safetensors" mode: DYNAMIC }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationWithLocalLora) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "/path/to/weights.safetensors", "weights.safetensors", ovms::LoraSourceType::LOCAL_FILE});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenWithLocalLora, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

TEST_F(LoraGraphCreationTest, imageGenerationHfRepoLoraWithoutFilenameReturnsError) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    // HF_REPO adapter without @filename and without pull — safetensorsFile is nullopt
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "juliensimon/sd-pokemon-lora", std::nullopt, ovms::LoraSourceType::HF_REPO});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

// ===================== Composite LoRA Tests =====================

TEST(ImageGenCLILoraParsingTest, CompositeLoraBasic) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,anime=org/anime-lora,pokemon_anime=@pokemon+@anime";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 2);
    ASSERT_EQ(graphSettings.compositeLoraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].alias, "pokemon_anime");
    ASSERT_EQ(graphSettings.compositeLoraAdapters[0].components.size(), 2);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].components[0].adapterAlias, "pokemon");
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[0].alpha, 1.0f);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].components[1].adapterAlias, "anime");
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[1].alpha, 1.0f);
}

TEST(ImageGenCLILoraParsingTest, CompositeLoraWithAlphas) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,anime=org/anime-lora,blend=@pokemon:0.7+@anime:0.5";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.compositeLoraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].alias, "blend");
    ASSERT_EQ(graphSettings.compositeLoraAdapters[0].components.size(), 2);
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[0].alpha, 0.7f);
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[1].alpha, 0.5f);
}

TEST(ImageGenCLILoraParsingTest, CompositeLoraUnknownRefThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,blend=@pokemon+@nonexistent";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, CompositeLoraInvalidComponentThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,blend=@pokemon+noatsign";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, SingleLoraWithAlpha) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora:0.75";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "pokemon");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "org/pokemon-lora");
    ASSERT_TRUE(graphSettings.loraAdapters[0].alpha.has_value());
    EXPECT_FLOAT_EQ(graphSettings.loraAdapters[0].alpha.value(), 0.75f);
}

TEST(ImageGenCLILoraParsingTest, SingleLoraWithAlphaOne) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora:1.0";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    ASSERT_TRUE(graphSettings.loraAdapters[0].alpha.has_value());
    EXPECT_FLOAT_EQ(graphSettings.loraAdapters[0].alpha.value(), 1.0f);
}

TEST(ImageGenCLILoraParsingTest, InvalidAlphaThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora:abc";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidAlphaPartialFloatThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora:0.5abc";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidAlphaWithFilenameThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/repo@some.safetensors:wikingowie";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidAlphaMultipleDotsThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/repo@some.safetensors:2.5.32";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, UrlLoraWithAlpha) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=https://huggingface.co/org/repo/resolve/main/weights.safetensors:0.5";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "https://huggingface.co/org/repo/resolve/main/weights.safetensors");
    ASSERT_TRUE(graphSettings.loraAdapters[0].alpha.has_value());
    EXPECT_FLOAT_EQ(graphSettings.loraAdapters[0].alpha.value(), 0.5f);
}

TEST(ImageGenCLILoraParsingTest, UrlLoraWithoutAlphaPreservesDefault) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=https://huggingface.co/org/repo/resolve/main/weights.safetensors";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "https://huggingface.co/org/repo/resolve/main/weights.safetensors");
    EXPECT_FALSE(graphSettings.loraAdapters[0].alpha.has_value());
}

TEST(ImageGenCLILoraParsingTest, NPURejectsMultiLoraWithoutComposites) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,anime=org/anime-lora";
    hfSettings.exportSettings.targetDevice = "NPU";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, NPUAllowsCompositeAdapters) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "pokemon=org/pokemon-lora,anime=org/anime-lora,blend=@pokemon:0.5+@anime:0.5";
    hfSettings.exportSettings.targetDevice = "NPU";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_NO_THROW(parser.prepare(serverSettings, hfSettings, "test_model"));
}

const std::string expectedImageGenWithCompositeLora = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
          lora_adapters { alias: "pokemon" path: "loras/org/pokemon-lora/weights.safetensors" mode: DYNAMIC }
          lora_adapters { alias: "anime" path: "loras/org/anime-lora/weights.safetensors" mode: DYNAMIC }
          composite_lora_adapters {
            alias: "blend"
            components { adapter_alias: "pokemon" alpha: 0.7 }
            components { adapter_alias: "anime" alpha: 0.5 }
          }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationWithCompositeLora) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "org/pokemon-lora", "weights.safetensors", ovms::LoraSourceType::HF_REPO});
    imageGenerationGraphSettings.loraAdapters.push_back({"anime", "org/anime-lora", "weights.safetensors", ovms::LoraSourceType::HF_REPO});
    imageGenerationGraphSettings.compositeLoraAdapters.push_back({"blend", {{"pokemon", 0.7f}, {"anime", 0.5f}}});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenWithCompositeLora, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

// ===================== LoRA Alias Validation Tests =====================

TEST(ImageGenCLILoraParsingTest, InvalidAliasWithSpacesThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "my pokemon=org/repo";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidAliasWithSlashThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "my/pokemon=org/repo";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, InvalidAliasWithSpecialCharsThrows) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "poke@mon=org/repo";
    ovms::ImageGenerationGraphCLIParser parser;
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST(ImageGenCLILoraParsingTest, ValidAliasWithHyphensUnderscoresDots) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "my-lora_v1.0=org/repo";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "my-lora_v1.0");
}

// ===================== LoRA Local File Path Tests =====================

TEST_F(ImageGenCLILoraParsingWithTempDir, LocalFileAbsoluteUnixPath) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "model.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    hfSettings.sourceLoras = "pokemon=" + tmpFile;
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::LOCAL_FILE);
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "model.safetensors");
}

#ifdef _WIN32
TEST_F(ImageGenCLILoraParsingWithTempDir, LocalFileWindowsAbsolutePath) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "model.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    // On Windows, directoryPath uses native path with backslashes
    hfSettings.sourceLoras = "pokemon=" + tmpFile;
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::LOCAL_FILE);
}

TEST_F(ImageGenCLILoraParsingWithTempDir, LocalFileWindowsRelativeDotBackslash) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    // Create file at CWD-relative path
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "model.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    // Use .\ relative path (Windows-style)
    hfSettings.sourceLoras = "pokemon=.\\" + std::filesystem::path(tmpFile).filename().string();
    ovms::ImageGenerationGraphCLIParser parser;
    // This will throw because relative path won't resolve to existing file from CWD,
    // but it should at least be detected as LOCAL_FILE source type (i.e. not HF_REPO)
    EXPECT_THROW(parser.prepare(serverSettings, hfSettings, "test_model"), std::invalid_argument);
}

TEST_F(ImageGenCLILoraParsingWithTempDir, LocalFileWindowsAbsolutePathWithAlpha) {
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    std::string tmpFile = ovms::FileSystem::joinPath({this->directoryPath, "model.safetensors"});
    {
        std::ofstream f(tmpFile);
        f << "test";
    }
    // Windows path with alpha: C:\path\to\model.safetensors:0.6
    // The drive letter colon (C:) must not be confused with alpha separator
    hfSettings.sourceLoras = "pokemon=" + tmpFile + ":0.6";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "test_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    ASSERT_EQ(graphSettings.loraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceType, ovms::LoraSourceType::LOCAL_FILE);
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "model.safetensors");
    ASSERT_TRUE(graphSettings.loraAdapters[0].alpha.has_value());
    EXPECT_FLOAT_EQ(graphSettings.loraAdapters[0].alpha.value(), 0.6f);
}
#endif

// ===================== Full Composite LoRA End-to-End CLI Test =====================

TEST(ImageGenCLILoraParsingTest, FullCompositeWithAlphasAndTwoLoras) {
    // Two individual LoRAs + one composite referencing both with explicit alphas
    ovms::ServerSettingsImpl serverSettings;
    serverSettings.serverMode = ovms::HF_PULL_MODE;
    ovms::HFSettingsImpl hfSettings;
    hfSettings.sourceLoras = "landscape=civitai/landscapes-lora@Fantastic_Landscape.safetensors,"
                             "portrait=org/portrait-lora,"
                             "scenic_blend=@landscape:0.3+@portrait:0.8";
    ovms::ImageGenerationGraphCLIParser parser;
    parser.prepare(serverSettings, hfSettings, "sd_model");
    auto& graphSettings = std::get<ovms::ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);

    // Verify individual LoRAs
    ASSERT_EQ(graphSettings.loraAdapters.size(), 2);
    EXPECT_EQ(graphSettings.loraAdapters[0].alias, "landscape");
    EXPECT_EQ(graphSettings.loraAdapters[0].sourceLora, "civitai/landscapes-lora");
    EXPECT_EQ(graphSettings.loraAdapters[0].safetensorsFile.value(), "Fantastic_Landscape.safetensors");
    EXPECT_EQ(graphSettings.loraAdapters[1].alias, "portrait");
    EXPECT_EQ(graphSettings.loraAdapters[1].sourceLora, "org/portrait-lora");

    // Verify composite
    ASSERT_EQ(graphSettings.compositeLoraAdapters.size(), 1);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].alias, "scenic_blend");
    ASSERT_EQ(graphSettings.compositeLoraAdapters[0].components.size(), 2);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].components[0].adapterAlias, "landscape");
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[0].alpha, 0.3f);
    EXPECT_EQ(graphSettings.compositeLoraAdapters[0].components[1].adapterAlias, "portrait");
    EXPECT_FLOAT_EQ(graphSettings.compositeLoraAdapters[0].components[1].alpha, 0.8f);
}

const std::string expectedImageGenFullComposite = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: ""
          lora_adapters { alias: "landscape" path: "loras/civitai/landscapes-lora/Fantastic_Landscape.safetensors" mode: DYNAMIC }
          lora_adapters { alias: "portrait" path: "loras/org/portrait-lora/weights.safetensors" mode: DYNAMIC }
          composite_lora_adapters {
            alias: "scenic_blend"
            components { adapter_alias: "landscape" alpha: 0.3 }
            components { adapter_alias: "portrait" alpha: 0.8 }
          }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationFullCompositeWithAlphas) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"landscape", "civitai/landscapes-lora", "Fantastic_Landscape.safetensors", ovms::LoraSourceType::HF_REPO});
    imageGenerationGraphSettings.loraAdapters.push_back({"portrait", "org/portrait-lora", "weights.safetensors", ovms::LoraSourceType::HF_REPO});
    imageGenerationGraphSettings.compositeLoraAdapters.push_back({"scenic_blend", {{"landscape", 0.3f}, {"portrait", 0.8f}}});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenFullComposite, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

const std::string expectedImageGenNpuStatic = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: "NPU"
          lora_adapters { alias: "pokemon" path: "loras/org/pokemon-lora/weights.safetensors" alpha: 0.8 mode: STATIC }
      }
  }
}

)";

TEST_F(LoraGraphCreationTest, imageGenerationNpuAutoStaticMode) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    hfSettings.exportSettings.targetDevice = "NPU";
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    imageGenerationGraphSettings.loraAdapters.push_back({"pokemon", "org/pokemon-lora", "weights.safetensors", ovms::LoraSourceType::HF_REPO, 0.8f});
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenNpuStatic, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}
