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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_with_temp_dir.hpp"
#include "light_test_utils.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../graph_export/graph_export.hpp"
#include "../filesystem.hpp"
#include "../status.hpp"
#include "../version.hpp"

const std::string expectedOneSettingPluginGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            plugin_config: '{"KV_CACHE_PRECISION":"u8"}',
            enable_prefix_caching: true,
            cache_size: 10,
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
    }
)";

const std::string expectedFullPluginGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            plugin_config: '{"KV_CACHE_PRECISION":"u8","MAX_PROMPT_LEN":123,"MODEL_DISTRIBUTION_POLICY":"PIPELINE_PARALLEL"}',
            enable_prefix_caching: true,
            cache_size: 10,
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
    }
)";

const std::string expectedGraphContentsWithResponseParser = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 10,
            reasoning_parser: "REASONING_PARSER",
            tool_parser: "TOOL_PARSER",
            enable_tool_guided_generation: true,
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
    }
)";

const std::string expectedDefaultGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 10,
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
    }
)";

const std::string expectedDraftAndFuseGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 10,
            dynamic_split_fuse: false,
            # Speculative decoding configuration
            draft_models_path: "-ovms-src-test-llm_testing-facebook-opt-125m",
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
    }
)";

const std::string expectedGGUFGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./PRETTY_GOOD_GGUF_MODEL.gguf",
            enable_prefix_caching: true,
            cache_size: 10,
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
    }
)";

const std::string expectedGGUFGraphContents2 = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./PRETTY_GOOD_GGUF_MODEL_Q8-00001-of-20000.gguf",
            enable_prefix_caching: true,
            cache_size: 10,
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
    }
)";

const std::string expectedRerankGraphContentsNonDefault = R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: "myModel",
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {
            models_path: "/some/path",
            max_allowed_chunks: 18,
            target_device: "GPU",
            plugin_config: '{"NUM_STREAMS":"2"}',
        }
    }
}
)";

const std::string expectedRerankGraphContentsDefault = R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: "",
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {
            models_path: "./",
            max_allowed_chunks: 10000,
            target_device: "CPU",
            plugin_config: '{"NUM_STREAMS":"1"}',
        }
    }
}
)";

const std::string expectedEmbeddingsGraphContents = R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: "myModel",
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
            models_path: "/model1/path",
            normalize_embeddings: false,
            truncate: true,
            pooling: LAST,
            target_device: "GPU",
            plugin_config: '{"NUM_STREAMS":"2"}',
        }
    }
}
)";

const std::string expectedEmbeddingsGraphContentsDefault = R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: "",
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
            models_path: "./",
            normalize_embeddings: true,
            truncate: false,
            pooling: CLS,
            target_device: "CPU",
            plugin_config: '{"NUM_STREAMS":"1"}',
        }
    }
}
)";

const std::string expectedTextToSpeechGraphContents = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: "myModel"
    calculator: "T2sCalculator"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/model1/path"
            target_device: "GPU"
            plugin_config: '{"NUM_STREAMS":"2"}'
        }
    }
}
)";

const std::string expectedTextToSpeechGraphContentsDefault = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: ""
    calculator: "T2sCalculator"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "./"
            target_device: "CPU"
            }
    }
}
)";

const std::string expectedSpeechToTextGraphContents = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: "myModel"
    calculator: "S2tCalculator"
    input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {
            models_path: "/model1/path"
            target_device: "GPU"
            plugin_config: '{"NUM_STREAMS":"2"}'
        }
    }
}
)";

const std::string expectedSpeechToTextGraphContentsDefault = R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: ""
    calculator: "S2tCalculator"
    input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {
            models_path: "./"
            target_device: "CPU"
            }
    }
}
)";

const std::string expectedImageGenerationGraphContents = R"(
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
          plugin_config: '{"NUM_STREAMS":"14","CACHE_DIR":"/cache"}'
          max_resolution: "3000x4000"
          default_resolution: "300x400"
          max_num_images_per_prompt: 7
          default_num_inference_steps: 2
          max_num_inference_steps: 3
      }
  }
}

)";

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
          device: "CPU"
      }
  }
}

)";

class GraphCreationTest : public TestWithTempDir {
protected:
    void TearDown() {
        TestWithTempDir::TearDown();
    }

    // Removes # OpenVINO Model Server REPLACE_PROJECT_VERSION comment added for debug purpose in graph export at the begging of graph.pbtxt
    // This string differs per build and setup
    std::string removeVersionString(std::string input) {
        return input.erase(0, input.find("\n") + 1);
    }

    std::string getVersionString() {
        std::stringstream expected;
        expected << "# File created with: " << PROJECT_NAME << " " << PROJECT_VERSION << std::endl;
        return expected.str();
    }
};

TEST_F(GraphCreationTest, positiveDefaultWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedDefaultGraphContents;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveRerankWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.exportSettings.pluginConfig.numStreams = 1;
    hfSettings.task = ovms::RERANK_GRAPH;
    ovms::RerankGraphSettingsImpl rerankGraphSettings;
    hfSettings.graphSettings = std::move(rerankGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedRerankGraphContentsDefault;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveEmbeddingsWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.exportSettings.pluginConfig.numStreams = 1;
    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    ovms::EmbeddingsGraphSettingsImpl embeddingsGraphSettings;
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedEmbeddingsGraphContentsDefault;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveTextToSpeechWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    ovms::TextToSpeechGraphSettingsImpl textToSpeechGraphSettings;
    hfSettings.graphSettings = std::move(textToSpeechGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedTextToSpeechGraphContentsDefault;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveSTTWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    ovms::SpeechToTextGraphSettingsImpl speechToTextGraphSettings;
    hfSettings.graphSettings = std::move(speechToTextGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedSpeechToTextGraphContentsDefault;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveImageGenWithVersionString) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::string expected = getVersionString() + expectedImageGenerationGraphContentsDefault;
    ASSERT_EQ(expected, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedDefaultGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, positiveDraftAndFuse) {
    ovms::HFSettingsImpl hfSettings;
    ovms::TextGenGraphSettingsImpl graphSettings;
    graphSettings.draftModelDirName = "/ovms/src/test/llm_testing/facebook/opt-125m";
    graphSettings.dynamicSplitFuse = "false";

    hfSettings.graphSettings = std::move(graphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedDraftAndFuseGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, positiveGGUF) {
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    ovms::HFSettingsImpl hfSettings;
    hfSettings.ggufFilename = "PRETTY_GOOD_GGUF_MODEL.gguf";
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedGGUFGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, WillOverwriteExistingGraphPbtxtGGUF) {
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    ovms::HFSettingsImpl hfSettings;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();

    hfSettings.ggufFilename = "PRETTY_GOOD_GGUF_MODEL.gguf";
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedGGUFGraphContents, removeVersionString(graphContents)) << graphContents;

    hfSettings.ggufFilename = "PRETTY_GOOD_GGUF_MODEL_Q8-00001-of-20000.gguf";
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedGGUFGraphContents2, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, rerankPositiveNonDefault) {
    ovms::HFSettingsImpl hfSettings;
    auto& exportSettings = hfSettings.exportSettings;
    hfSettings.task = ovms::RERANK_GRAPH;
    ovms::RerankGraphSettingsImpl rerankGraphSettings;
    exportSettings.targetDevice = "GPU";
    exportSettings.modelName = "myModel";
    exportSettings.modelPath = "/some/path";
    exportSettings.pluginConfig.numStreams = 2;
    rerankGraphSettings.maxAllowedChunks = 18;
    hfSettings.graphSettings = std::move(rerankGraphSettings);

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedRerankGraphContentsNonDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, rerankPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.exportSettings.pluginConfig.numStreams = 1;
    hfSettings.task = ovms::RERANK_GRAPH;
    ovms::RerankGraphSettingsImpl rerankGraphSettings;
    hfSettings.graphSettings = std::move(rerankGraphSettings);

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedRerankGraphContentsDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, rerankCreatedPbtxtInvalid) {
    ovms::HFSettingsImpl hfSettings;
    auto& exportSettings = hfSettings.exportSettings;
    hfSettings.task = ovms::RERANK_GRAPH;
    ovms::RerankGraphSettingsImpl rerankGraphSettings;
    exportSettings.targetDevice = "GPU";
    exportSettings.modelName = "myModel\"";
    exportSettings.pluginConfig.numStreams = 2;
    hfSettings.graphSettings = std::move(rerankGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
#if (MEDIAPIPE_DISABLE == 0)
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
#else
    ASSERT_EQ(status, ovms::StatusCode::OK);
#endif
}

TEST_F(GraphCreationTest, embeddingsPositiveNonDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    ovms::EmbeddingsGraphSettingsImpl embeddingsGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel";
    hfSettings.exportSettings.modelPath = "/model1/path";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    embeddingsGraphSettings.normalize = "false";
    embeddingsGraphSettings.truncate = "true";
    embeddingsGraphSettings.pooling = "LAST";
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedEmbeddingsGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, embeddingsPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    ovms::EmbeddingsGraphSettingsImpl embeddingsGraphSettings;
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
    hfSettings.exportSettings.pluginConfig.numStreams = 1;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedEmbeddingsGraphContentsDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, embeddingsCreatedPbtxtInvalid) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    ovms::EmbeddingsGraphSettingsImpl embeddingsGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel\"";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    embeddingsGraphSettings.normalize = "true";
    embeddingsGraphSettings.pooling = "CLS";
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
#if (MEDIAPIPE_DISABLE == 0)
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
#else
    ASSERT_EQ(status, ovms::StatusCode::OK);
#endif
}
TEST_F(GraphCreationTest, embeddingsDoubleSetNumStreams) {
    // by default for embeddings we set numStreams=1 in CLI
    // we should ignore double setting & check if equals the one from `--plugin_config`
    // if both `--num_streams` is used to change from 1 and `--plugin_config` is used with
    // num streams we trigger error
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    ovms::EmbeddingsGraphSettingsImpl embeddingsGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel";
    hfSettings.exportSettings.pluginConfig.numStreams = 1;  // imitates default from CLI
    // note here we use int not string num_streams so this shows way to overwrite type
    hfSettings.exportSettings.pluginConfig.manualString = "{\"NUM_STREAMS\":1}";
    embeddingsGraphSettings.normalize = "true";
    embeddingsGraphSettings.pooling = "CLS";
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    hfSettings.exportSettings.pluginConfig.numStreams = 2;  // non-default value - it should fail
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS) << status.string();
}

TEST_F(GraphCreationTest, textToSpeechPositiveNonDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    ovms::TextToSpeechGraphSettingsImpl textToSpeechGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel";
    hfSettings.exportSettings.modelPath = "/model1/path";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    hfSettings.graphSettings = std::move(textToSpeechGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedTextToSpeechGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, textToSpeechPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    ovms::TextToSpeechGraphSettingsImpl textToSpeechGraphSettings;
    hfSettings.graphSettings = std::move(textToSpeechGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedTextToSpeechGraphContentsDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, textToSpeechCreatedPbtxtInvalid) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    ovms::TextToSpeechGraphSettingsImpl textToSpeechGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel\"";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    hfSettings.graphSettings = std::move(textToSpeechGraphSettings);
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
#if (MEDIAPIPE_DISABLE == 0)
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
#else
    ASSERT_EQ(status, ovms::StatusCode::OK);
#endif
}

TEST_F(GraphCreationTest, speechToTextPositiveNonDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    ovms::SpeechToTextGraphSettingsImpl speechToTextGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel";
    hfSettings.exportSettings.modelPath = "/model1/path";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    hfSettings.graphSettings = std::move(speechToTextGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedSpeechToTextGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, speechToTextPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    ovms::SpeechToTextGraphSettingsImpl speechToTextGraphSettings;
    hfSettings.graphSettings = std::move(speechToTextGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedSpeechToTextGraphContentsDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, speechToTextCreatedPbtxtInvalid) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    ovms::SpeechToTextGraphSettingsImpl speechToTextGraphSettings;
    hfSettings.exportSettings.targetDevice = "GPU";
    hfSettings.exportSettings.modelName = "myModel\"";
    hfSettings.exportSettings.pluginConfig.numStreams = 2;
    hfSettings.graphSettings = std::move(speechToTextGraphSettings);
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
#if (MEDIAPIPE_DISABLE == 0)
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
#else
    ASSERT_EQ(status, ovms::StatusCode::OK);
#endif
}

TEST_F(GraphCreationTest, positivePluginConfigAll) {
    ovms::HFSettingsImpl hfSettings;
    ovms::TextGenGraphSettingsImpl graphSettings;
    hfSettings.exportSettings.pluginConfig.kvCachePrecision = "u8";
    hfSettings.exportSettings.pluginConfig.maxPromptLength = 123;
    hfSettings.exportSettings.pluginConfig.modelDistributionPolicy = "PIPELINE_PARALLEL";

    hfSettings.graphSettings = std::move(graphSettings);

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedFullPluginGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, positiveWithParsersAndToolGuidedGeneration) {
    ovms::HFSettingsImpl hfSettings;
    ovms::TextGenGraphSettingsImpl graphSettings;
    graphSettings.reasoningParser = "REASONING_PARSER";
    graphSettings.toolParser = "TOOL_PARSER";
    graphSettings.enableToolGuidedGeneration = "true";

    hfSettings.graphSettings = std::move(graphSettings);

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedGraphContentsWithResponseParser, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, positivePluginConfigOne) {
    ovms::HFSettingsImpl hfSettings;
    ovms::TextGenGraphSettingsImpl graphSettings;
    hfSettings.exportSettings.pluginConfig.kvCachePrecision = "u8";
    hfSettings.graphSettings = std::move(graphSettings);

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedOneSettingPluginGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, negativeCreateFileWrongDirectoryPaths) {
    ovms::HFSettingsImpl hfSettings;

    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig("", hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = graphExporter->createServableConfig("/does/not/exist", hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST_F(GraphCreationTest, negativeGraphOptionsNotInitialized) {
    ovms::HFSettingsImpl hfSettings;

    hfSettings.task = ovms::RERANK_GRAPH;
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    hfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    hfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    hfSettings.task = ovms::UNKNOWN_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);

    // Default constructable variant
    hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
    status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(GraphCreationTest, negativeCreatedPbtxtInvalid) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
    ovms::TextGenGraphSettingsImpl graphSettings;
    hfSettings.exportSettings.modelPath = "invalid\"";
    hfSettings.graphSettings = std::move(graphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::string subconfigPath = ovms::FileSystem::appendSlash(this->directoryPath) + "subconfig.json";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
#if (MEDIAPIPE_DISABLE == 0)
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
#else
    ASSERT_EQ(status, ovms::StatusCode::OK);
#endif
}
TEST_F(GraphCreationTest, positiveTextGeneration) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
    ovms::TextGenGraphSettingsImpl graphSettings;
    hfSettings.graphSettings = std::move(graphSettings);
    hfSettings.exportSettings.targetDevice = "NPU";
    hfSettings.exportSettings.pluginConfig.useNpuPrefixCaching = true;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::string subconfigPath = ovms::FileSystem::appendSlash(this->directoryPath) + "subconfig.json";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(GraphCreationTest, imageGenerationPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenerationGraphContentsDefault, removeVersionString(graphContents)) << graphContents;
}

TEST_F(GraphCreationTest, imageGenerationPositiveFull) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl imageGenerationGraphSettings;
    hfSettings.exportSettings.pluginConfig.numStreams = 14;
    hfSettings.exportSettings.pluginConfig.cacheDir = "/cache";
    hfSettings.exportSettings.targetDevice = "GPU";
    imageGenerationGraphSettings.defaultResolution = "300x400";
    imageGenerationGraphSettings.maxResolution = "3000x4000";
    imageGenerationGraphSettings.maxNumberImagesPerPrompt = 7;
    imageGenerationGraphSettings.defaultNumInferenceSteps = 2;
    imageGenerationGraphSettings.maxNumInferenceSteps = 3;
    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedImageGenerationGraphContents, removeVersionString(graphContents)) << graphContents;
}
TEST_F(GraphCreationTest, pluginConfigAsString) {
    ovms::ExportSettings exportSettings;

    auto& pluginConfig = exportSettings.pluginConfig;
    std::optional<std::string> stringPluginConfig;
    pluginConfig.kvCachePrecision = "u8";
    pluginConfig.maxPromptLength = 256;
    pluginConfig.modelDistributionPolicy = "TENSOR_PARALLEL";
    // note here we use int not string num_streams so this shows way to overwrite type
    pluginConfig.manualString = "{\"NUM_STREAMS\":4}";
    auto res = ovms::GraphExport::createPluginString(exportSettings);
    ASSERT_TRUE(std::holds_alternative<std::optional<std::string>>(res));
    ASSERT_EQ(std::get<std::optional<std::string>>(res).value(),
        "{\"NUM_STREAMS\":4,\"KV_CACHE_PRECISION\":\"u8\",\"MAX_PROMPT_LEN\":256,\"MODEL_DISTRIBUTION_POLICY\":\"TENSOR_PARALLEL\"}");
}
TEST_F(GraphCreationTest, pluginConfigNegative) {
    using ovms::Status;
    ovms::PluginConfigSettingsImpl pluginConfig;
    ovms::ExportSettings exportSettings;
    std::optional<std::string> stringPluginConfig;
    pluginConfig.kvCachePrecision = "u8";
    pluginConfig.maxPromptLength = 256;
    pluginConfig.modelDistributionPolicy = "TENSOR_PARALLEL";
    pluginConfig.cacheDir = "/cache";

    exportSettings.pluginConfig = pluginConfig;
    exportSettings.pluginConfig.manualString = "{\"KV_CACHE_PRECISION\":\"fp16\"}";
    auto res = ovms::GraphExport::createPluginString(exportSettings);
    ASSERT_TRUE(std::holds_alternative<ovms::Status>(res));
    ASSERT_EQ(std::get<Status>(res), ovms::StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS);

    exportSettings.pluginConfig.manualString = "{\"MAX_PROMPT_LEN\":512}";
    res = ovms::GraphExport::createPluginString(exportSettings);
    ASSERT_TRUE(std::holds_alternative<ovms::Status>(res));
    ASSERT_EQ(std::get<Status>(res), ovms::StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS);

    exportSettings.pluginConfig.manualString = "{\"CACHE_DIR\":\"/cache\"}";
    res = ovms::GraphExport::createPluginString(exportSettings);
    ASSERT_TRUE(std::holds_alternative<ovms::Status>(res));
    ASSERT_EQ(std::get<Status>(res), ovms::StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS);
    exportSettings.pluginConfig.manualString = "{\"MODEL_DISTRIBUTION_POLICY\":\"PIPELINE_PARALLEL\"}";
    res = ovms::GraphExport::createPluginString(exportSettings);
    ASSERT_TRUE(std::holds_alternative<ovms::Status>(res));
    ASSERT_EQ(std::get<Status>(res), ovms::StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS);
}
