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

#include "test_utils.hpp"
#include "../filesystem.hpp"
#include "../status.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../graph_export/graph_export.hpp"

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
            plugin_config: '{"KV_CACHE_PRECISION":"u8","MAX_PROMPT_LEN":"123","MODEL_DISTRIBUTION_POLICY":"PIPELINE_PARALLEL"}',
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
            plugin_config: '{ }',
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

const std::string expectedRerankJsonContents = R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": "myModel_tokenizer_model",
                    "base_path": "tokenizer"
                }
            },
            { "config":
                {
                    "name": "myModel_rerank_model",
                    "base_path": "rerank",
                    "target_device": "GPU",
                    "plugin_config": { "NUM_STREAMS": "2" }
                }
            }
        ]
    }
)";

const std::string expectedEmbeddingsJsonContents = R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": "myModel_tokenizer_model",
                    "base_path": "tokenizer"
                }
            },
            { "config":
                {
                    "name": "myModel_embeddings_model",
                    "base_path": "embeddings",
                    "target_device": "GPU",
                    "plugin_config": { "NUM_STREAMS": "2" }
                }
            }
        ]
    }
)";

const std::string expectedRerankGraphContents = R"(
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:tokenizer"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: "myModel_tokenizer_model"
        }
    }
    }
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:rerank"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: "myModel_rerank_model"
        }
    }
    }
    node {
        input_side_packet: "TOKENIZER_SESSION:tokenizer"
        input_side_packet: "RERANK_SESSION:rerank"
        calculator: "RerankCalculator"
        input_stream: "REQUEST_PAYLOAD:input"
        output_stream: "RESPONSE_PAYLOAD:output"
    }
)";

const std::string expectedEmbeddingsGraphContents = R"(
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:tokenizer"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: "myModel_tokenizer_model"
        }
    }
    }
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:embeddings"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: "myModel_embeddings_model"
        }
    }
    }
    node {
        input_side_packet: "TOKENIZER_SESSION:tokenizer"
        input_side_packet: "EMBEDDINGS_SESSION:embeddings"
        calculator: "EmbeddingsCalculator"
        input_stream: "REQUEST_PAYLOAD:input"
        output_stream: "RESPONSE_PAYLOAD:output"
        node_options: {
            [type.googleapis.com / mediapipe.EmbeddingsCalculatorOptions]: {
            normalize_embeddings: true,
        }
    }
    }
)";

class GraphCreationTest : public TestWithTempDir {
protected:
    void TearDown() {
        TestWithTempDir::TearDown();
    }
};

TEST_F(GraphCreationTest, positiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedDefaultGraphContents, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, rerankPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::rerank;
    hfSettings.rerankGraphSettings.targetDevice = "GPU";
    hfSettings.rerankGraphSettings.modelName = "myModel";
    hfSettings.rerankGraphSettings.numStreams = 2;
    hfSettings.rerankGraphSettings.maxDocLength = 18;
    hfSettings.rerankGraphSettings.version = 2;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::string subconfigPath = ovms::FileSystem::appendSlash(this->directoryPath) + "subconfig.json";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedRerankGraphContents, graphContents) << graphContents;

    std::string jsonContents = GetFileContents(subconfigPath);
    ASSERT_EQ(expectedRerankJsonContents, jsonContents) << jsonContents;
}

TEST_F(GraphCreationTest, embeddingsPositiveDefault) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.task = ovms::embeddings;
    hfSettings.embeddingsGraphSettings.targetDevice = "GPU";
    hfSettings.embeddingsGraphSettings.modelName = "myModel";
    hfSettings.embeddingsGraphSettings.numStreams = 2;
    hfSettings.embeddingsGraphSettings.truncate = "true";
    hfSettings.embeddingsGraphSettings.normalize = "true";
    hfSettings.embeddingsGraphSettings.version = 2;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::string subconfigPath = ovms::FileSystem::appendSlash(this->directoryPath) + "subconfig.json";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedEmbeddingsGraphContents, graphContents) << graphContents;

    std::string jsonContents = GetFileContents(subconfigPath);
    ASSERT_EQ(expectedEmbeddingsJsonContents, jsonContents) << jsonContents;
}

TEST_F(GraphCreationTest, positivePluginConfigAll) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.graphSettings.pluginConfig.kvCachePrecision = "u8";
    hfSettings.graphSettings.pluginConfig.maxPromptLength = 123;
    hfSettings.graphSettings.pluginConfig.modelDistributionPolicy = "PIPELINE_PARALLEL";

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedFullPluginGraphContents, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, positivePluginConfigOne) {
    ovms::HFSettingsImpl hfSettings;
    hfSettings.graphSettings.pluginConfig.kvCachePrecision = "u8";

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig(this->directoryPath, hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    ASSERT_EQ(expectedOneSettingPluginGraphContents, graphContents) << graphContents;
}

TEST_F(GraphCreationTest, negativeCreateFileWrongDirectoryPaths) {
    ovms::HFSettingsImpl hfSettings;

    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createServableConfig("", hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = graphExporter->createServableConfig("/does/not/exist", hfSettings);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}
