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
            device: CPU,
            models_path: "./",
            plugin_config: '{"KV_CACHE_PRECISION":"u8"}',
            enable_prefix_caching: true,
            cache_size: 10,
            pipeline_type: AUTO,
            dynamic_split_fuse: false,
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
            device: CPU,
            models_path: "./",
            plugin_config: '{"KV_CACHE_PRECISION":"u8","MAX_PROMPT_LEN":"123","MODEL_DISTRIBUTION_POLICY":"PIPELINE_PARALLEL"}',
            enable_prefix_caching: true,
            cache_size: 10,
            pipeline_type: AUTO,
            dynamic_split_fuse: false,
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
            device: CPU,
            models_path: "./",
            plugin_config: '{ }',
            enable_prefix_caching: true,
            cache_size: 10,
            pipeline_type: AUTO,
            dynamic_split_fuse: false,
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

class GraphCreationTest : public TestWithTempDir {
protected:
    void TearDown() {
        TestWithTempDir::TearDown();
    }
};

TEST_F(GraphCreationTest, positiveDefault) {
    ovms::GraphSettingsImpl graphSettings;
    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createGraphFile(this->directoryPath, graphSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::cout << graphContents << std::endl;
    ASSERT_EQ(expectedDefaultGraphContents, graphContents);
}

TEST_F(GraphCreationTest, positivePluginConfigAll) {
    ovms::GraphSettingsImpl graphSettings;
    graphSettings.pluginConfig.kvCachePrecision = "u8";
    graphSettings.pluginConfig.maxPromptLength = 123;
    graphSettings.pluginConfig.modelDistributionPolicy = "PIPELINE_PARALLEL";

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createGraphFile(this->directoryPath, graphSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::cout << graphContents << std::endl;
    ASSERT_EQ(expectedFullPluginGraphContents, graphContents);
}

TEST_F(GraphCreationTest, positivePluginConfigOne) {
    ovms::GraphSettingsImpl graphSettings;
    graphSettings.pluginConfig.kvCachePrecision = "u8";

    std::string graphPath = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::unique_ptr<ovms::GraphExport> graphExporter = std::make_unique<ovms::GraphExport>();
    auto status = graphExporter->createGraphFile(this->directoryPath, graphSettings);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string graphContents = GetFileContents(graphPath);
    std::cout << graphContents << std::endl;
    ASSERT_EQ(expectedOneSettingPluginGraphContents, graphContents);
}
