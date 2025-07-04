//***************************************************************************
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
#include "graph_export.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "../config.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../schema.hpp"
#include "graph_export_types.hpp"

#if (MEDIAPIPE_DISABLE == 0)
#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#endif
namespace ovms {

static Status createTextGenerationGraphTemplate(const std::string& directoryPath, const TextGenGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
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
            max_num_seqs:)"
        << graphSettings.maxNumSeqs << R"(,
            device: ")"
        << graphSettings.targetDevice << R"(",
            models_path: ")"
        << graphSettings.modelPath << R"(",
            plugin_config: ')"
        << GraphExport::createPluginString(graphSettings.pluginConfig) << R"(',
            enable_prefix_caching: )"
        << graphSettings.enablePrefixCaching << R"(,
            cache_size: )"
        << graphSettings.cacheSize << R"(,)";
    if (graphSettings.pipelineType.has_value()) {
        oss << R"(
            pipeline_type: )" << graphSettings.pipelineType.value() << R"(,)";
    }
    if (graphSettings.maxNumBatchedTokens.has_value()) {
        oss << R"(
            max_num_batched_tokens: )" << graphSettings.maxNumBatchedTokens.value() << R"(,)";
    }
    if (graphSettings.responseParser.has_value()) {
        oss << R"(
            response_parser: ")" << graphSettings.responseParser.value() << R"(",)";
    }
    if (graphSettings.dynamicSplitFuse != "true") {
        oss << R"(
            dynamic_split_fuse: false,)";
    }
    if (graphSettings.draftModelDirName.has_value()) {
        oss << R"(
            # Speculative decoding configuration)";
        oss << R"(
            draft_models_path: )" << graphSettings.draftModelDirName.value() << R"(,)";
    }
    oss << R"(
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
#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created graph config file couldn't be parsed - check used task parameters values.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createRerankGraphTemplate(const std::string& directoryPath, const RerankGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // Windows path creation - graph parser needs forward slashes in paths
    std::string graphOkPath = graphSettings.modelPath;
    if (FileSystem::getOsSeparator() != "/") {
        std::replace(graphOkPath.begin(), graphOkPath.end(), '\\', '/');
    }
    // clang-format off
    oss << R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << graphSettings.modelName << R"(",
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {
            models_path: ")"
            << graphOkPath << R"(",
            max_allowed_chunks: )"
            << graphSettings.maxAllowedChunks << R"(,
            target_device: ")" << graphSettings.targetDevice << R"(",
            plugin_config: '{ "NUM_STREAMS": ")" << graphSettings.numStreams << R"("}',
        }
    }
})";

#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created rerank graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createEmbeddingsGraphTemplate(const std::string& directoryPath, const EmbeddingsGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // Windows path creation - graph parser needs forward slashes in paths
    std::string graphOkPath = graphSettings.modelPath;
    if (FileSystem::getOsSeparator() != "/") {
        std::replace(graphOkPath.begin(), graphOkPath.end(), '\\', '/');
    }

    // clang-format off
    oss << R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << graphSettings.modelName << R"(",
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
            models_path: ")"
            << graphOkPath << R"(",
            normalize_embeddings: )"
            << graphSettings.normalize << R"(,
            mean_pooling: )"
            << graphSettings.meanPooling << R"(,
            target_device: ")" << graphSettings.targetDevice << R"(",
            plugin_config: '{ "NUM_STREAMS": ")" << graphSettings.numStreams << R"("}',
        }
    }
})";

#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created embeddings graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createImageGenerationGraphTemplate(const std::string& directoryPath, const ImageGenerationGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
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
          models_path: ")" << graphSettings.modelPath << R"("
          device: ")" << graphSettings.targetDevice << R"(")";

    if (graphSettings.pluginConfig.size()) {
        oss << R"(
          plugin_config: ')" << graphSettings.pluginConfig << R"(')";
    }

    if (graphSettings.maxResolution.size()) {
        oss << R"(
          max_resolution: ")" << graphSettings.maxResolution << R"(")";
    }

    if (graphSettings.defaultResolution.size()) {
        oss << R"(
          default_resolution: ")" << graphSettings.defaultResolution << R"(")";
    }

    if (graphSettings.maxNumberImagesPerPrompt.has_value()) {
        oss << R"(
          max_num_images_per_prompt: )" << graphSettings.maxNumberImagesPerPrompt.value();
    }

    if (graphSettings.defaultNumInferenceSteps.has_value()) {
        oss << R"(
          default_num_inference_steps: )" << graphSettings.defaultNumInferenceSteps.value();
    }

    if (graphSettings.maxNumInferenceSteps.has_value()) {
        oss << R"(
          max_num_inference_steps: )" << graphSettings.maxNumInferenceSteps.value();
    }

    oss << R"(
      }
  }
}
)";

    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

GraphExport::GraphExport() {
}

Status GraphExport::createServableConfig(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (directoryPath.empty()) {
        SPDLOG_ERROR("Directory path empty: {}", directoryPath);
        return StatusCode::PATH_INVALID;
    }

    bool exists = false;
    auto status = LocalFileSystem::exists(directoryPath, &exists);
    if (!status.ok())
        return status;

    bool is_dir = false;
    status = LocalFileSystem::isDir(directoryPath, &is_dir);
    if (!status.ok())
        return status;

    if (!is_dir) {
        SPDLOG_ERROR("Graph path is not a directory: {}", directoryPath);
        return StatusCode::PATH_INVALID;
    }

    if (hfSettings.task == TEXT_GENERATION_GRAPH) {
        if (std::holds_alternative<TextGenGraphSettingsImpl>(hfSettings.graphSettings)) {
            return createTextGenerationGraphTemplate(directoryPath, std::get<TextGenGraphSettingsImpl>(hfSettings.graphSettings));
        } else {
            SPDLOG_ERROR("Graph options not initialized for text generation.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else if (hfSettings.task == EMBEDDINGS_GRAPH) {
        if (std::holds_alternative<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings)) {
            return createEmbeddingsGraphTemplate(directoryPath, std::get<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings));
        } else {
            SPDLOG_ERROR("Graph options not initialized for embeddings.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else if (hfSettings.task == RERANK_GRAPH) {
        if (std::holds_alternative<RerankGraphSettingsImpl>(hfSettings.graphSettings)) {
            return createRerankGraphTemplate(directoryPath, std::get<RerankGraphSettingsImpl>(hfSettings.graphSettings));
        } else {
            SPDLOG_ERROR("Graph options not initialized for rerank.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else if (hfSettings.task == IMAGE_GENERATION_GRAPH) {
        if (std::holds_alternative<ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings)) {
            return createImageGenerationGraphTemplate(directoryPath, std::get<ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings));
        } else {
            SPDLOG_ERROR("Graph options not initialized for image generation.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else if (hfSettings.task == UNKNOWN_GRAPH) {
        SPDLOG_ERROR("Graph options not initialized.");
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::INTERNAL_ERROR;
}

std::string GraphExport::createPluginString(const PluginConfigSettingsImpl& pluginConfig) {
    rapidjson::Document d;
    d.SetObject();
    bool configNotEmpty = false;

    if (pluginConfig.kvCachePrecision.has_value()) {
        rapidjson::Value name;
        name.SetString(pluginConfig.kvCachePrecision.value().c_str(), d.GetAllocator());
        d.AddMember("KV_CACHE_PRECISION", name, d.GetAllocator());
        configNotEmpty = true;
    }

    if (pluginConfig.maxPromptLength.has_value()) {
        rapidjson::Value name;
        name.SetString(std::to_string(pluginConfig.maxPromptLength.value()).c_str(), d.GetAllocator());
        d.AddMember("MAX_PROMPT_LEN", name, d.GetAllocator());
        configNotEmpty = true;
    }

    if (pluginConfig.modelDistributionPolicy.has_value()) {
        rapidjson::Value name;
        name.SetString(pluginConfig.modelDistributionPolicy.value().c_str(), d.GetAllocator());
        d.AddMember("MODEL_DISTRIBUTION_POLICY", name, d.GetAllocator());
        configNotEmpty = true;
    }

    std::string pluginString = "{ }";

    if (configNotEmpty) {
        // Serialize the document to a JSON string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);

        // Output the JSON string
        pluginString = buffer.GetString();
    }

    return pluginString;
}

}  // namespace ovms
