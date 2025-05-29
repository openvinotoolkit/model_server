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

#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_graph.h"

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
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created graph config file couldn't be parsed - check used task parameters values.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status validateSubconfigSchema(const std::string& subconfig, const std::string& type) {
    rapidjson::Document subconfigJson;
    rapidjson::ParseResult parseResult = subconfigJson.Parse(subconfig.c_str());
    if (parseResult.Code()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Created {} subconfig file is not a valid JSON file. Error: {}", type, rapidjson::GetParseError_En(parseResult.Code()));
        return StatusCode::JSON_INVALID;
    }
    if (validateJsonAgainstSchema(subconfigJson, MEDIAPIPE_SUBCONFIG_SCHEMA.c_str()) != StatusCode::OK) {
        SPDLOG_ERROR("Created {} subconfig file is not in valid configuration format", type);
        return StatusCode::JSON_INVALID;
    }
    return StatusCode::OK;
}

static Status createRerankSubconfigTemplate(const std::string& directoryPath, const RerankGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_tokenizer_model",
                    "base_path": "tokenizer"
                }
            },
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_rerank_model",
                    "base_path": "rerank",
                    "target_device": ")" << graphSettings.targetDevice << R"(",
                    "plugin_config": { "NUM_STREAMS": ")" << graphSettings.numStreams << R"(" }
                }
            }
        ]
    })";
    auto status = validateSubconfigSchema(oss.str(), "rerank");
    if (!status.ok()){
        return status;
    }
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "subconfig.json"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createEmbeddingsSubconfigTemplate(const std::string& directoryPath, const EmbeddingsGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_tokenizer_model",
                    "base_path": "tokenizer"
                }
            },
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_embeddings_model",
                    "base_path": "embeddings",
                    "target_device": ")" << graphSettings.targetDevice << R"(",
                    "plugin_config": { "NUM_STREAMS": ")" << graphSettings.numStreams << R"(" }
                }
            }
        ]
    })";
    auto status = validateSubconfigSchema(oss.str(), "embeddings");
    if (!status.ok()){
        return status;
    }
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "subconfig.json"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createRerankGraphTemplate(const std::string& directoryPath, const RerankGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:tokenizer"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: ")"
        << graphSettings.modelName << R"(_tokenizer_model"
        }
    }
    }
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:rerank"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: ")"
        << graphSettings.modelName << R"(_rerank_model"
        }
    }
    }
    node {
        input_side_packet: "TOKENIZER_SESSION:tokenizer"
        input_side_packet: "RERANK_SESSION:rerank"
        calculator: "RerankCalculator"
        input_stream: "REQUEST_PAYLOAD:input"
        output_stream: "RESPONSE_PAYLOAD:output"
    })";

    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created rerank graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    auto status = FileSystem::createFileOverwrite(fullPath, oss.str());
    if (!status.ok())
        return status;

    return createRerankSubconfigTemplate(directoryPath, graphSettings);
}

static Status createEmbeddingsGraphTemplate(const std::string& directoryPath, const EmbeddingsGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:tokenizer"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: ")"
        << graphSettings.modelName << R"(_tokenizer_model"
        }
    }
    }
    node {
    calculator: "OpenVINOModelServerSessionCalculator"
    output_side_packet: "SESSION:embeddings"
    node_options: {
        [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
        servable_name: ")"
        << graphSettings.modelName << R"(_embeddings_model"
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
            normalize_embeddings: )"
            << graphSettings.normalize << R"(,
        }
    }
    })";

    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created embeddings graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    auto status = FileSystem::createFileOverwrite(fullPath, oss.str());
    if (!status.ok())
        return status;

    return createEmbeddingsSubconfigTemplate(directoryPath, graphSettings);
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
    } else if (hfSettings.task == UNKNOWN_GRAPH) {
        SPDLOG_ERROR("Graph options not initialized.");
        return StatusCode::INTERNAL_ERROR;
    }
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
