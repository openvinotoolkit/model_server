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
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "../config.hpp"
#include "../filesystem.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"

namespace ovms {

static std::string createTextGenerationGraphTemplate(const GraphSettingsImpl& graphSettings) {
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
            device: )"
        << graphSettings.targetDevice << R"(,
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

    // clang-format on
    return oss.str();
}

GraphExport::GraphExport() {
}

Status GraphExport::createGraphFile(const std::string& directoryPath, const GraphSettingsImpl& graphSettings) {
    std::string graphString = createTextGenerationGraphTemplate(graphSettings);
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    // Always overwrite
    {
        std::ofstream graphFile(fullPath, std::ios::trunc | std::ofstream::binary);
        if (graphFile.is_open()) {
            graphFile << graphString << std::endl;
        } else {
            SPDLOG_ERROR("Unable to open file: ", fullPath);
            return StatusCode::FILE_INVALID;
        }
    }
    return StatusCode::OK;
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
