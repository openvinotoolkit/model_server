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

static std::string createTextGenerationGraphTemplate(const std::string& pipelineType, const std::string& modelPath, const std::string& maxNumSeqs, const std::string& targetDevice,
    const std::string& pluginConfig, const std::string& enablePrefixCaching, const std::string& cacheSize, const std::string& maxNumBatchedTokens, const std::string& dynamicSplitFuse,
    const std::string& draftModelDirName) {
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
        << maxNumSeqs << R"(,
            device: )"
        << targetDevice << R"(,
            models_path: ")"
        << modelPath << R"(",
            plugin_config: ')"
        << pluginConfig << R"(',
            enable_prefix_caching: )"
        << enablePrefixCaching << R"(,
            cache_size: )"
        << cacheSize << R"(,)";
    if (pipelineType != "") {
        oss << R"(
            pipeline_type: )" << pipelineType << R"(,)";
    }
    if (maxNumBatchedTokens != "") {
        oss << R"(
            max_num_batched_tokens: )" << maxNumBatchedTokens << R"(,)";
    }
    if (dynamicSplitFuse != "true") {
        oss << R"(
            dynamic_split_fuse: false,)";
    }
    if (draftModelDirName != "") {
        oss << R"(
            # Speculative decoding configuration)";
        oss << R"(
            draft_models_path: )" << draftModelDirName << R"(,)";
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

GraphExport::GraphExport(const GraphSettingsImpl& graphSettings) {
    this->graphString = createTextGenerationGraphTemplate(graphSettings.pipelineType, graphSettings.modelPath, graphSettings.maxNumSeqs, graphSettings.targetDevice,
        GraphExport::createPluginString(graphSettings.pluginConfig), graphSettings.enablePrefixCaching, graphSettings.cacheSize, graphSettings.maxNumBatchedTokens, graphSettings.dynamicSplitFuse,
        graphSettings.draftModelDirName);
}

Status GraphExport::createGraphFile(const std::string& directoryPath) {
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    // Always overwrite
    {
        std::ofstream graphFile(fullPath, std::ios::trunc | std::ofstream::binary);
        if (graphFile.is_open()) {
            graphFile << this->graphString << std::endl;
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

    if (pluginConfig.kvCachePrecision != "") {
        rapidjson::Value name;
        name.SetString(pluginConfig.kvCachePrecision.c_str(), d.GetAllocator());
        d.AddMember("KV_CACHE_PRECISION", name, d.GetAllocator());
        configNotEmpty = true;
    }

    if (pluginConfig.maxPromptLength != "") {
        rapidjson::Value name;
        name.SetString(pluginConfig.maxPromptLength.c_str(), d.GetAllocator());
        d.AddMember("MAX_PROMPT_LEN", name, d.GetAllocator());
        configNotEmpty = true;
    }

    if (pluginConfig.modelDistributionPolicy != "") {
        rapidjson::Value name;
        name.SetString(pluginConfig.modelDistributionPolicy.c_str(), d.GetAllocator());
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
