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

#include <string>
#include <utility>

#include "../config.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"

namespace ovms {

static std::string createTextGenerationGraphTemplate(const std::string& pipelineType, const std::string& modelPath, const std::string& maxNumSeqs, const std::string& targetDevice,
    const std::string& pluginConfig, const std::string& enablePrefixCaching, const std::string& cacheSize, const std::string& maxNumBatchedTokens, bool dynamicSplitFuse,
    const std::string& draftModelDirName) {
    std::ostringstream oss;
    oss << R"(input_stream: "HTTP_REQUEST_PAYLOAD:input"
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
            models_path: )"
        << modelPath << R"(,
            plugin_config: )"
        << pluginConfig << R"(,
            enable_prefix_caching: )"
        << enablePrefixCaching << R"(,
            cache_size: )"
        << cacheSize << R"(,)";
    if (pipelineType != "") {
        oss << R"(pipeline_type: )" << pipelineType << R"(,)";
    }
    if (maxNumBatchedTokens != "") {
        oss << R"(max_num_batched_tokens: )" << maxNumBatchedTokens << R"(,)";
    }
    if (!dynamicSplitFuse) {
        oss << R"(dynamic_split_fuse: false)";
    }
    if (draftModelDirName != "") {
        oss << R"(# Speculative decoding configuration)";
        oss << R"(draft_models_path: )" << draftModelDirName << R"(,)";
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

    return oss.str();
}

GraphExport::GraphExport(const std::string& pipelineType, const std::string& modelPath, const std::string& maxNumSeqs, const std::string& targetDevice,
    const std::string& pluginConfig, const std::string& enablePrefixCaching, const std::string& cacheSize, const std::string& maxNumBatchedTokens, bool dynamicSplitFuse,
    const std::string& draftModelDirName) {
    this->graphString = createTextGenerationGraphTemplate(pipelineType, modelPath, maxNumSeqs, targetDevice,
        pluginConfig, enablePrefixCaching, cacheSize, maxNumBatchedTokens, dynamicSplitFuse,
        draftModelDirName);
}

Status GraphExport::createGraphFile(const std::string directoryPath) {
}

}  // namespace ovms
