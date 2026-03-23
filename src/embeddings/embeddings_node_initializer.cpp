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
#include <utility>

#include "../mediapipe_internal/graph_side_packets.hpp"
#include "../mediapipe_internal/node_initializer.hpp"
#include "../stringutils.hpp"
#include "embeddings_servable.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/embeddings/embeddings_calculator_ov.pb.h"

#include "../logging.hpp"

namespace ovms {
class EmbeddingsNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "EmbeddingsCalculatorOV";

public:
    bool matches(const std::string& calculatorName) const override {
        return endsWith(calculatorName, CALCULATOR_NAME);
    }
    Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        PythonBackend* /*pythonBackend*/) override {
        auto& embeddingsServableMap = sidePackets.embeddingsServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("Embeddings node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("Embeddings node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (embeddingsServableMap.find(nodeName) != embeddingsServableMap.end()) {
            SPDLOG_ERROR("Embeddings node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        mediapipe::EmbeddingsCalculatorOVOptions nodeOptions;
        nodeConfig.node_options(0).UnpackTo(&nodeOptions);
        auto servable = std::make_shared<EmbeddingsServable>(
            nodeOptions.models_path(),
            nodeOptions.target_device(),
            nodeOptions.plugin_config(),
            basePath,
            nodeOptions.pooling(),
            nodeOptions.normalize_embeddings());
        servable->initialize(
            nodeOptions.models_path(),
            nodeOptions.target_device(),
            nodeOptions.plugin_config(),
            basePath);
        embeddingsServableMap.insert(std::pair<std::string, std::shared_ptr<EmbeddingsServable>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool embeddingsNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<EmbeddingsNodeInitializer>());
    return true;
}();
}  // namespace ovms
