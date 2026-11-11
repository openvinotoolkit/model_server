//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "src/graph_metadata.hpp"
#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/stringutils.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"
#include "mediapipe/framework/calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class LlmNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "LLMCalculator";

public:
    bool matches(const std::string& calculatorName) const override {
        return endsWith(calculatorName, CALCULATOR_NAME);
    }
    Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        GraphMetadata& metadata,
        PythonBackend* /*pythonBackend*/) override {
        auto& genAiServableMap = sidePackets.genAiServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("LLM node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("LLM node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (genAiServableMap.find(nodeName) != genAiServableMap.end()) {
            SPDLOG_ERROR("LLM node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        std::shared_ptr<GenAiServable> servable;
        Status status = initializeGenAiServable(servable, nodeConfig, basePath);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to process LLM node graph {}", graphName);
            return status;
        }
        const auto maxModelLength = servable->getProperties()->maxModelLength;
        if (maxModelLength.has_value() && !metadata.setInt("max_model_len", maxModelLength.value())) {
            SPDLOG_WARN("Metadata key max_model_len is already set in graph: {}, ignoring the value from LLM node: {}", graphName, nodeName);
        }
        genAiServableMap.insert(std::pair<std::string, std::shared_ptr<GenAiServable>>(nodeName, std::move(servable)));
        sidePackets.genAiExecutionContextMap.emplace(
            nodeName, std::make_shared<GenAiExecutionContextHolder>());
        return StatusCode::OK;
    }
};

static bool llmNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<LlmNodeInitializer>());
    return true;
}();
}  // namespace ovms
