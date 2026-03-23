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

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "pythonnoderesources.hpp"
#include "mediapipe/framework/calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class PythonNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "PythonExecutorCalculator";

public:
    bool matches(const std::string& calculatorName) const override {
        return calculatorName == CALCULATOR_NAME;
    }
    Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        PythonBackend* pythonBackend) override {
        auto& pythonNodeResourcesMap = sidePackets.pythonNodeResourcesMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("Python node missing options in graph: {}. ", graphName);
            return StatusCode::PYTHON_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("Python node name is missing in graph: {}. ", graphName);
            return StatusCode::PYTHON_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (pythonNodeResourcesMap.find(nodeName) != pythonNodeResourcesMap.end()) {
            SPDLOG_ERROR("Python node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::PYTHON_NODE_NAME_ALREADY_EXISTS;
        }
        std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
        Status status = PythonNodeResources::createPythonNodeResources(nodeResources, nodeConfig, pythonBackend, basePath);
        if (nodeResources == nullptr || !status.ok()) {
            SPDLOG_ERROR("Failed to process python node graph {}", graphName);
            return status;
        }
        pythonNodeResourcesMap.insert(std::pair<std::string, std::shared_ptr<PythonNodeResources>>(nodeName, std::move(nodeResources)));
        return StatusCode::OK;
    }
};

static bool pythonNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<PythonNodeInitializer>());
    return true;
}();
}  // namespace ovms
