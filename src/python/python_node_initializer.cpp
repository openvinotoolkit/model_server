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
#include "src/mediapipe_internal/mediapipe_utils.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "mediapipe/framework/calculator.pb.h"

#include "src/logging.hpp"
#include "src/python/python_executor_calculator.pb.h"
#include "python_calculators_plugin_api.hpp"
#include "python_calculators_plugin_loader.hpp"
#include "python_node_config.hpp"

namespace ovms {

static void createOutputTagNameMapping(PythonNodeConfig& config, const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig) {
    for (const auto& name : nodeConfig.output_stream()) {
        std::string delimiter = ":";
        std::string streamTag, streamName;
        size_t tagDelimiterPos = name.find(delimiter, 0);

        if (tagDelimiterPos == std::string::npos) {
            // Empty tag - example: output_stream: "output"
            streamTag = "";
            streamName = name;
        } else {
            streamTag = name.substr(0, tagDelimiterPos);
            size_t indexDelimiterPos = name.find(delimiter, tagDelimiterPos + 1);
            if (indexDelimiterPos == std::string::npos) {
                // Only tag, no index - example: output_stream: "OUTPUT:output"
                streamName = name.substr(tagDelimiterPos + 1, std::string::npos);
            } else {
                // Both tag and index - example: output_stream: "OUTPUT:0:output"
                // It's permitted by MediaPipe, but PythonExecutorCalculator ignores it.
                streamName = name.substr(indexDelimiterPos + 1, std::string::npos);
            }
        }
        // PythonExecutorCalculator ignores index value, so only Tag gets mapped
        config.outputsNameTagMapping.insert({streamName, streamTag});
    }
}

PythonNodeConfig toPythonNodeConfig(const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig, const std::string& graphPath) {
    mediapipe::PythonExecutorCalculatorOptions nodeOptions;
    nodeConfig.node_options(0).UnpackTo(&nodeOptions);

    PythonNodeConfig config;
    config.nodeName = nodeConfig.name();
    config.handlerPath = nodeOptions.handler_path();
    config.graphPath = graphPath;
    for (const auto& name : nodeConfig.input_stream()) {
        config.inputNames.push_back(getStreamName(name));
    }
    for (const auto& name : nodeConfig.output_stream()) {
        config.outputNames.push_back(getStreamName(name));
    }
    createOutputTagNameMapping(config, nodeConfig);
    return config;
}

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
        const auto* api = getPythonCalculatorsPluginApi();
        if (api == nullptr) {
            SPDLOG_ERROR("Python calculators plugin is not loaded. Cannot initialize python node: {} in graph: {}", nodeName, graphName);
            return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
        }
        std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
        Status status = static_cast<StatusCode>(api->createNodeResources(toPythonNodeConfig(nodeConfig, basePath), pythonBackend, nodeResources));
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
