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
#include <variant>

#include <openvino/core/except.hpp>

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/stringutils.hpp"
#include "imagegen_init.hpp"
#include "pipelines.hpp"
#include "mediapipe/framework/calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class ImageGenNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "ImageGenCalculator";

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
        auto& imageGenPipelinesMap = sidePackets.imageGenPipelinesMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("Image Gen node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("Image Gen node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (imageGenPipelinesMap.find(nodeName) != imageGenPipelinesMap.end()) {
            SPDLOG_ERROR("Image Gen node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        auto statusOrArgs = prepareImageGenPipelineArgs(nodeConfig.node_options(0), basePath);
        if (std::holds_alternative<Status>(statusOrArgs)) {
            SPDLOG_ERROR("Failed to prepare Image Gen pipeline args for node: {}. Error: {}", graphName, std::get<Status>(statusOrArgs).string());
            return std::get<Status>(statusOrArgs);
        }
        std::shared_ptr<ImageGenerationPipelines> servable;
        try {
            servable = std::make_shared<ImageGenerationPipelines>(std::get<ImageGenPipelineArgs>(statusOrArgs));
        } catch (ov::Exception& e) {
            SPDLOG_ERROR("Failed to create Image Generation pipelines: {}. Error: {}", graphName, e.what());
            return StatusCode::INTERNAL_ERROR;
        } catch (...) {
            SPDLOG_ERROR("Failed to create Image Generation pipelines: {}. Unknown error", graphName);
            return StatusCode::INTERNAL_ERROR;
        }
        imageGenPipelinesMap.insert(std::pair<std::string, std::shared_ptr<ImageGenerationPipelines>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool imageGenNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<ImageGenNodeInitializer>());
    return true;
}();
}  // namespace ovms
