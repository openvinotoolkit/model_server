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
#include "src/stringutils.hpp"
#include "s2t_servable.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/audio/speech_to_text/s2t_calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class SttNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "S2tCalculator";

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
        auto& sttServableMap = sidePackets.sttServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("SpeechToText node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("SpeechToText node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (sttServableMap.find(nodeName) != sttServableMap.end()) {
            SPDLOG_ERROR("SpeechToText node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        mediapipe::S2tCalculatorOptions nodeOptions;
        const auto& calculatorOptions = nodeConfig.node_options(0);
        if (!calculatorOptions.UnpackTo(&nodeOptions)) {
            SPDLOG_ERROR("Failed to unpack calculator options");
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
        auto servable = std::make_shared<SttServable>(nodeOptions, basePath);
        sttServableMap.insert(std::pair<std::string, std::shared_ptr<SttServable>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool sttNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<SttNodeInitializer>());
    return true;
}();
}  // namespace ovms
