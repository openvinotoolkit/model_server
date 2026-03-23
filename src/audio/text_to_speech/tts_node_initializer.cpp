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
#include <stdexcept>
#include <string>
#include <utility>

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/stringutils.hpp"
#include "t2s_servable.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class TtsNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "T2sCalculator";

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
        auto& ttsServableMap = sidePackets.ttsServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("TextToSpeech node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("TextToSpeech node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (ttsServableMap.find(nodeName) != ttsServableMap.end()) {
            SPDLOG_ERROR("TextToSpeech node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        mediapipe::T2sCalculatorOptions nodeOptions;
        const auto& calculatorOptions = nodeConfig.node_options(0);
        if (!calculatorOptions.UnpackTo(&nodeOptions)) {
            SPDLOG_ERROR("Failed to unpack calculator options");
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
        try {
            auto servable = std::make_shared<TtsServable>(nodeOptions.models_path(), nodeOptions.target_device(), nodeOptions.voices(), nodeOptions.plugin_config(), basePath);
            ttsServableMap.insert(std::pair<std::string, std::shared_ptr<TtsServable>>(nodeName, std::move(servable)));
        } catch (const std::runtime_error& e) {
            SPDLOG_ERROR("TextToSpeech node name: {} initialization failed: {}. ", nodeName, e.what());
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
        return StatusCode::OK;
    }
};

static bool ttsNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<TtsNodeInitializer>());
    return true;
}();
}  // namespace ovms
