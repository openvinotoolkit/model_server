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
#include "src/mediapipe_internal/live_graph_config.hpp"
#include "src/mediapipe_internal/streaming_s2t_graph.hpp"
#include "src/stringutils.hpp"
#include "s2t_servable.hpp"
#include "streaming_s2t_session.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/audio/speech_to_text/s2t_calculator.pb.h"

#include "src/logging.hpp"

namespace ovms {
class SttNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "S2tCalculator";

public:
    Status prepareLiveGraph(const ::mediapipe::CalculatorGraphConfig& config,
        GraphSidePackets& sidePackets) override {
        if (config.node_size() != 1 || sidePackets.sttServableMap.size() != 1)
            return StatusCode::OK;
        const auto& node = config.node(0);
        if (node.calculator() != CALCULATOR_NAME && node.calculator() != "StreamingS2TCalculator")
            return StatusCode::OK;
        const auto servable = sidePackets.sttServableMap.find(node.name());
        if (servable == sidePackets.sttServableMap.end())
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Missing initialized STT resources");
        auto profile = std::make_shared<LiveGraphConfig>();
        profile->graph = config;
        profile->graph.set_max_queue_size(64);
        profile->graph.clear_input_stream();
        profile->graph.add_input_stream("audio");
        profile->graph.clear_output_stream();
        profile->graph.add_output_stream("events");
        auto* audioNode = profile->graph.mutable_node(0);
        audioNode->clear_input_stream();
        audioNode->add_input_stream("AUDIO:audio");
        audioNode->clear_output_stream();
        audioNode->add_output_stream("EVENT:events");
        audioNode->clear_input_stream_info();
        audioNode->clear_input_stream_handler();
        profile->inputs.emplace("audio", LiveGraphStreamCodec::typed<StreamingS2TInput>());
        profile->outputs.emplace("events", LiveGraphStreamCodec::typed<StreamingS2TEvent>());
        mediapipe::CalculatorGraph validationGraph;
        StreamingS2TGraphExecution validation(validationGraph);
        const auto status = validation.initialize(profile->graph);
        if (!status.ok()) {
            SPDLOG_ERROR("Invalid live STT graph: {}", status.ToString());
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, status.ToString());
        }
        const char* name = servable->second->streamingBackend ? STT_INCREMENTAL_PROFILE : STT_UTTERANCE_PROFILE;
        if (!sidePackets.liveGraphProfiles.emplace(name, std::move(profile)).second)
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Duplicate live STT profile");
        return StatusCode::OK;
    }

    bool matches(const std::string& calculatorName) const override {
        return endsWith(calculatorName, CALCULATOR_NAME) || calculatorName == "StreamingS2TCalculator";
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
        if (nodeConfig.calculator() == "StreamingS2TCalculator" && !servable->streamingBackend) {
            SPDLOG_ERROR("StreamingS2TCalculator requires an incremental ASR backend in graph: {}", graphName);
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
        sttServableMap.insert(std::pair<std::string, std::shared_ptr<SttServable>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool sttNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<SttNodeInitializer>());
    return true;
}();
}  // namespace ovms
