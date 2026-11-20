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
#include "node_initializer.hpp"

#include <utility>

#include "mediapipe/framework/calculator_graph.h"
#include "src/logging.hpp"
#include "graph_side_packets.hpp"
#include "live_graph_config.hpp"
#include "streaming_s2t_graph.hpp"

namespace ovms {
std::shared_ptr<LiveGraphConfig> makeLiveGraphConfig(const mediapipe::CalculatorGraphConfig& config,
    int queueSize, std::map<std::string, LiveGraphStreamCodec> inputs,
    std::map<std::string, LiveGraphStreamCodec> outputs) {
    auto profile = std::make_shared<LiveGraphConfig>();
    profile->graph = config;
    profile->graph.set_max_queue_size(queueSize);
    profile->graph.clear_input_stream();
    profile->graph.clear_output_stream();
    for (const auto& entry : inputs)
        profile->graph.add_input_stream(entry.first);
    for (const auto& entry : outputs)
        profile->graph.add_output_stream(entry.first);
    profile->inputs = std::move(inputs);
    profile->outputs = std::move(outputs);
    return profile;
}

Status registerLiveGraphProfile(const std::string& name, std::shared_ptr<LiveGraphConfig> profile,
    GraphSidePackets& sidePackets) {
    mediapipe::CalculatorGraph graph;
    StreamingS2TGraphExecution validation(graph);
    const auto status = validation.initialize(profile->graph);
    if (!status.ok()) {
        SPDLOG_ERROR("Invalid live graph profile {}: {}", name, status.ToString());
        return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, status.ToString());
    }
    if (!sidePackets.liveGraphProfiles.emplace(name, std::move(profile)).second)
        return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Duplicate live graph profile: " + name);
    return StatusCode::OK;
}

NodeInitializerRegistry& NodeInitializerRegistry::instance() {
    static NodeInitializerRegistry registry;
    return registry;
}

void NodeInitializerRegistry::add(std::unique_ptr<NodeInitializer> initializer) {
    initializers_.push_back(std::move(initializer));
}

const std::vector<std::unique_ptr<NodeInitializer>>& NodeInitializerRegistry::all() const {
    return initializers_;
}
}  // namespace ovms
