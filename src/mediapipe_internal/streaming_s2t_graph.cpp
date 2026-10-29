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
#include "streaming_s2t_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>

namespace ovms {

StreamingS2TGraphExecution::StreamingS2TGraphExecution(mediapipe::CalculatorGraph& graph) :
    graph_(graph) {}

StreamingS2TGraphExecution::~StreamingS2TGraphExecution() {
    invalidate();
}

absl::Status StreamingS2TGraphExecution::initialize(const mediapipe::CalculatorGraphConfig& config) {
    const bool hasStreamingS2T = std::any_of(config.node().begin(), config.node().end(),
        [](const auto& node) { return node.calculator() == "StreamingS2TCalculator"; });
    if (!hasStreamingS2T)
        return graph_.Initialize(config);

    auto runtimeConfig = config;
    size_t index = 0;
    for (auto& node : *runtimeConfig.mutable_node()) {
        if (node.calculator() != "StreamingS2TCalculator")
            continue;
        const std::string input = "__ovms_stt_ready_" + std::to_string(index++);
        runtimeConfig.add_input_stream(input);
        node.add_input_stream("WAKE:" + input);
        node.add_input_side_packet("STREAMING_S2T_RESOURCES:__ovms_stt_bridges");
        node.mutable_input_stream_handler()->Clear();
        node.mutable_input_stream_handler()->set_input_stream_handler("ImmediateInputStreamHandler");
        // Old exported HTTP graphs contain a self-sustaining polling loop.
        for (auto* streams : {node.mutable_input_stream(), node.mutable_output_stream()}) {
            for (int i = streams->size() - 1; i >= 0; --i) {
                if (streams->Get(i).rfind("LOOPBACK:", 0) == 0)
                    streams->DeleteSubrange(i, 1);
            }
        }
        node.clear_input_stream_info();
        auto timestamp = std::make_shared<int64_t>(0);
        bridges_.emplace(node.name(), std::make_shared<StreamingS2TBridge>(
            [this, input, timestamp] {
                return graph_.AddPacketToInputStream(input,
                    mediapipe::MakePacket<bool>(true).At(mediapipe::Timestamp(++*timestamp)));
            }));
    }
    if (!bridges_.empty())
        runtimeConfig.add_input_side_packet("__ovms_stt_bridges");
    auto status = graph_.Initialize(runtimeConfig);
    if (status.ok() && !bridges_.empty())
        graph_.SetGraphInputStreamAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
    return status;
}

void StreamingS2TGraphExecution::addSidePackets(std::map<std::string, mediapipe::Packet>& packets) const {
    if (!bridges_.empty())
        packets["__ovms_stt_bridges"] = mediapipe::MakePacket<StreamingS2TBridgeMap>(bridges_)
                                               .At(mediapipe::Timestamp(0));
}

absl::Status StreamingS2TGraphExecution::waitUntilIdle() {
    if (bridges_.empty())
        return graph_.WaitUntilIdle();
    // WaitUntilIdle alone does not include work submitted to inference workers.
    while (true) {
        auto status = graph_.WaitUntilIdle();
        if (!status.ok())
            return status;
        bool busy = false;
        for (const auto& entry : bridges_) {
            status = entry.second->waitFor(std::chrono::milliseconds(20));
            if (!status.ok())
                return status;
            busy = entry.second->busy() || busy;
        }
        if (!busy)
            return graph_.WaitUntilIdle();
    }
}

void StreamingS2TGraphExecution::invalidate() {
    for (const auto& entry : bridges_)
        entry.second->invalidate();
}
}  // namespace ovms
