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
#include "graphqueue.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "../queue.hpp"
#if (PYTHON_DISABLE == 0)
#include "src/python/pythonnoderesources.hpp"
#endif

#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma warning(pop)

#include "graph_executor_constants.hpp"
#include "outputstreamobserver.hpp"
#include "side_packet_builder.hpp"
namespace ovms {
GraphQueue::GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int streamsLength) :
    Queue(streamsLength),
    sidePacketMaps(sidePacketMaps) {
    inferRequests.reserve(streamsLength);
    for (auto i = 0; i < streamsLength; ++i) {
        // Build observer map locally before constructing GraphHelper (const map)
        std::unordered_map<std::string, std::shared_ptr<ObserverHolder>> observers;
        for (auto& name : config.output_stream()) {
            std::string streamName = getStreamName(name);
            auto holder = std::make_shared<ObserverHolder>();
            holder->current = std::make_shared<NullOutputStreamObserver>();
            observers[streamName] = holder;
        }

        auto graphHelper = std::make_shared<GraphHelper>(std::move(observers));
        graphHelper->graph = std::make_unique<::mediapipe::CalculatorGraph>();
        graphHelper->currentTimestamp = ::mediapipe::Timestamp(0);

        auto absStatus = graphHelper->graph->Initialize(config);
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph queue initialization failed: {}", absStatus.ToString());
            throw std::runtime_error(absStatus.ToString());
        }
        for (const auto& [streamName, holder] : graphHelper->outStreamObservers) {
            // Lambda captures holder (shared_ptr) by value — safe regardless of map layout
            absStatus = graphHelper->graph->ObserveOutputStream(streamName, [holder](const ::mediapipe::Packet& packet) -> absl::Status { return holder->current->handlePacket(packet); });
            if (!absStatus.ok()) {
                SPDLOG_ERROR("Graph queue ObserveOutputStream failed: {}", absStatus.ToString());
                throw std::runtime_error(absStatus.ToString());
            }
        }
        for (const auto& [nodeName, _] : sidePacketMaps->genAiServableMap) {
            graphHelper->genAiExecutionContextMap[nodeName] = std::make_shared<GenAiExecutionContextHolder>();
        }
        std::map<std::string, mediapipe::Packet> inputSidePackets;
        buildInputSidePackets(inputSidePackets, *sidePacketMaps);
        // Override execution context with per-graph instance
        inputSidePackets[LLM_EXECUTION_CONTEXT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiExecutionContextMap>(graphHelper->genAiExecutionContextMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        absStatus = graphHelper->graph->StartRun(inputSidePackets);
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph queue StartRun failed: {}", absStatus.ToString());
            throw std::runtime_error(absStatus.ToString());
        }
        this->inferRequests.emplace_back(std::move(graphHelper));
    }
}
GraphHelper::~GraphHelper() {
    if (!graph) {
        return;
    }
    auto absStatus = graph->WaitUntilIdle();
    if (!absStatus.ok()) {
        SPDLOG_DEBUG("GraphHelper WaitUntilIdle error: {}", absStatus.ToString());
    }
    absStatus = graph->CloseAllPacketSources();
    if (!absStatus.ok()) {
        SPDLOG_DEBUG("GraphHelper CloseAllPacketSources error: {}", absStatus.ToString());
    }
    absStatus = graph->WaitUntilDone();
    if (!absStatus.ok()) {
        SPDLOG_DEBUG("GraphHelper WaitUntilDone error: {}", absStatus.ToString());
    }
    graph->Cancel();
}

void GraphHelper::reinitialize(const ::mediapipe::CalculatorGraphConfig& config, const GraphSidePackets& sidePacketMaps) {
    SPDLOG_DEBUG("Reinitializing graph after error");
    // Tear down the old graph (best-effort, errors expected since graph is in bad state)
    if (this->graph) {
        auto absStatus = this->graph->CloseAllPacketSources();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("reinitialize: CloseAllPacketSources: {}", absStatus.ToString());
        }
        absStatus = this->graph->WaitUntilDone();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("reinitialize: WaitUntilDone: {}", absStatus.ToString());
        }
        this->graph->Cancel();
    }
    // Create fresh graph
    graph = std::make_unique<::mediapipe::CalculatorGraph>();
    currentTimestamp = ::mediapipe::Timestamp(0);

    auto absStatus = graph->Initialize(config);
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Graph reinitialize: Initialize failed: {}", absStatus.ToString());
        graph.reset();
        return;
    }
    for (const auto& [streamName, holder] : outStreamObservers) {
        absStatus = graph->ObserveOutputStream(streamName, [holder](const ::mediapipe::Packet& packet) -> absl::Status {
            return holder->current->handlePacket(packet);
        });
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph reinitialize: ObserveOutputStream failed: {}", absStatus.ToString());
            graph.reset();
            return;
        }
    }
    // Reset observers to null sentinel
    for (const auto& [streamName, holder] : outStreamObservers) {
        holder->current = std::make_shared<NullOutputStreamObserver>();
    }
    // Reset execution contexts
    for (auto& [nodeName, ctx] : genAiExecutionContextMap) {
        ctx->reset();
    }
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    buildInputSidePackets(inputSidePackets, sidePacketMaps);
    inputSidePackets[LLM_EXECUTION_CONTEXT_SESSION_SIDE_PACKET_TAG] =
        mediapipe::MakePacket<GenAiExecutionContextMap>(genAiExecutionContextMap)
            .At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
    absStatus = graph->StartRun(inputSidePackets);
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Graph reinitialize: StartRun failed: {}", absStatus.ToString());
        graph.reset();
        return;
    }
    SPDLOG_DEBUG("Graph reinitialized successfully");
}
GraphQueue::~GraphQueue() = default;
}  // namespace ovms
