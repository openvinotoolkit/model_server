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
#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/queue.hpp"

#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "graph_executor_constants.hpp"
#include "graph_side_packets.hpp"
#include "outputstreamobserver.hpp"
namespace ovms {
class OutputStreamObserverI;
class NullOutputStreamObserver;
struct ObserverHolder;
struct GraphHelper {
    std::unique_ptr<::mediapipe::CalculatorGraph> graph;
    // const after construction: keys are fixed, but observer implementations
    // can be swapped via the mutable ObserverHolder inside each shared_ptr.
    const std::unordered_map<std::string, std::shared_ptr<ObserverHolder>> outStreamObservers;
    GenAiExecutionContextMap genAiExecutionContextMap;
    ::mediapipe::Timestamp currentTimestamp;
    GraphHelper() = default;
    // Constructor that takes the pre-built observer map
    GraphHelper(std::unordered_map<std::string, std::shared_ptr<ObserverHolder>>&& observers) :
        outStreamObservers(std::move(observers)) {}
    GraphHelper(const GraphHelper&) = delete;
    GraphHelper& operator=(const GraphHelper&) = delete;
    GraphHelper(GraphHelper&& gh) :
        graph(std::move(gh.graph)),
        outStreamObservers(std::move(const_cast<std::unordered_map<std::string, std::shared_ptr<ObserverHolder>>&>(gh.outStreamObservers))),
        genAiExecutionContextMap(std::move(gh.genAiExecutionContextMap)),
        currentTimestamp(gh.currentTimestamp) {}
    GraphHelper& operator=(GraphHelper&&) = delete;
};
// we need to keep Graph alive during MP reload hence shared_ptr
class GraphQueue : public Queue<std::shared_ptr<GraphHelper>> {
public:  // XXX TODO make private? we need to access in mediapipegraphdefinition to set side packets though
    std::shared_ptr<GraphSidePackets> sidePacketMaps;

public:
    GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int streamsLength);
    ~GraphQueue();
};

struct GraphIdGuard {
    std::weak_ptr<GraphQueue> weakQueue;
    const int id;
    // shared_ptr because GraphIdGuard (and the executor holding it) must keep
    // the GraphHelper alive even after the GraphQueue is destroyed during
    // mediapipe graph reload/retire — the in-flight request continues using
    // the old graph until completion.
    std::shared_ptr<GraphHelper> gh;
    ::mediapipe::CalculatorGraph& graph;
    GraphIdGuard(std::shared_ptr<GraphQueue>& queue) :
        weakQueue(queue),
        id(queue->getIdleStream().get()),
        gh((queue->getInferRequest(id))),
        graph(*gh->graph) {
    }
    GraphIdGuard(GraphIdGuard&&) = default;
    GraphIdGuard(const GraphIdGuard&) = delete;
    ~GraphIdGuard() {
        auto existingQueue = weakQueue.lock();
        if (existingQueue)
            existingQueue->returnStream(this->id);
    }
};
}  // namespace ovms
