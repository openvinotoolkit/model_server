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
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "../queue.hpp"

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"

#include "graph_executor_constants.hpp"
#include "graph_side_packets.hpp"
#include "outputstreamobserver.hpp"
namespace ovms {
class OutputStreamObserverI;
class NullOutputStreamObserver;
struct GraphHelper {
    std::shared_ptr<::mediapipe::CalculatorGraph> graph;  // TODO FIXME this does not have to be shared_ptr
    std::unordered_map<std::string, std::shared_ptr<OutputStreamObserverI>> outStreamObservers;
    ::mediapipe::Timestamp currentTimestamp;  // TODO FIXME const
    // TODO FIXME move constr/=
    GraphHelper() = default;
    GraphHelper(const GraphHelper&) = delete;
    GraphHelper& operator=(const GraphHelper&) = delete;
    GraphHelper(GraphHelper&& gh) :
        graph(std::move(gh.graph)),
        outStreamObservers(std::move(gh.outStreamObservers)),
        currentTimestamp(gh.currentTimestamp) {}
    GraphHelper& operator=(GraphHelper&& gh) = default;
};
// we need to keep Graph alive during MP reload hence shared_ptr
//class GraphQueue : public Queue<std::shared_ptr<::mediapipe::CalculatorGraph>> {
class GraphQueue : public Queue<std::shared_ptr<GraphHelper>> {
    public: // XXX TODO make private? we need to acces in mediapipegraphdefinition to set side packets though
    std::shared_ptr<GraphSidePackets> sidePacketMaps;

public:
    GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int streamsLength);
    ~GraphQueue();
};

struct GraphIdGuard {
    std::weak_ptr<GraphQueue> weakQueue;
    const int id;
    std::shared_ptr<GraphHelper> gh;
    // TODO FIXME shared_ptr
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
