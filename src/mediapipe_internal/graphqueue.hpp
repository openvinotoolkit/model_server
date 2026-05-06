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

// Dynamic graph pool that starts at initialSize and expands on demand up to maxSize.
// Does NOT inherit from Queue<T> — isolated from the OV model inference path.
class GraphQueue {
public:
    GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int initialSize, int maxSize);
    ~GraphQueue();

    std::future<int> getIdleStream();
    void returnStream(int id);
    std::shared_ptr<GraphHelper>& getInferRequest(int id);

    int getCurrentSize() const { return currentSize_.load(std::memory_order_relaxed); }
    int getMaxSize() const { return maxSize_; }

private:
    std::shared_ptr<GraphHelper> createOneGraph();

    const ::mediapipe::CalculatorGraphConfig config_;
    std::shared_ptr<GraphSidePackets> sidePacketMaps_;
    const int maxSize_;
    std::atomic<int> currentSize_{0};

    // Pre-allocated to maxSize_; slots filled as pool expands.
    std::vector<std::shared_ptr<GraphHelper>> inferRequests_;

    // Idle stream management
    std::mutex mutex_;
    std::queue<int> idleIds_;
    std::queue<std::promise<int>> waiters_;
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
