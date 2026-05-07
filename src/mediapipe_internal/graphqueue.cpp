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
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

#include "src/python/pythonnoderesources.hpp"
#include "src/llm/servable.hpp"

#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma warning(pop)

#include "graph_executor_constants.hpp"
#include "outputstreamobserver.hpp"
#include "side_packet_builder.hpp"
namespace ovms {

std::shared_ptr<GraphHelper> GraphQueue::createOneGraph() {
    std::unordered_map<std::string, std::shared_ptr<ObserverHolder>> observers;
    for (auto& name : config_.output_stream()) {
        std::string streamName = getStreamName(name);
        auto holder = std::make_shared<ObserverHolder>();
        holder->current = std::make_shared<NullOutputStreamObserver>();
        observers.emplace(std::move(streamName), std::move(holder));
    }

    auto gh = std::make_shared<GraphHelper>(std::move(observers));
    gh->graph = std::make_unique<::mediapipe::CalculatorGraph>();
    gh->currentTimestamp = ::mediapipe::Timestamp(0);

    auto absStatus = gh->graph->Initialize(config_);
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Graph queue initialization failed: {}", absStatus.ToString());
        throw std::runtime_error(absStatus.ToString());
    }
    for (const auto& [streamName, holder] : gh->outStreamObservers) {
        absStatus = gh->graph->ObserveOutputStream(streamName, [holder](const ::mediapipe::Packet& packet) -> absl::Status { return holder->current->handlePacket(packet); });
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph queue ObserveOutputStream failed: {}", absStatus.ToString());
            throw std::runtime_error(absStatus.ToString());
        }
    }
    for (const auto& [nodeName, _] : sidePacketMaps_->genAiServableMap) {
        gh->genAiExecutionContextMap[nodeName] = std::make_shared<GenAiExecutionContextHolder>();
    }
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    buildInputSidePackets(inputSidePackets, *sidePacketMaps_);
    inputSidePackets[LLM_EXECUTION_CONTEXT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiExecutionContextMap>(gh->genAiExecutionContextMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
    absStatus = gh->graph->StartRun(inputSidePackets);
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Graph queue StartRun failed: {}", absStatus.ToString());
        throw std::runtime_error(absStatus.ToString());
    }
    return gh;
}

GraphQueue::GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int initialSize, int maxSize) :
    config_(config),
    sidePacketMaps_(sidePacketMaps),
    maxSize_(maxSize) {
    inferRequests_.resize(maxSize_);
    for (int i = 0; i < initialSize; ++i) {
        inferRequests_[i] = createOneGraph();
        idleIds_.push(i);
    }
    currentSize_.store(initialSize, std::memory_order_relaxed);
    SPDLOG_DEBUG("Graph queue created with initial size {} and max size {}", initialSize, maxSize_);
}

GraphQueue::~GraphQueue() {
    int size = currentSize_.load(std::memory_order_relaxed);
    for (int i = 0; i < size; ++i) {
        auto& graphHelper = inferRequests_[i];
        if (!graphHelper)
            continue;
        auto absStatus = graphHelper->graph->WaitUntilIdle();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue WaitUntilIdle error: {}", absStatus.ToString());
        }
        absStatus = graphHelper->graph->CloseAllPacketSources();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue CloseAllPacketSources error: {}", absStatus.ToString());
        }
        absStatus = graphHelper->graph->WaitUntilDone();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue WaitUntilDone error: {}", absStatus.ToString());
        }
        graphHelper->graph->Cancel();
        graphHelper->graph.reset();
    }
}

std::future<int> GraphQueue::getIdleStream() {
    std::promise<int> promise;
    std::future<int> future = promise.get_future();

    std::unique_lock<std::mutex> lk(mutex_);
    if (!idleIds_.empty()) {
        int id = idleIds_.front();
        idleIds_.pop();
        lk.unlock();
        promise.set_value(id);
        return future;
    }

    // No idle graph available — try to expand
    int currentSize = currentSize_.load(std::memory_order_relaxed);
    if (currentSize < maxSize_) {
        int newId = currentSize;
        currentSize_.store(currentSize + 1, std::memory_order_relaxed);
        lk.unlock();
        // Create graph outside the lock (expensive but only blocks this request)
        try {
            inferRequests_[newId] = createOneGraph();
            SPDLOG_DEBUG("Graph queue expanded to size {}/{}", newId + 1, maxSize_);
        } catch (const std::exception& e) {
            // Rollback size on failure
            currentSize_.fetch_sub(1, std::memory_order_relaxed);
            promise.set_exception(std::make_exception_ptr(e));
            return future;
        }
        promise.set_value(newId);
        return future;
    }

    // At max capacity — must wait for a graph to be returned
    waiters_.push(std::move(promise));
    return future;
}

void GraphQueue::returnStream(int id) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!waiters_.empty()) {
        std::promise<int> waiter = std::move(waiters_.front());
        waiters_.pop();
        lk.unlock();
        waiter.set_value(id);
        return;
    }
    idleIds_.push(id);
}

std::shared_ptr<GraphHelper>& GraphQueue::getInferRequest(int id) {
    return inferRequests_[id];
}

}  // namespace ovms
