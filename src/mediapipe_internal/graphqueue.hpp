//*****************************************************************************
// Copyright 2021 Intel Corporation
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
namespace ovms {
// we need to keep Graph alive during MP reload hence shared_ptr
class GraphQueue : public Queue<std::shared_ptr<::mediapipe::CalculatorGraph>> {
public:
    GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, int streamsLength) :
        Queue(streamsLength) {
        SPDLOG_ERROR("ER Constr graph queue:{}", (void*)this);
        inferRequests.reserve(streamsLength);
        for (auto i = 0; i < streamsLength; ++i) {
            inferRequests.emplace_back(std::make_shared<::mediapipe::CalculatorGraph>());
            std::ignore = inferRequests.back()->Initialize(config);  // TODO FIXME
        }
    }
    ~GraphQueue() {
        SPDLOG_ERROR("ER Destroy graph queue:{}", (void*)this);
    }
};

struct GraphIdGuard {
    std::weak_ptr<GraphQueue> weakQueue;
    const int id;
    ::mediapipe::CalculatorGraph& graph;
    GraphIdGuard(std::shared_ptr<GraphQueue>& queue) :
        weakQueue(queue),
        id(queue->getIdleStream().get()),
        graph(*(queue->getInferRequest(id).get())) {}
    GraphIdGuard(GraphIdGuard&&) = default;
    GraphIdGuard(const GraphIdGuard&) = delete;
    ~GraphIdGuard() {
        auto existingQueue = weakQueue.lock();
        SPDLOG_ERROR("ER DEstroy Guard begin qu:{}", (void*)existingQueue.get());
        if (existingQueue)
            existingQueue->returnStream(this->id);
        SPDLOG_ERROR("ER Destroy Guard end qu:{}", (void*)existingQueue.get());
    }
};
}  // namespace ovms
