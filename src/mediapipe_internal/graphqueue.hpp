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

class GraphQueue : public Queue<std::unique_ptr<::mediapipe::CalculatorGraph>> {
public:
    /**
    * @brief Allocating idle stream for execution
    */
    std::future<int> getIdleStream() {
        // OVMS_PROFILE_FUNCTION();
        int value;
        std::promise<int> idleStreamPromise;
        std::future<int> idleStreamFuture = idleStreamPromise.get_future();
        std::unique_lock<std::mutex> lk(front_mut);
        if (streams[front_idx] < 0) {  // we need to wait for any idle stream to be returned
            std::unique_lock<std::mutex> queueLock(queue_mutex);
            promises.push(std::move(idleStreamPromise));
        } else {  // we can give idle stream right away
            value = streams[front_idx];
            streams[front_idx] = -1;  // negative value indicate consumed vector index
            front_idx = (front_idx + 1) % streams.size();
            lk.unlock();
            idleStreamPromise.set_value(value);
        }
        return idleStreamFuture;
    }

    GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, int streamsLength) : Queue(streamsLength) {
        inferRequests.reserve(streamsLength);
        for (auto i =0; i < streamsLength; ++i) {
            inferRequests.emplace_back(std::make_unique<::mediapipe::CalculatorGraph>());
            std::ignore = inferRequests.back()->Initialize(config); // TODO FIXME
        }
    }
};

struct GraphIdGuard {
    GraphQueue& queue;
    const int id;
    ::mediapipe::CalculatorGraph& graph;
GraphIdGuard(GraphQueue& queue) :
        queue(queue),
       id(queue.getIdleStream().get()),
       graph(*(queue.getInferRequest(id).get())) {}
~GraphIdGuard(){
        this->queue.returnStream(this->id);
}
};
}  // namespace ovms
