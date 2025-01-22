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

// #include "profiler.hpp"

namespace ovms {

template <typename T>
class Queue {
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

    std::optional<int> tryToGetIdleStream() {
        // OVMS_PROFILE_FUNCTION();
        int value;
        std::unique_lock<std::mutex> lk(front_mut);
        if (streams[front_idx] < 0) {  // we need to wait for any idle stream to be returned
            return std::nullopt;
        } else {  // we can give idle stream right away
            value = streams[front_idx];
            streams[front_idx] = -1;  // negative value indicate consumed vector index
            front_idx = (front_idx + 1) % streams.size();
            return value;
        }
    }

    /**
    * @brief Release stream after execution
    */
    void returnStream(int streamID) {
        // OVMS_PROFILE_FUNCTION();
        std::unique_lock<std::mutex> lk(queue_mutex);
        if (promises.size()) {
            std::promise<int> promise = std::move(promises.front());
            promises.pop();
            lk.unlock();
            promise.set_value(streamID);
            return;
        }
        std::uint32_t old_back = back_idx.load();
        while (!back_idx.compare_exchange_weak(
            old_back,
            (old_back + 1) % streams.size(),
            std::memory_order_relaxed)) {
        }
        streams[old_back] = streamID;
    }

    /**
    * @brief Constructor with initialization
    */
    Queue(int streamsLength) :
        streams(streamsLength),
        front_idx{0},
        back_idx{0} {
        for (int i = 0; i < streamsLength; ++i) {
            streams[i] = i;
        }
    }

    /**
     * @brief Give InferRequest
     */
    T& getInferRequest(int streamID) {
        return inferRequests[streamID];
    }

protected:
    /**
    * @brief Vector representing circular buffer for infer queue
    */
    std::vector<int> streams;

    /**
    * @brief Index of the front of the idle streams list
    */
    std::uint32_t front_idx;

    /**
    * @brief Index of the back of the idle streams list
    */
    std::atomic<std::uint32_t> back_idx;

    /**
    * @brief Vector representing OV streams and used for notification about completed inference operations
    */
    std::mutex front_mut;
    std::mutex queue_mutex;
    /**
     * 
     */
    std::vector<T> inferRequests;
    std::queue<std::promise<int>> promises;
};
}  // namespace ovms
