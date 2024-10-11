//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <condition_variable>
#include <optional>
#include <queue>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

namespace ovms {

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() {}
    ~ThreadSafeQueue() {}
    void push(const T& element) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.push(std::move(element));
        lock.unlock();
        signal.notify_one();
    }

    void push(T&& element) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.push(std::move(element));
        lock.unlock();
        signal.notify_one();
    }

    std::optional<T> tryPull(const uint32_t waitDurationMicroseconds) {
        std::unique_lock<std::mutex> lock(mtx);
        if (signal.wait_for(lock,
                std::chrono::microseconds(waitDurationMicroseconds),
                [this]() { return queue.size() > 0; })) {
            T element = std::move(queue.front());
            queue.pop();
            return std::optional<T>{std::move(element)};
        } else {
            return std::nullopt;
        }
    }

    size_t size() {
        return queue.size();
    }

private:
    std::mutex mtx;
    std::queue<T> queue;
    std::condition_variable signal;
};
}  // namespace ovms
