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
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "servable_loading_task.hpp"

namespace ovms {

using TaskProcessor = std::function<Status(ServableLoadingTask&)>;

enum class TaskEvent {
    Scheduled,
    Executed
};
using TaskObserver = std::function<void(TaskEvent, const ServableLoadingTask&)>;

class ServableLoadingQueue {
public:
    ServableLoadingQueue() = default;
    ~ServableLoadingQueue();

    ServableLoadingQueue(const ServableLoadingQueue&) = delete;
    ServableLoadingQueue& operator=(const ServableLoadingQueue&) = delete;

    void start(TaskProcessor processor);
    // Blocks until worker joins, then drains pending tasks. Must always be called.
    void stop();
    // Signals worker to stop without blocking. stop() must still be called after.
    void requestStop();

    void setTaskObserver(TaskObserver observer);

    std::future<Status> scheduleTask(ServableLoadingTask task);

private:
    void workerLoop();

    TaskProcessor processor;
    TaskObserver taskObserver;
    std::thread worker;
    std::deque<ServableLoadingTask> queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool running = false;
};

}  // namespace ovms
