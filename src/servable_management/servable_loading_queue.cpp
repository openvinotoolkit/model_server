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
#include "servable_loading_queue.hpp"

#include <utility>

#include "src/logging.hpp"

namespace ovms {

ServableLoadingQueue::~ServableLoadingQueue() {
    stop();
}

void ServableLoadingQueue::start(TaskProcessor processor) {
    std::lock_guard<std::mutex> lock(this->mutex);
    if (this->running) {
        return;
    }
    this->processor = std::move(processor);
    this->running = true;
    this->worker = std::thread(&ServableLoadingQueue::workerLoop, this);
}

void ServableLoadingQueue::requestStop() {
    {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->running) {
            return;
        }
        this->running = false;
    }
    this->cv.notify_one();
}

void ServableLoadingQueue::setTaskObserver(TaskObserver observer) {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->taskObserver = std::move(observer);
}

void ServableLoadingQueue::stop() {
    requestStop();
    if (this->worker.joinable()) {
        this->worker.join();
    }
    std::lock_guard<std::mutex> lock(this->mutex);
    while (!this->queue.empty()) {
        auto& task = this->queue.front();
        task.completion.set_value(StatusCode::SERVER_SHUTTING_DOWN);
        this->queue.pop_front();
    }
}

std::future<Status> ServableLoadingQueue::scheduleTask(ServableLoadingTask task) {
    auto future = task.completion.get_future();
    {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->running) {
            task.completion.set_value(StatusCode::SERVER_SHUTTING_DOWN);
            return future;
        }
        if (this->taskObserver) {
            this->taskObserver(TaskEvent::Scheduled, task);
        }
        if (task.urgent) {
            this->queue.push_front(std::move(task));
        } else {
            this->queue.push_back(std::move(task));
        }
    }
    this->cv.notify_one();
    return future;
}

void ServableLoadingQueue::workerLoop() {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Started servable loading queue thread");
    while (true) {
        ServableLoadingTask task{ServableLoadingTaskType::LoadModel, ""};
        TaskObserver observer;
        {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->cv.wait(lock, [this] { return !this->queue.empty() || !this->running; });
            if (!this->running) {
                break;
            }
            task = std::move(this->queue.front());
            this->queue.pop_front();
            // Copy under the lock: reading it unlocked would race with setTaskObserver(),
            // and calling it locked would deadlock an observer that re-enters the queue.
            observer = this->taskObserver;
        }
        if (observer) {
            observer(TaskEvent::Executed, task);
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Processing {} task for: {}",
            static_cast<int>(task.type), task.name);
        Status status = this->processor(task);
        task.completion.set_value(status);
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Stopped servable loading queue thread");
}

}  // namespace ovms
