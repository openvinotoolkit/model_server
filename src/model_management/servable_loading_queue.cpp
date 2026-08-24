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

void ServableLoadingQueue::stop() {
    {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->running) {
            return;
        }
        this->running = false;
    }
    this->cv.notify_one();
    if (this->worker.joinable()) {
        this->worker.join();
    }
}

std::future<Status> ServableLoadingQueue::scheduleTask(ServableLoadingTask task, bool urgent) {
    auto future = task.completion.get_future();
    {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (urgent) {
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
        {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->cv.wait(lock, [this] { return !this->queue.empty() || !this->running; });
            if (!this->running && this->queue.empty()) {
                break;
            }
            task = std::move(this->queue.front());
            this->queue.pop_front();
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Processing {} task for: {}",
            static_cast<int>(task.type), task.name);
        Status status = this->processor(task);
        task.completion.set_value(status);
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Stopped servable loading queue thread");
}

}  // namespace ovms
