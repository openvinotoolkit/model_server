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

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

namespace ovms {

template <typename RequestT>
struct Executor {
    using Request = RequestT;

    std::condition_variable cv;
    std::queue<RequestT> requests;
    std::mutex queueMutex;

    virtual void processRequest() = 0;

    bool hasRequests() {
        std::lock_guard<std::mutex> lock(queueMutex);
        return !requests.empty();
    }

    size_t requestsQueueSize() {
        std::lock_guard<std::mutex> lock(queueMutex);
        return requests.size();
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal) {
        std::unique_lock<std::mutex> lock(queueMutex);
        cv.wait(lock, [this, receivedEndSignal] { return !requests.empty() || *receivedEndSignal; });
    }

    void notify() {
        std::lock_guard<std::mutex> lock(queueMutex);
        cv.notify_one();
    }

    void scheduleRequest(RequestT&& request) {
        std::lock_guard<std::mutex> lock(queueMutex);
        requests.push(std::move(request));
        cv.notify_one();
    }
};

template <typename RequestT>
void runExecutorLoop(Executor<RequestT>* executor, std::atomic<bool>* receivedEndSignal, const std::shared_ptr<spdlog::logger>& logger) {
    while (!(*receivedEndSignal)) {
        SPDLOG_LOGGER_INFO(logger, "All requests: {};", executor->requestsQueueSize());
        if (executor->hasRequests()) {
            executor->processRequest();
        } else {
            executor->waitForRequests(receivedEndSignal);
        }
    }
}

template <typename ExecutorT>
class ExecutorWrapper {
    std::thread executorThread;

    using Request = typename ExecutorT::Request;

    static void run(Executor<Request>* exec, std::atomic<bool>* stop, std::shared_ptr<spdlog::logger> logger) {
        try {
            runExecutorLoop(exec, stop, logger);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(logger, "Error occurred in executor: {}.", e.what());
            exit(1);
        }
    }

protected:
    std::shared_ptr<ExecutorT> executor;
    std::atomic<bool> finishExecutorThread = false;

public:
    ExecutorWrapper(std::shared_ptr<spdlog::logger> logger, std::shared_ptr<ExecutorT> executor) :
        executor(std::move(executor)) {
        executorThread = std::thread(run, this->executor.get(), &finishExecutorThread, logger);
    }

    void addRequest(Request request) {
        if (finishExecutorThread) {
            throw std::runtime_error("Cannot schedule request - executor is stopping");
        }
        executor->scheduleRequest(std::move(request));
    }

    ~ExecutorWrapper() {
        finishExecutorThread = true;
        executor->notify();
        if (executorThread.joinable()) {
            executorThread.join();
        }
    }
};

}  // namespace ovms
