//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <thread>

#include <continuous_batching_pipeline.hpp>

#include "../logging.hpp"
#include "../profiler.hpp"

namespace ovms {
struct LLMExecutor {
    // For logging purposes we could have more information about graph and node here
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<ContinuousBatchingPipeline> pipe = nullptr;

    LLMExecutor(std::shared_ptr<ContinuousBatchingPipeline> pipe) {
        this->pipe = pipe;
    }

    bool hasRequests() {
        return (pipe->has_non_finished_requests());
    }

    void step() {
        OVMS_PROFILE_FUNCTION();
        pipe->step();
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this, receivedEndSignal] { return (pipe->has_non_finished_requests() || *receivedEndSignal); });
    }

    void notify() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_one();
    }
};

class LLMExecutorWrapper {
    LLMExecutor llmExecutor;
    std::thread llmExecutorThread;
    std::atomic<bool> finishExecutorThread = false;

    static void run(LLMExecutor* llmExecutor, std::atomic<bool>* receivedEndSignal) {
        while (!(*receivedEndSignal)) {
            try {
                if (llmExecutor->hasRequests()) {
                    llmExecutor->step();
                } else {
                    llmExecutor->waitForRequests(receivedEndSignal);
                }
            } catch (std::exception& e) {
                SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in LLM executor: {}.", e.what());
            }
        }
    }

public:
    LLMExecutorWrapper(std::shared_ptr<ContinuousBatchingPipeline> pipe) :
        llmExecutor(pipe) {
        llmExecutorThread = std::thread(LLMExecutorWrapper::run, &llmExecutor, &finishExecutorThread);
    }

    ~LLMExecutorWrapper() {
        finishExecutorThread = true;
        llmExecutor.notify();
        llmExecutorThread.join();
    }

    void notifyNewRequestArrived() {
        llmExecutor.notify();
    }
};
}  // namespace ovms
