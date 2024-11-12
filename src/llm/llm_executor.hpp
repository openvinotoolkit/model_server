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
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <openvino/genai/continuous_batching_pipeline.hpp>

#include "../logging.hpp"
#include "../profiler.hpp"

namespace ovms {
struct LLMExecutor {
    // For logging purposes we could have more information about graph and node here
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe = nullptr;

    LLMExecutor(std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe) {
        this->pipe = std::move(pipe);
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    void printMetrics() {
        ov::genai::PipelineMetrics metrics = pipe->get_metrics();
        SPDLOG_LOGGER_INFO(llm_executor_logger, "All requests: {}; Scheduled requests: {}; Cache usage {:.1f}%;",
            metrics.requests, metrics.scheduled_requests, metrics.cache_usage);
    }
};
#pragma GCC diagnostic pop

class LLMExecutorWrapper {
    LLMExecutor llmExecutor;
    std::thread llmExecutorThread;
    std::atomic<bool> finishExecutorThread = false;

    static void run(LLMExecutor* llmExecutor, std::atomic<bool>* receivedEndSignal) {
        const uint8_t printMetricsEveryNumberOfSteps = 10;
        uint8_t stepCounter = 0;
        while (!(*receivedEndSignal)) {
            try {
                if (stepCounter == printMetricsEveryNumberOfSteps) {
                    llmExecutor->printMetrics();
                    stepCounter = 0;
                }
                if (llmExecutor->hasRequests()) {
                    stepCounter++;
                    llmExecutor->step();
                } else {
                    SPDLOG_LOGGER_INFO(llm_executor_logger, "All requests: {}; Scheduled requests: {};", 0, 0);
                    llmExecutor->waitForRequests(receivedEndSignal);
                }
            } catch (std::exception& e) {
                SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in LLM executor: {}.", e.what());
                exit(1);
            }
        }
    }

public:
    LLMExecutorWrapper(std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe) :
        llmExecutor(std::move(pipe)) {
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
