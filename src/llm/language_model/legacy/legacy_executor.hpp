//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <queue>

#include <openvino/genai/continuous_batching_pipeline.hpp>

#include "../../../logging.hpp"
#include "../../../profiler.hpp"

#include <chrono>
#include <thread>
#include <future>

namespace ovms {
struct LegacyServableExecutionContext : public GenAiServableExecutionContext {
    ov::genai::EncodedResults results;
    std::mutex mutex;
    std::promise<void> readySignal;
    std::future<void> finished = readySignal.get_future();
};

struct LegacyExecutor {
    // For logging purposes we could have more information about graph and node here
    std::condition_variable cv;
    std::queue<std::shared_ptr<LegacyServableExecutionContext>> requests;
    std::mutex queueMutex;
    std::shared_ptr<ov::genai::LLMPipeline> pipe;

    LegacyExecutor(std::shared_ptr<ov::genai::LLMPipeline> pipe) {
        this->pipe = std::move(pipe);
    }

    bool hasRequests() {
        return (requests.size() > 0);
    }

    bool requestsQueueSize() {
        return requests.size();
    }

    void processRequest() {
        OVMS_PROFILE_FUNCTION();
        requests.front()->results = pipe->generate(requests.front()->inputIds, requests.front()->apiHandler->createGenerationConfig(), requests.front()->textStreamer);
        std::unique_lock<std::mutex> lock(queueMutex);
        requests.front()->readySignal.set_value();
        requests.pop();
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal) {
        std::unique_lock<std::mutex> lock(queueMutex);
        cv.wait(lock, [this, receivedEndSignal] { return (requests.size() > 0 || *receivedEndSignal); });
    }

    void addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
        std::lock_guard<std::mutex> guard(queueMutex);
        requests.push(request);
        cv.notify_one();
    }

    void notify() {
        std::unique_lock<std::mutex> lock(queueMutex);
        cv.notify_one();
    }
};

class LegacyExecutorWrapper {
    LegacyExecutor legacyExecutor;
    std::thread legacyExecutorThread;
    std::atomic<bool> finishExecutorThread = false;

    static void run(LegacyExecutor* legacyExecutor, std::atomic<bool>* receivedEndSignal) {
        // TODO add metrics
        while (!(*receivedEndSignal)) {
            try {
                if (legacyExecutor->hasRequests()) {
                    legacyExecutor->processRequest();
                } else {
                    SPDLOG_LOGGER_INFO(llm_executor_logger, "Awaiting requests: {};", legacyExecutor->requestsQueueSize());
                    legacyExecutor->waitForRequests(receivedEndSignal);
                }
            } catch (std::exception& e) {
                SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in LLM executor: {}.", e.what());
                exit(1);
            }
        }
    }

public:
    LegacyExecutorWrapper(std::shared_ptr<ov::genai::LLMPipeline> pipe) :
        legacyExecutor(std::move(pipe)) {
        legacyExecutorThread = std::thread(LegacyExecutorWrapper::run, &legacyExecutor, &finishExecutorThread);
    }

    ~LegacyExecutorWrapper() {
        finishExecutorThread = true;
        legacyExecutor.notify();
        legacyExecutorThread.join();
    }
    void addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
        legacyExecutor.addRequest(request);
    }
};
}  // namespace ovms
