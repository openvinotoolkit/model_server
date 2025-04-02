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

#include <openvino/genai/visual_language/pipeline.hpp>

#include "../../../logging.hpp"
#include "../../../profiler.hpp"

#include <chrono>
#include <future>

namespace ovms {
struct VisualLanguageModelLegacyServableExecutionContext;

struct VisualLanguageModelLegacyExecutor {
    std::condition_variable cv;
    std::queue<std::shared_ptr<VisualLanguageModelLegacyServableExecutionContext>> requests;
    std::mutex queueMutex;
    std::shared_ptr<ov::genai::VLMPipeline> pipe;

    VisualLanguageModelLegacyExecutor(std::shared_ptr<ov::genai::VLMPipeline> pipe);

    bool hasRequests();

    bool requestsQueueSize();
    void processRequest();

    void waitForRequests(std::atomic<bool>* receivedEndSignal);

    void addRequest(std::shared_ptr<VisualLanguageModelLegacyServableExecutionContext> request);

    void notify();
};

class VisualLanguageModelLegacyExecutorWrapper {
    VisualLanguageModelLegacyExecutor executor;
    std::thread executorThread;
    std::atomic<bool> finishExecutorThread = false;

    static void run(VisualLanguageModelLegacyExecutor* executor, std::atomic<bool>* receivedEndSignal);

public:
    VisualLanguageModelLegacyExecutorWrapper(std::shared_ptr<ov::genai::VLMPipeline> pipe);
    ~VisualLanguageModelLegacyExecutorWrapper();
    void addRequest(std::shared_ptr<VisualLanguageModelLegacyServableExecutionContext> request);
};
}  // namespace ovms
