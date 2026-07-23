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
#include <thread>

#include <openvino/genai/omni/pipeline.hpp>

#include "../../../logging.hpp"
#include "../../../profiler.hpp"

namespace ovms {
struct OmniModelLegacyServableExecutionContext;

struct OmniModelLegacyExecutor {
    std::condition_variable cv;
    std::queue<std::shared_ptr<OmniModelLegacyServableExecutionContext>> requests;
    std::mutex queueMutex;
    std::shared_ptr<ov::genai::OmniPipeline> pipe;

    OmniModelLegacyExecutor(std::shared_ptr<ov::genai::OmniPipeline> pipe);

    bool hasRequests();
    size_t requestsQueueSize();
    void processRequest();
    void waitForRequests(std::atomic<bool>* receivedEndSignal);
    void addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request);
    void notify();
};

class OmniModelLegacyExecutorWrapper {
    OmniModelLegacyExecutor executor;
    std::thread executorThread;
    std::atomic<bool> finishExecutorThread = false;

    static void run(OmniModelLegacyExecutor* executor, std::atomic<bool>* receivedEndSignal);

public:
    OmniModelLegacyExecutorWrapper(std::shared_ptr<ov::genai::OmniPipeline> pipe);
    ~OmniModelLegacyExecutorWrapper();

    OmniModelLegacyExecutorWrapper(const OmniModelLegacyExecutorWrapper&) = delete;
    OmniModelLegacyExecutorWrapper& operator=(const OmniModelLegacyExecutorWrapper&) = delete;

    void addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request);
};
}  // namespace ovms
