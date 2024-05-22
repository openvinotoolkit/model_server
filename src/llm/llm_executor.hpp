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
namespace ovms {
class LLMExecutor {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> finishExecutorThread;
    std::shared_ptr<ContinuousBatchingPipeline> pipe;
    std::thread llmExecutorThread;

    static void run(std::shared_ptr<ContinuousBatchingPipeline> pipe, std::atomic<bool>* finishExecutorThread) {
        while (!(*finishExecutorThread)) {
            if (pipe->has_awaiting_requests() || pipe->has_running_requests()) {
                pipe->step();
            }
            else {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]{return !(pipe->has_awaiting_requests() || pipe->has_running_requests());});
            }
        }
    }

public:
    LLMExecutor(std::shared_ptr<ContinuousBatchingPipeline> pipe) :
        finishExecutorThread(false),
        pipe(pipe) {
        llmExecutorThread = std::thread(LLMExecutor::run, pipe, &finishExecutorThread);
    }

    ~LLMExecutor() {
        finishExecutorThread = true;
        llmExecutorThread.join();
    }

    void notifyNewRequestArrived() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_one();
    }
};
}  // namespace ovms
