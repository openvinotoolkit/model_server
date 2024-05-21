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
#include <memory>
#include <thread>

#include <continuous_batching_pipeline.hpp>
namespace ovms {
class LLMExecutor {
    std::atomic<bool> end;
    std::shared_ptr<ContinuousBatchingPipeline> pipe;
    std::thread llmExecutorThread;

    static void run(std::shared_ptr<ContinuousBatchingPipeline> pipe, std::atomic<bool>* end) {
        while (!(*end)) {
            pipe->step();
        }
    }

public:
    LLMExecutor(std::shared_ptr<ContinuousBatchingPipeline> pipe) :
        end(false),
        pipe(pipe) {
        llmExecutorThread = std::thread(LLMExecutor::run, pipe, &end);
    }

    ~LLMExecutor() {
        end = true;
        llmExecutorThread.join();
    }
};
}  // namespace ovms
