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

#include <future>
#include <memory>
#include <mutex>
#include <utility>

#include "openvino/genai/whisper_pipeline.hpp"
#include "src/executor_base.hpp"
#include "src/logging.hpp"

namespace ovms {

using SttStreamingJob = std::packaged_task<ov::genai::WhisperDecodedResults()>;

struct SttServableExecutionContext {
    SttStreamingJob job;
    std::future<ov::genai::WhisperDecodedResults> finished;

    explicit SttServableExecutionContext(SttStreamingJob&& streamingJob) :
        job(std::move(streamingJob)),
        finished(job.get_future()) {}
};

struct SttExecutor : public Executor<std::shared_ptr<SttServableExecutionContext>> {
    void processRequest() {
        std::shared_ptr<SttServableExecutionContext> requestExecutionContext;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (requests.empty()) {
                return;
            }
            requestExecutionContext = std::move(requests.front());
            requests.pop();
        }
        requestExecutionContext->job();
    }
};

class SttExecutorWrapper : public ExecutorWrapper<SttExecutor> {
public:
    SttExecutorWrapper() :
        ExecutorWrapper(s2t_calculator_logger) {}
};

}  // namespace ovms
