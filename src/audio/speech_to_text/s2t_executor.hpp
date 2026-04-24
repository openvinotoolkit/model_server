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
#include <mutex>
#include <stdexcept>

#include "openvino/genai/whisper_pipeline.hpp"
#include "src/executor_base.hpp"
#include "src/logging.hpp"

namespace ovms {

using SttStreamingJob = std::packaged_task<ov::genai::WhisperDecodedResults()>;

struct SttExecutor : public Executor<SttStreamingJob> {
    void processRequest() {
        SttStreamingJob request;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (requests.empty()) {
                return;
            }
            request = std::move(requests.front());
            requests.pop();
        }
        request();
    }

    std::future<ov::genai::WhisperDecodedResults> addRequest(SttStreamingJob&& request) {
        auto resultFuture = request.get_future();
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            requests.push(std::move(request));
        }
        cv.notify_one();
        return resultFuture;
    }
};

class SttExecutorWrapper : public ExecutorWrapper<SttExecutor> {
public:
    SttExecutorWrapper() :
        ExecutorWrapper(s2t_calculator_logger) {}

    std::future<ov::genai::WhisperDecodedResults> addRequest(SttStreamingJob&& request) {
        if (finishExecutorThread) {
            throw std::runtime_error("Cannot schedule STT streaming job - executor is stopping");
        }
        return executor.addRequest(std::move(request));
    }
};

}  // namespace ovms
