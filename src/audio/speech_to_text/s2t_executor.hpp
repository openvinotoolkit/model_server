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

#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/genai/whisper_pipeline.hpp"
#include "src/executor_base.hpp"
#include "src/logging.hpp"

namespace ovms {

struct SttServableExecutionContext {
    std::vector<float> rawSpeech;
    ov::genai::WhisperGenerationConfig config;
    std::function<ov::genai::StreamingStatus(std::string)> streamerCallback;
    std::function<void()> onFinished;
    std::promise<ov::genai::WhisperDecodedResults> finishedPromise;
    std::future<ov::genai::WhisperDecodedResults> finished;

    SttServableExecutionContext(
        std::vector<float> rawSpeech,
        ov::genai::WhisperGenerationConfig config,
        std::function<ov::genai::StreamingStatus(std::string)> streamerCallback,
        std::function<void()> onFinished) :
        rawSpeech(std::move(rawSpeech)),
        config(std::move(config)),
        streamerCallback(std::move(streamerCallback)),
        onFinished(std::move(onFinished)),
        finished(finishedPromise.get_future()) {}
};

struct SttExecutor : public Executor<std::shared_ptr<SttServableExecutionContext>> {
    std::shared_ptr<ov::genai::WhisperPipeline> sttPipeline;
    std::mutex& sttPipelineMutex;

    SttExecutor(std::shared_ptr<ov::genai::WhisperPipeline> sttPipeline, std::mutex& sttPipelineMutex) :
        sttPipeline(std::move(sttPipeline)),
        sttPipelineMutex(sttPipelineMutex) {}

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
        try {
            std::unique_lock lock(sttPipelineMutex);
            auto result = sttPipeline->generate(
                requestExecutionContext->rawSpeech,
                requestExecutionContext->config,
                requestExecutionContext->streamerCallback);
            lock.unlock();
            requestExecutionContext->finishedPromise.set_value(std::move(result));
        } catch (...) {
            requestExecutionContext->finishedPromise.set_exception(std::current_exception());
        }
        requestExecutionContext->onFinished();
    }
};

class SttExecutorWrapper : public ExecutorWrapper<SttExecutor> {
public:
    SttExecutorWrapper(std::shared_ptr<ov::genai::WhisperPipeline> sttPipeline, std::mutex& sttPipelineMutex) :
        ExecutorWrapper(s2t_calculator_logger, std::make_shared<SttExecutor>(std::move(sttPipeline), sttPipelineMutex)) {}
};

}  // namespace ovms
