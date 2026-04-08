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

#include "legacy_executor.hpp"
#include "servable.hpp"
#include <vector>

namespace ovms {
VisualLanguageModelLegacyExecutor::VisualLanguageModelLegacyExecutor(std::shared_ptr<ov::genai::VLMPipeline> pipe) {
    this->pipe = std::move(pipe);
}

bool VisualLanguageModelLegacyExecutor::hasRequests() {
    return (requests.size() > 0);
}

size_t VisualLanguageModelLegacyExecutor::requestsQueueSize() {
    return requests.size();
}

void VisualLanguageModelLegacyExecutor::processRequest() {
    OVMS_PROFILE_FUNCTION();
    auto& requestExecutionContext = requests.front();
    if (requestExecutionContext->clientDisconnected) {
        requestExecutionContext->success = false;
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Client disconnected, skipping request processing.");
    } else {
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation started");
        try {
            requestExecutionContext->results = pipe->generate(requestExecutionContext->inputText, requestExecutionContext->inputImages, requestExecutionContext->generationConfigBuilder->getConfig(), requestExecutionContext->textStreamer);
        } catch (std::exception& e) {
            requestExecutionContext->success = false;
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "VLM pipeline generation failed: {}.", e.what());
        }
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation ended");
    }
    requestExecutionContext->readySignal.set_value();
    requestExecutionContext->executionInProgress.notify_one();
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.pop();
}

void VisualLanguageModelLegacyExecutor::waitForRequests(std::atomic<bool>* receivedEndSignal) {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.wait(lock, [this, receivedEndSignal] { return (requests.size() > 0 || *receivedEndSignal); });
}

void VisualLanguageModelLegacyExecutor::addRequest(std::shared_ptr<VisualLanguageModelLegacyServableExecutionContext> request) {
    std::lock_guard<std::mutex> guard(queueMutex);
    requests.push(request);
    cv.notify_one();
}

void VisualLanguageModelLegacyExecutor::notify() {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.notify_one();
}

void VisualLanguageModelLegacyExecutorWrapper::run(VisualLanguageModelLegacyExecutor* executor, std::atomic<bool>* receivedEndSignal) {
    // TODO add metrics
    while (!(*receivedEndSignal)) {
        try {
            SPDLOG_LOGGER_INFO(llm_executor_logger, "All requests: {};", executor->requestsQueueSize());
            if (executor->hasRequests()) {
                executor->processRequest();
            } else {
                executor->waitForRequests(receivedEndSignal);
            }
        } catch (std::exception& e) {
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in LLM executor: {}.", e.what());
            exit(1);
        }
    }
}

VisualLanguageModelLegacyExecutorWrapper::VisualLanguageModelLegacyExecutorWrapper(std::shared_ptr<ov::genai::VLMPipeline> pipe) :
    executor(std::move(pipe)) {
    executorThread = std::thread(VisualLanguageModelLegacyExecutorWrapper::run, &executor, &finishExecutorThread);
}

VisualLanguageModelLegacyExecutorWrapper::~VisualLanguageModelLegacyExecutorWrapper() {
    finishExecutorThread = true;
    executor.notify();
    executorThread.join();
}
void VisualLanguageModelLegacyExecutorWrapper::addRequest(std::shared_ptr<VisualLanguageModelLegacyServableExecutionContext> request) {
    executor.addRequest(request);
}
}  // namespace ovms
