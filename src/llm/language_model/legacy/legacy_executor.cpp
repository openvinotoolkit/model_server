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

namespace ovms {
LegacyExecutor::LegacyExecutor(std::shared_ptr<ov::genai::LLMPipeline> pipe) {
    this->pipe = std::move(pipe);
}

bool LegacyExecutor::hasRequests() {
    return requests.size() > 0;
}

bool LegacyExecutor::requestsQueueSize() {
    return requests.size();
}

void LegacyExecutor::processRequest() {
    OVMS_PROFILE_FUNCTION();
    SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation started");
    try {
        pipe->start_chat();
        requests.front()->results = pipe->generate(requests.front()->inputIds, requests.front()->apiHandler->createGenerationConfig(), requests.front()->textStreamer);
        pipe->finish_chat();
    } catch (std::exception& e) {
        requests.front()->success = false;
        SPDLOG_LOGGER_ERROR(llm_executor_logger, "LLM pipeline generation failed: {}.", e.what());
    }
    SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation ended");
    requests.front()->readySignal.set_value();
    requests.front()->executionInProgress.notify_one();
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.pop();
}

void LegacyExecutor::waitForRequests(std::atomic<bool>* receivedEndSignal) {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.wait(lock, [this, receivedEndSignal] { return (requests.size() > 0 || *receivedEndSignal); });
}

void LegacyExecutor::addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.push(request);
    cv.notify_one();
}

void LegacyExecutor::notify() {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.notify_one();
}

void LegacyExecutorWrapper::run(LegacyExecutor* legacyExecutor, std::atomic<bool>* receivedEndSignal) {
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

LegacyExecutorWrapper::LegacyExecutorWrapper(std::shared_ptr<ov::genai::LLMPipeline> pipe) :
    legacyExecutor(std::move(pipe)) {
    legacyExecutorThread = std::thread(LegacyExecutorWrapper::run, &legacyExecutor, &finishExecutorThread);
}

LegacyExecutorWrapper::~LegacyExecutorWrapper() {
    finishExecutorThread = true;
    legacyExecutor.notify();
    legacyExecutorThread.join();
}
void LegacyExecutorWrapper::addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
    legacyExecutor.addRequest(request);
}
}  // namespace ovms
