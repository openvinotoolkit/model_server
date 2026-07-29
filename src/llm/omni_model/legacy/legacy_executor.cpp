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

#include "legacy_executor.hpp"

#include <string>
#include <utility>
#include <vector>

#include "servable.hpp"

namespace ovms {
OmniModelLegacyExecutor::OmniModelLegacyExecutor(std::shared_ptr<ov::genai::OmniPipeline> pipe) {
    this->pipe = std::move(pipe);
}

bool OmniModelLegacyExecutor::hasRequests() {
    return (requests.size() > 0);
}

size_t OmniModelLegacyExecutor::requestsQueueSize() {
    return requests.size();
}

void OmniModelLegacyExecutor::processRequest() {
    OVMS_PROFILE_FUNCTION();
    auto& requestExecutionContext = requests.front();
    if (requestExecutionContext->clientDisconnected) {
        requestExecutionContext->success = false;
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Client disconnected, skipping request processing.");
    } else {
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni generation started");
        try {
            std::vector<ov::genai::VideoMetadata> videosMetadata;
            requestExecutionContext->results = pipe->generate(
                requestExecutionContext->inputRequest.promptText,
                requestExecutionContext->inputRequest.inputImages,
                requestExecutionContext->inputRequest.inputVideos,
                videosMetadata,
                requestExecutionContext->inputRequest.inputAudios,
                requestExecutionContext->inputRequest.generationConfig,
                requestExecutionContext->speechConfig,
                requestExecutionContext->textStreamer,
                requestExecutionContext->speechStreamer);
        } catch (std::exception& e) {
            requestExecutionContext->success = false;
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Omni pipeline generation failed: {}.", e.what());
        }
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni generation ended");
    }
    requestExecutionContext->readySignal.set_value();
    requestExecutionContext->deltaChannel.signalComplete();
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.pop();
}

void OmniModelLegacyExecutor::waitForRequests(std::atomic<bool>* receivedEndSignal) {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.wait(lock, [this, receivedEndSignal] { return (requests.size() > 0 || *receivedEndSignal); });
}

void OmniModelLegacyExecutor::addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request) {
    std::lock_guard<std::mutex> guard(queueMutex);
    requests.push(request);
    cv.notify_one();
}

void OmniModelLegacyExecutor::notify() {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.notify_one();
}

void OmniModelLegacyExecutorWrapper::run(OmniModelLegacyExecutor* executor, std::atomic<bool>* receivedEndSignal) {
    while (!(*receivedEndSignal)) {
        try {
            SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni executor all requests: {};", executor->requestsQueueSize());
            if (executor->hasRequests()) {
                executor->processRequest();
            } else {
                executor->waitForRequests(receivedEndSignal);
            }
        } catch (std::exception& e) {
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in Omni executor: {}.", e.what());
            exit(1);
        }
    }
}

OmniModelLegacyExecutorWrapper::OmniModelLegacyExecutorWrapper(std::shared_ptr<ov::genai::OmniPipeline> pipe) :
    executor(std::move(pipe)) {
    executorThread = std::thread(OmniModelLegacyExecutorWrapper::run, &executor, &finishExecutorThread);
}

OmniModelLegacyExecutorWrapper::~OmniModelLegacyExecutorWrapper() {
    finishExecutorThread = true;
    executor.notify();
    executorThread.join();
}

void OmniModelLegacyExecutorWrapper::addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request) {
    executor.addRequest(request);
}
}  // namespace ovms
