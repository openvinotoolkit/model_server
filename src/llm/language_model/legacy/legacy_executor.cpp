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

void LegacyExecutor::processRequest() {
    OVMS_PROFILE_FUNCTION();
    auto& requestExecutionContext = requests.front();
    if (requestExecutionContext->clientDisconnected) {
        requestExecutionContext->success = false;
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Client disconnected, skipping request processing.");
    } else {
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation started");
        try {
            requestExecutionContext->results = pipe->generate(requestExecutionContext->inputIds, requestExecutionContext->generationConfigBuilder->getConfig(), requestExecutionContext->textStreamer);
        } catch (std::exception& e) {
            requestExecutionContext->success = false;
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "LLM pipeline generation failed: {}.", e.what());
        }
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Generation ended");
    }
    requestExecutionContext->readySignal.set_value();
    requestExecutionContext->executionInProgress.notify_one();
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.pop();
}

void LegacyExecutor::addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.push(request);
    cv.notify_one();
}

LegacyExecutorWrapper::LegacyExecutorWrapper(std::shared_ptr<ov::genai::LLMPipeline> pipe) :
    ExecutorWrapper(llm_executor_logger, std::move(pipe)) {}

void LegacyExecutorWrapper::addRequest(std::shared_ptr<LegacyServableExecutionContext> request) {
    executor.addRequest(request);
}
}  // namespace ovms
