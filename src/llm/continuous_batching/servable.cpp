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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../logging.hpp"
#include "../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../http_payload.hpp"
#include "../../mediapipe_internal/mediapipe_utils.hpp"
#include "../apis/openai_completions.hpp"
#include "../text_processor.hpp"
#include "llm_executor.hpp"
#include "servable.hpp"

namespace ovms {

void ContinuousBatchingServable::notifyExecutorThread() {
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Notifying executor thread");
    if (properties->llmExecutorWrapper == nullptr) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "LLMExecutorWrapper is not initialized");
        return;
    }
    properties->llmExecutorWrapper->notifyNewRequestArrived();
}

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> ContinuousBatchingServable::createExecutionContext() {
    return std::make_shared<ContinuousBatchingServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> ContinuousBatchingServable::getProperties() {
    return properties;
}

absl::Status ContinuousBatchingServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    cbExecutionContext->generationHandle = properties->pipeline->add_request(currentRequestId++,  // to be removed from API?
        cbExecutionContext->inputIds,
        cbExecutionContext->apiHandler->createGenerationConfig());

    cbExecutionContext->payload.client->registerDisconnectionCallback([genHandle = cbExecutionContext->generationHandle]() {
        genHandle->stop();
    });
    notifyExecutorThread();

    return absl::OkStatus();
}

absl::Status ContinuousBatchingServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    cbExecutionContext->generationOutputs = cbExecutionContext->generationHandle->read_all();
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::STOP) {
        return absl::CancelledError();
    }
    RET_CHECK(cbExecutionContext->generationOutputs.size() >= 1);
    return absl::OkStatus();
}

absl::Status ContinuousBatchingServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Relevant properties read");
    // Streaming scenario
    // Each iteration is single execution of Process() method in the calculator
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::STOP) {
        return absl::CancelledError();
    }

    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::RUNNING || cbExecutionContext->generationHandle->can_read()) {
        // Subsequent iteration
        OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
        ov::genai::GenerationOutputs generationOutputs = cbExecutionContext->generationHandle->read();
        RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
        cbExecutionContext->generationOutputs = {generationOutputs.begin()->second};
    }
    return absl::OkStatus();
}

}  // namespace ovms
