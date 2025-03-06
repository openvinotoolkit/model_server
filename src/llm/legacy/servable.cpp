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
#include "servable.hpp"

namespace ovms {

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> LegacyServable::createExecutionContext() {
    return std::make_shared<LegacyServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> LegacyServable::getProperties() {
    return properties;
}

absl::Status LegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return absl::OkStatus();
}

absl::Status LegacyServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::lock_guard<std::mutex> guard(properties->pipelineMutex);
    legacyExecutionContext->results = properties->pipeline->generate(legacyExecutionContext->inputIds, legacyExecutionContext->apiHandler->createGenerationConfig());
    return absl::OkStatus();
}

absl::Status LegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return absl::OkStatus();
}

absl::Status LegacyServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(legacyExecutionContext->results);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

// absl::Status LegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
//     auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
//     if (legacyExecutionContext->payload.client->isDisconnected()) {
//         return absl::CancelledError();
//     }
//     if(!legacyExecutionContext->streamerInitialized) {
//         auto streamer = [&legacyExecutionContext](int64_t token) {
//             std::lock_guard<std::mutex> guard(legacyExecutionContext->tokensMutex);
//             legacyExecutionContext->tokens.push_back(token);
//             return ov::genai::StreamingStatus::RUNNING;
//         };
//         legacyExecutionContext->streamerInitialized = true;
//         std::lock_guard<std::mutex> guard(properties->pipelineMutex);
//         legacyExecutionContext->results = properties->pipeline->generate(legacyExecutionContext->inputIds, legacyExecutionContext->apiHandler->createGenerationConfig(), streamer);
//     }
//     return absl::OkStatus();
// }
}  // namespace ovms
