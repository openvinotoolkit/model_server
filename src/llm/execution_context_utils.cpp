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
#include "execution_context_utils.hpp"

#include <utility>

#include "../logging.hpp"
#include "servable.hpp"

namespace ovms {

Status initializeLlmExecutionContexts(const GenAiServableMap& servableMap, GenAiExecutionContextMap& executionContextMap) {
    for (const auto& [nodeName, servable] : servableMap) {
        auto it = executionContextMap.find(nodeName);
        if (it == executionContextMap.end() || !it->second) {
            SPDLOG_DEBUG("Missing LLM execution context holder for node: {}", nodeName);
            return StatusCode::INTERNAL_ERROR;
        }
        auto& holder = it->second;
        auto ctx = servable->createExecutionContext();
        if (!ctx) {
            SPDLOG_DEBUG("Failed to create LLM execution context for node: {}", nodeName);
            return StatusCode::INTERNAL_ERROR;
        }
        holder->set(std::move(ctx));
    }
    return StatusCode::OK;
}

void resetLlmExecutionContexts(GenAiExecutionContextMap& executionContextMap) {
    for (auto& [_, holder] : executionContextMap) {
        if (!holder) {
            continue;
        }
        holder->reset();
    }
}

}  // namespace ovms
