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

#ifdef __linux__
#include <dlfcn.h>
#elif _WIN32
#include <windows.h>
#endif

#include "../logging.hpp"

namespace ovms {
namespace {

using CreateExecutionContextFn = int (*)(void*, std::shared_ptr<GenAiServableExecutionContext>*);

CreateExecutionContextFn resolveCreateExecutionContextFn() {
#ifdef __linux__
    return reinterpret_cast<CreateExecutionContextFn>(dlsym(RTLD_DEFAULT, "OVMS_LLMCreateExecutionContext"));
#elif _WIN32
    HMODULE currentModule = GetModuleHandleA(nullptr);
    if (currentModule == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<CreateExecutionContextFn>(GetProcAddress(currentModule, "OVMS_LLMCreateExecutionContext"));
#else
    return nullptr;
#endif
}

}  // namespace

Status initializeLlmExecutionContexts(const GenAiServableMap& servableMap, GenAiExecutionContextMap& executionContextMap) {
    if (servableMap.empty()) {
        return StatusCode::OK;
    }
    auto* createExecutionContext = resolveCreateExecutionContextFn();
    if (createExecutionContext == nullptr) {
        SPDLOG_DEBUG("LLM execution context provider is unavailable in Python-free MediaPipe runtime");
        return StatusCode::INTERNAL_ERROR;
    }

    for (const auto& [nodeName, servable] : servableMap) {
        auto it = executionContextMap.find(nodeName);
        if (it == executionContextMap.end() || !it->second) {
            SPDLOG_DEBUG("Missing LLM execution context holder for node: {}", nodeName);
            return StatusCode::INTERNAL_ERROR;
        }
        std::shared_ptr<GenAiServableExecutionContext> ctx;
        if (createExecutionContext(servable.get(), &ctx) != 0 || !ctx) {
            SPDLOG_DEBUG("Failed to create LLM execution context for node: {}", nodeName);
            return StatusCode::INTERNAL_ERROR;
        }
        it->second->set(std::move(ctx));
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