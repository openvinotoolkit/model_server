//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "runtime_chat_template_runtime_loader.hpp"

#include <cstdlib>
#include <mutex>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../logging.hpp"

namespace ovms {
namespace {

using EnsurePythonInterpreterInitializedFn = bool (*)(const char** errorMessage);
using CreatePreparedFn = bool (*)(const char*, const char*, const char*, const char*, void**, const char**);
using ApplyPreparedFn = bool (*)(void*, const char*, const char**);
using DestroyPreparedFn = void (*)(void*);

struct RuntimeChatTemplateRuntimeState {
    bool available = false;
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    EnsurePythonInterpreterInitializedFn ensurePythonInterpreterInitialized = nullptr;
    CreatePreparedFn createPrepared = nullptr;
    ApplyPreparedFn applyPrepared = nullptr;
    DestroyPreparedFn destroyPrepared = nullptr;
};

RuntimeChatTemplateRuntimeState& runtimeState() {
    static RuntimeChatTemplateRuntimeState state;
    return state;
}

std::once_flag& runtimeInitFlag() {
    static std::once_flag flag;
    return flag;
}

#if defined(_WIN32)
void* resolveSymbol(HMODULE handle, const char* symbolName) {
    return reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
}
#else
void* resolveSymbol(void* handle, const char* symbolName) {
    return dlsym(handle, symbolName);
}
#endif

void initializeRuntimeState() {
    auto& state = runtimeState();
#if defined(_WIN32)
    state.handle = LoadLibraryA("libovmspython.dll");
    if (state.handle == nullptr) {
        SPDLOG_DEBUG("Python runtime library libovmspython.dll not available for runtime chat template processing");
#else
    state.handle = dlopen("libovmspython.so", RTLD_NOW | RTLD_LOCAL);
    if (state.handle == nullptr) {
        SPDLOG_DEBUG("Python runtime library libovmspython.so not available for runtime chat template processing");
#endif
        return;
    }

    state.ensurePythonInterpreterInitialized = reinterpret_cast<EnsurePythonInterpreterInitializedFn>(
        resolveSymbol(state.handle, "OVMS_ensurePythonInterpreterInitialized"));
    state.createPrepared = reinterpret_cast<CreatePreparedFn>(resolveSymbol(state.handle, "OVMS_createPreparedChatTemplateRuntime"));
    state.applyPrepared = reinterpret_cast<ApplyPreparedFn>(resolveSymbol(state.handle, "OVMS_applyPreparedChatTemplateRuntime"));
    state.destroyPrepared = reinterpret_cast<DestroyPreparedFn>(resolveSymbol(state.handle, "OVMS_destroyPreparedChatTemplateRuntime"));

    if (state.ensurePythonInterpreterInitialized == nullptr) {
        SPDLOG_WARN("Python runtime library is missing OVMS_ensurePythonInterpreterInitialized");
        return;
    }

    state.available = true;
}

}  // namespace

bool ensurePythonRuntimeInitialized(std::string& errorMessage) {
    std::call_once(runtimeInitFlag(), initializeRuntimeState);
    auto& state = runtimeState();
    if (!state.available) {
        errorMessage = "Python runtime library or interpreter initialization symbol is unavailable";
        return false;
    }
    const char* runtimeError = nullptr;
    if (!state.ensurePythonInterpreterInitialized(&runtimeError)) {
        errorMessage = runtimeError != nullptr ? runtimeError : "Python interpreter initialization failed";
        return false;
    }
    errorMessage.clear();
    return true;
}

PreparedChatTemplateRuntime::~PreparedChatTemplateRuntime() {
    if (handle != nullptr) {
        destroy(handle);
    }
}

bool PreparedChatTemplateRuntime::prepare(const std::string& modelsPath, const std::string& chatTemplate,
    const std::string& bosToken, const std::string& eosToken, std::string& errorMessage) {
    std::call_once(runtimeInitFlag(), initializeRuntimeState);
    auto& state = runtimeState();
    if (!state.available || state.createPrepared == nullptr || state.applyPrepared == nullptr || state.destroyPrepared == nullptr) {
        errorMessage = "Python chat template runtime is unavailable";
        return false;
    }
    const char* runtimeOutput = nullptr;
    void* newHandle = nullptr;
    if (!state.createPrepared(modelsPath.c_str(), chatTemplate.c_str(), bosToken.c_str(), eosToken.c_str(), &newHandle, &runtimeOutput)) {
        errorMessage = runtimeOutput != nullptr ? runtimeOutput : "Failed to prepare Python chat template";
        return false;
    }
    if (newHandle == nullptr) {
        errorMessage = "Python chat template runtime returned no prepared template";
        return false;
    }
    if (handle != nullptr) {
        destroy(handle);
    }
    handle = newHandle;
    destroy = state.destroyPrepared;
    render = state.applyPrepared;
    errorMessage.clear();
    return true;
}

bool PreparedChatTemplateRuntime::apply(const std::string& requestBody, std::string& output) const {
    if (handle == nullptr) {
        output = "Python chat template is not prepared";
        return false;
    }
    const char* runtimeOutput = nullptr;
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Executing Python jinja chat template processor to apply chat template");
    const bool success = render(handle, requestBody.c_str(), &runtimeOutput);
    output = runtimeOutput != nullptr ? runtimeOutput : "Python chat template rendering failed";
    return success;
}

}  // namespace ovms
