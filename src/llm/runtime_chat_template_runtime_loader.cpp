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
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../logging.hpp"

namespace ovms {
namespace {

struct RuntimeChatTemplateRuntimeState {
    bool available = false;
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    RuntimeChatTemplateRuntimeApi api;
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

bool tryLoadLibrary(const std::vector<std::string>& candidates, HMODULE& handleOut) {
    for (const auto& candidate : candidates) {
        const auto handle = LoadLibraryA(candidate.c_str());
        if (handle != nullptr) {
            handleOut = handle;
            return true;
        }
    }
    return false;
}
#else
void* resolveSymbol(void* handle, const char* symbolName) {
    return dlsym(handle, symbolName);
}

bool tryLoadLibrary(const std::vector<std::string>& candidates, void*& handleOut) {
    for (const auto& candidate : candidates) {
        const auto handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            handleOut = handle;
            return true;
        }
    }
    return false;
}
#endif

std::vector<std::string> buildCandidates() {
    std::vector<std::string> candidates;
#if defined(_WIN32)
    candidates.insert(candidates.end(), {
                                            "libovmspython.dll",
                                            "./libovmspython.dll",
                                            "src/python/libovmspython.dll",
                                            "./src/python/libovmspython.dll",
                                            "bazel-bin/src/libovmspython.dll",
                                            "./bazel-bin/src/libovmspython.dll",
                                            "bazel-bin/src/python/libovmspython.dll",
                                            "./bazel-bin/src/python/libovmspython.dll",
                                        });
#else
    candidates.insert(candidates.end(), {
                                            "libovmspython.so",
                                            "./libovmspython.so",
                                            "src/python/libovmspython.so",
                                            "./src/python/libovmspython.so",
                                            "bazel-bin/src/python/libovmspython.so",
                                            "./bazel-bin/src/python/libovmspython.so",
                                        });
#endif
    return candidates;
}

void initializeRuntimeState() {
    auto& state = runtimeState();
    const auto candidates = buildCandidates();
    if (!tryLoadLibrary(candidates, state.handle)) {
#if defined(_WIN32)
        SPDLOG_DEBUG("Python runtime library libovmspython.dll not available for runtime chat template processing");
#else
        SPDLOG_DEBUG("Python runtime library libovmspython.so not available for runtime chat template processing");
#endif
        return;
    }

    state.api.createPreparedFn = reinterpret_cast<RuntimeChatTemplateRuntimeApi::CreatePreparedChatTemplateRuntimeFn>(
        resolveSymbol(state.handle, "OVMS_createPreparedChatTemplateRuntime"));
    state.api.applyPreparedFn = reinterpret_cast<RuntimeChatTemplateRuntimeApi::ApplyPreparedChatTemplateRuntimeFn>(
        resolveSymbol(state.handle, "OVMS_applyPreparedChatTemplateRuntime"));
    state.api.destroyPreparedFn = reinterpret_cast<RuntimeChatTemplateRuntimeApi::DestroyPreparedChatTemplateRuntimeFn>(
        resolveSymbol(state.handle, "OVMS_destroyPreparedChatTemplateRuntime"));

    if (state.api.createPreparedFn == nullptr ||
        state.api.applyPreparedFn == nullptr ||
        state.api.destroyPreparedFn == nullptr) {
        SPDLOG_WARN("Python runtime library missing prepared chat template runtime symbols");
        return;
    }

    state.available = true;
}

}  // namespace

const RuntimeChatTemplateRuntimeApi* getRuntimeChatTemplateRuntimeApi() {
    std::call_once(runtimeInitFlag(), initializeRuntimeState);
    auto& state = runtimeState();
    if (!state.available) {
        return nullptr;
    }
    return &state.api;
}

}  // namespace ovms
