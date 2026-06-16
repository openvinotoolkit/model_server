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

#include "runtime_chat_template.hpp"

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

using ApplyChatTemplateRuntimeFn = bool (*)(
    const char* modelsPath,
    const char* requestBody,
    const char* chatTemplate,
    const char* bosToken,
    const char* eosToken,
    const char** output);

struct RuntimeApi {
    bool initialized = false;
    bool available = false;
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    ApplyChatTemplateRuntimeFn applyFn = nullptr;
};

RuntimeApi& runtimeApi() {
    static RuntimeApi api;
    return api;
}

std::mutex& runtimeApiMutex() {
    static std::mutex m;
    return m;
}

void initializeRuntimeApiLocked() {
    auto& api = runtimeApi();
    if (api.initialized) {
        return;
    }

    api.initialized = true;
#if defined(_WIN32)
    const std::vector<std::string> candidates = {
        "libovmspython.dll",
        "./libovmspython.dll",
        "src/python/libovmspython.dll",
        "./src/python/libovmspython.dll",
        "bazel-bin/src/python/libovmspython.dll",
        "./bazel-bin/src/python/libovmspython.dll",
    };

    for (const auto& candidate : candidates) {
        api.handle = LoadLibraryA(candidate.c_str());
        if (api.handle != nullptr) {
            break;
        }
    }

    if (api.handle == nullptr) {
        SPDLOG_DEBUG("Python runtime library libovmspython.dll not available for runtime chat template processing");
        return;
    }

    api.applyFn = reinterpret_cast<ApplyChatTemplateRuntimeFn>(
        GetProcAddress(api.handle, "OVMS_applyChatTemplateRuntime"));
#else
    const std::vector<std::string> candidates = {
        "libovmspython.so",
        "./libovmspython.so",
        "src/python/libovmspython.so",
        "./src/python/libovmspython.so",
        "bazel-bin/src/python/libovmspython.so",
        "./bazel-bin/src/python/libovmspython.so",
    };

    for (const auto& candidate : candidates) {
        api.handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (api.handle != nullptr) {
            break;
        }
    }

    if (api.handle == nullptr) {
        SPDLOG_DEBUG("Python runtime library libovmspython.so not available for runtime chat template processing");
        return;
    }

    api.applyFn = reinterpret_cast<ApplyChatTemplateRuntimeFn>(
        dlsym(api.handle, "OVMS_applyChatTemplateRuntime"));
#endif

    if (api.applyFn == nullptr) {
        SPDLOG_WARN("Python runtime library missing symbol OVMS_applyChatTemplateRuntime");
        return;
    }

    api.available = true;
}

}  // namespace

RuntimeChatTemplateStatus tryApplyChatTemplateRuntime(
    const std::string& modelsPath,
    const std::string& requestBody,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    std::string& output) {
    std::lock_guard<std::mutex> lock(runtimeApiMutex());
    initializeRuntimeApiLocked();

    auto& api = runtimeApi();
    if (!api.available || api.applyFn == nullptr) {
        return RuntimeChatTemplateStatus::UNAVAILABLE;
    }

    const char* runtimeOutput = nullptr;
    bool ok = api.applyFn(
        modelsPath.c_str(),
        requestBody.c_str(),
        chatTemplate.c_str(),
        bosToken.c_str(),
        eosToken.c_str(),
        &runtimeOutput);

    if (runtimeOutput != nullptr) {
        output = runtimeOutput;
    } else {
        output.clear();
    }

    if (!ok) {
        if (output.empty()) {
            output = "Failed to apply chat template using runtime Python Jinja";
        }
        return RuntimeChatTemplateStatus::ERROR;
    }

    return RuntimeChatTemplateStatus::APPLIED;
}

}  // namespace ovms
