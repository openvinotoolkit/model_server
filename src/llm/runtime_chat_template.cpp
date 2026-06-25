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

#include <string>
#include <mutex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../logging.hpp"

namespace ovms {
namespace {

using CreatePreparedChatTemplateRuntimeFn = bool (*) (
    const char* modelsPath,
    const char* chatTemplate,
    const char* bosToken,
    const char* eosToken,
    void** preparedHandle,
    const char** output);

using ApplyPreparedChatTemplateRuntimeFn = bool (*) (
    void* preparedHandle,
    const char* requestBody,
    const char** output);

using DestroyPreparedChatTemplateRuntimeFn = void (*)(void* preparedHandle);

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
    CreatePreparedChatTemplateRuntimeFn createPreparedFn = nullptr;
    ApplyPreparedChatTemplateRuntimeFn applyPreparedFn = nullptr;
    DestroyPreparedChatTemplateRuntimeFn destroyPreparedFn = nullptr;
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
    api.createPreparedFn = reinterpret_cast<CreatePreparedChatTemplateRuntimeFn>(
        GetProcAddress(api.handle, "OVMS_createPreparedChatTemplateRuntime"));
    api.applyPreparedFn = reinterpret_cast<ApplyPreparedChatTemplateRuntimeFn>(
        GetProcAddress(api.handle, "OVMS_applyPreparedChatTemplateRuntime"));
    api.destroyPreparedFn = reinterpret_cast<DestroyPreparedChatTemplateRuntimeFn>(
        GetProcAddress(api.handle, "OVMS_destroyPreparedChatTemplateRuntime"));
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
    api.createPreparedFn = reinterpret_cast<CreatePreparedChatTemplateRuntimeFn>(
        dlsym(api.handle, "OVMS_createPreparedChatTemplateRuntime"));
    api.applyPreparedFn = reinterpret_cast<ApplyPreparedChatTemplateRuntimeFn>(
        dlsym(api.handle, "OVMS_applyPreparedChatTemplateRuntime"));
    api.destroyPreparedFn = reinterpret_cast<DestroyPreparedChatTemplateRuntimeFn>(
        dlsym(api.handle, "OVMS_destroyPreparedChatTemplateRuntime"));
#endif

    if (api.applyFn == nullptr || api.createPreparedFn == nullptr || api.applyPreparedFn == nullptr || api.destroyPreparedFn == nullptr) {
        SPDLOG_WARN("Python runtime library missing chat template runtime symbols");
        return;
    }

    api.available = true;
}

}  // namespace

PreparedRuntimeChatTemplate::PreparedRuntimeChatTemplate(PreparedRuntimeChatTemplate&& other) noexcept :
    handle(std::exchange(other.handle, nullptr)),
    destroyFn(std::exchange(other.destroyFn, nullptr)) {}

PreparedRuntimeChatTemplate& PreparedRuntimeChatTemplate::operator=(PreparedRuntimeChatTemplate&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    handle = std::exchange(other.handle, nullptr);
    destroyFn = std::exchange(other.destroyFn, nullptr);
    return *this;
}

PreparedRuntimeChatTemplate::~PreparedRuntimeChatTemplate() {
    reset();
}

bool PreparedRuntimeChatTemplate::isPrepared() const {
    return handle != nullptr;
}

void PreparedRuntimeChatTemplate::reset() {
    if (handle != nullptr && destroyFn != nullptr) {
        destroyFn(handle);
    }
    handle = nullptr;
    destroyFn = nullptr;
}

RuntimeChatTemplatePrepareStatus prepareRuntimeChatTemplate(
    const std::string& modelsPath,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    PreparedRuntimeChatTemplate& preparedTemplate,
    std::string& output) {
    std::lock_guard<std::mutex> lock(runtimeApiMutex());
    initializeRuntimeApiLocked();

    auto& api = runtimeApi();
    if (!api.available || api.createPreparedFn == nullptr || api.destroyPreparedFn == nullptr) {
        return RuntimeChatTemplatePrepareStatus::UNAVAILABLE;
    }

    preparedTemplate.reset();

    void* runtimeHandle = nullptr;
    const char* runtimeOutput = nullptr;
    bool ok = api.createPreparedFn(
        modelsPath.c_str(),
        chatTemplate.c_str(),
        bosToken.c_str(),
        eosToken.c_str(),
        &runtimeHandle,
        &runtimeOutput);

    if (runtimeOutput != nullptr) {
        output = runtimeOutput;
    } else {
        output.clear();
    }

    if (!ok) {
        if (output.empty()) {
            output = "Failed to prepare chat template using runtime Python Jinja";
        }
        return RuntimeChatTemplatePrepareStatus::ERROR;
    }

    preparedTemplate.handle = runtimeHandle;
    preparedTemplate.destroyFn = api.destroyPreparedFn;
    return RuntimeChatTemplatePrepareStatus::PREPARED;
}

RuntimeChatTemplateStatus tryApplyPreparedChatTemplateRuntime(
    const PreparedRuntimeChatTemplate& preparedTemplate,
    const std::string& requestBody,
    std::string& output) {
    std::lock_guard<std::mutex> lock(runtimeApiMutex());
    initializeRuntimeApiLocked();

    auto& api = runtimeApi();
    if (!api.available || api.applyPreparedFn == nullptr || !preparedTemplate.isPrepared()) {
        return RuntimeChatTemplateStatus::UNAVAILABLE;
    }

    const char* runtimeOutput = nullptr;
    bool ok = api.applyPreparedFn(
        preparedTemplate.handle,
        requestBody.c_str(),
        &runtimeOutput);

    if (runtimeOutput != nullptr) {
        output = runtimeOutput;
    } else {
        output.clear();
    }

    if (!ok) {
        if (output.empty()) {
            output = "Failed to apply prepared chat template using runtime Python Jinja";
        }
        return RuntimeChatTemplateStatus::ERROR;
    }

    return RuntimeChatTemplateStatus::APPLIED;
}

RuntimeChatTemplateStatus tryApplyChatTemplateRuntime(
    const std::string& modelsPath,
    const std::string& requestBody,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    std::string& output) {
    PreparedRuntimeChatTemplate preparedTemplate;
    auto prepareStatus = prepareRuntimeChatTemplate(
        modelsPath,
        chatTemplate,
        bosToken,
        eosToken,
        preparedTemplate,
        output);
    if (prepareStatus == RuntimeChatTemplatePrepareStatus::UNAVAILABLE) {
        return RuntimeChatTemplateStatus::UNAVAILABLE;
    }
    if (prepareStatus == RuntimeChatTemplatePrepareStatus::ERROR) {
        return RuntimeChatTemplateStatus::ERROR;
    }
    return tryApplyPreparedChatTemplateRuntime(preparedTemplate, requestBody, output);
}

}  // namespace ovms
