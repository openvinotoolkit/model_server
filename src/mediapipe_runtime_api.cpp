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

#include "mediapipe_runtime_api.hpp"

#include <array>
#include <memory>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include "logging.hpp"
#include "mediapipe_graph_executor_interface.hpp"
#include "servable_definition.hpp"

#ifdef __linux__
extern "C" void* OVMS_MPFactoryCreate(void*) __attribute__((weak));
extern "C" void OVMS_MPFactoryDestroy(void*) __attribute__((weak));
extern "C" const char* OVMS_MPFactoryGetLastError() __attribute__((weak));
extern "C" int OVMS_MPFactoryProcessConfig(void*, const ovms::MediapipeGraphConfig*, ovms::MetricProvider*, const ovms::ServableNameChecker*) __attribute__((weak));
extern "C" int OVMS_MPFactoryCreateExecutor(void*, const char*, std::unique_ptr<ovms::MediapipeGraphExecutor>*) __attribute__((weak));
extern "C" int OVMS_MPFactoryCreateExecutorHandle(void*, const char*, std::unique_ptr<ovms::MediapipeGraphExecutorInterface>*) __attribute__((weak));
extern "C" int OVMS_MPFactoryDefinitionExists(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryAliasesConflictExcluding(void*, const char*, const char*) __attribute__((weak));
extern "C" void OVMS_MPFactoryRetireOtherThan(void*, const char*) __attribute__((weak));
extern "C" const char* OVMS_MPFactoryGetNames(void*, int) __attribute__((weak));
extern "C" void* OVMS_MPFactoryFindServableDefinitionByName(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPGraphExportCreateServableConfig(const char*, const ovms::HFSettingsImpl*, int) __attribute__((weak));
#endif

namespace ovms {

struct MediapipeRuntimeApi::ApiSymbols {
#ifdef __linux__
    using LibraryHandle = void*;
#elif _WIN32
    using LibraryHandle = HMODULE;
#endif

    using CreateFn = void* (*)(void*);
    using DestroyFn = void (*)(void*);
    using LastErrorFn = const char* (*)();
    using ProcessConfigFn = int (*)(void*, const MediapipeGraphConfig*, MetricProvider*, const ServableNameChecker*);
    using CreateExecutorFn = int (*)(void*, const char*, std::unique_ptr<MediapipeGraphExecutor>*);
    using CreateExecutorHandleFn = int (*)(void*, const char*, std::unique_ptr<MediapipeGraphExecutorInterface>*);
    using DefinitionExistsFn = int (*)(void*, const char*);
    using AliasesConflictExcludingFn = int (*)(void*, const char*, const char*);
    using RetireOtherThanFn = void (*)(void*, const char*);
    using GetNamesFn = const char* (*)(void*, int);
    using FindServableDefinitionFn = void* (*)(void*, const char*);
    using CreateServableConfigFn = int (*)(const char*, const HFSettingsImpl*, int);
    LibraryHandle handle = nullptr;
    void* factoryHandle = nullptr;

    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    LastErrorFn lastError = nullptr;
    ProcessConfigFn processConfig = nullptr;
    CreateExecutorFn createExecutor = nullptr;
    CreateExecutorHandleFn createExecutorHandle = nullptr;
    DefinitionExistsFn definitionExists = nullptr;
    AliasesConflictExcludingFn aliasesConflictExcluding = nullptr;
    RetireOtherThanFn retireOtherThan = nullptr;
    GetNamesFn getNames = nullptr;
    FindServableDefinitionFn findServableDefinition = nullptr;
    CreateServableConfigFn createServableConfig = nullptr;
};

#ifdef __linux__
static void* resolveSymbol(void* handle, const char* name) {
    return dlsym(handle, name);
}
#elif _WIN32
static void* resolveSymbol(HMODULE handle, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(handle, name));
}
#endif

MediapipeRuntimeApi::MediapipeRuntimeApi(PythonBackend* pythonBackend) :
    api(std::make_unique<ApiSymbols>()) {
    bool loadedFromInProcessSymbols = false;

#ifdef __linux__
    const bool preferInProcessSymbols = []() {
        const char* value = std::getenv("OVMS_TEST_MEDIAPIPE_RUNTIME_INPROCESS");
        return value != nullptr && std::string(value) == "1";
    }();
    if (preferInProcessSymbols) {
        api->create = OVMS_MPFactoryCreate != nullptr ? OVMS_MPFactoryCreate : reinterpret_cast<ApiSymbols::CreateFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryCreate"));
        api->destroy = OVMS_MPFactoryDestroy != nullptr ? OVMS_MPFactoryDestroy : reinterpret_cast<ApiSymbols::DestroyFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryDestroy"));
        api->lastError = OVMS_MPFactoryGetLastError != nullptr ? OVMS_MPFactoryGetLastError : reinterpret_cast<ApiSymbols::LastErrorFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryGetLastError"));
        api->processConfig = OVMS_MPFactoryProcessConfig != nullptr ? OVMS_MPFactoryProcessConfig : reinterpret_cast<ApiSymbols::ProcessConfigFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryProcessConfig"));
        api->createExecutor = OVMS_MPFactoryCreateExecutor != nullptr ? OVMS_MPFactoryCreateExecutor : reinterpret_cast<ApiSymbols::CreateExecutorFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryCreateExecutor"));
        api->createExecutorHandle = OVMS_MPFactoryCreateExecutorHandle != nullptr ? OVMS_MPFactoryCreateExecutorHandle : reinterpret_cast<ApiSymbols::CreateExecutorHandleFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryCreateExecutorHandle"));
        api->definitionExists = OVMS_MPFactoryDefinitionExists != nullptr ? OVMS_MPFactoryDefinitionExists : reinterpret_cast<ApiSymbols::DefinitionExistsFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryDefinitionExists"));
        api->aliasesConflictExcluding = OVMS_MPFactoryAliasesConflictExcluding != nullptr ? OVMS_MPFactoryAliasesConflictExcluding : reinterpret_cast<ApiSymbols::AliasesConflictExcludingFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryAliasesConflictExcluding"));
        api->retireOtherThan = OVMS_MPFactoryRetireOtherThan != nullptr ? OVMS_MPFactoryRetireOtherThan : reinterpret_cast<ApiSymbols::RetireOtherThanFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryRetireOtherThan"));
        api->getNames = OVMS_MPFactoryGetNames != nullptr ? OVMS_MPFactoryGetNames : reinterpret_cast<ApiSymbols::GetNamesFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryGetNames"));
        api->findServableDefinition = OVMS_MPFactoryFindServableDefinitionByName != nullptr ? OVMS_MPFactoryFindServableDefinitionByName : reinterpret_cast<ApiSymbols::FindServableDefinitionFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPFactoryFindServableDefinitionByName"));
        api->createServableConfig = OVMS_MPGraphExportCreateServableConfig != nullptr ? OVMS_MPGraphExportCreateServableConfig : reinterpret_cast<ApiSymbols::CreateServableConfigFn>(resolveSymbol(RTLD_DEFAULT, "OVMS_MPGraphExportCreateServableConfig"));

        loadedFromInProcessSymbols =
            api->create != nullptr &&
            api->destroy != nullptr &&
            api->lastError != nullptr &&
            api->processConfig != nullptr &&
            api->createExecutor != nullptr &&
            api->createExecutorHandle != nullptr &&
            api->definitionExists != nullptr &&
            api->aliasesConflictExcluding != nullptr &&
            api->retireOtherThan != nullptr &&
            api->getNames != nullptr &&
            api->findServableDefinition != nullptr &&
            api->createServableConfig != nullptr;

        if (loadedFromInProcessSymbols) {
            SPDLOG_INFO("MediaPipe runtime API resolved from in-process symbols");
        } else {
            std::vector<std::string> missingSymbols;
            if (api->create == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryCreate");
            if (api->destroy == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryDestroy");
            if (api->lastError == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryGetLastError");
            if (api->processConfig == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryProcessConfig");
            if (api->createExecutor == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryCreateExecutor");
            if (api->createExecutorHandle == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryCreateExecutorHandle");
            if (api->definitionExists == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryDefinitionExists");
            if (api->aliasesConflictExcluding == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryAliasesConflictExcluding");
            if (api->retireOtherThan == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryRetireOtherThan");
            if (api->getNames == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryGetNames");
            if (api->findServableDefinition == nullptr) missingSymbols.emplace_back("OVMS_MPFactoryFindServableDefinitionByName");
            if (api->createServableConfig == nullptr) missingSymbols.emplace_back("OVMS_MPGraphExportCreateServableConfig");
            SPDLOG_WARN("OVMS_TEST_MEDIAPIPE_RUNTIME_INPROCESS=1 but in-process runtime API symbols are incomplete. Missing: {}", joinWithNewlines(missingSymbols));
        }
    }
#endif

#ifndef __linux__
    (void)loadedFromInProcessSymbols;
#endif

    if (!loadedFromInProcessSymbols) {
#ifdef __linux__
    std::vector<std::string> candidates{
        "libovms_mediapipe_runtime_shared.so",
        "/ovms/lib/libovms_mediapipe_runtime_shared.so",
        "./libovms_mediapipe_runtime_shared.so",
        "src/libovms_mediapipe_runtime_shared.so",
        "./src/libovms_mediapipe_runtime_shared.so",
        "bazel-bin/src/libovms_mediapipe_runtime_shared.so",
        "./bazel-bin/src/libovms_mediapipe_runtime_shared.so"};

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr) {
        std::vector<std::string> runfilesCandidates{
            std::string(testSrcDir) + "/_main/src/libovms_mediapipe_runtime_shared.so",
            std::string(testSrcDir) + "/ovms/src/libovms_mediapipe_runtime_shared.so"};
        candidates.insert(candidates.end(), runfilesCandidates.begin(), runfilesCandidates.end());
    }

    std::array<char, 4096> exePath{};
    ssize_t exePathLength = readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
    if (exePathLength > 0) {
        exePath[exePathLength] = '\0';
        std::filesystem::path exeDir = std::filesystem::path(exePath.data()).parent_path();
        std::vector<std::string> exeRelativeCandidates{
            (exeDir / "libovms_mediapipe_runtime_shared.so").string(),
            (exeDir / "src/libovms_mediapipe_runtime_shared.so").string()};
        candidates.insert(candidates.end(), exeRelativeCandidates.begin(), exeRelativeCandidates.end());
    }

    for (const auto& candidate : candidates) {
        api->handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (api->handle != nullptr) {
            SPDLOG_INFO("MediaPipe runtime API loaded from: {}", candidate);
            break;
        }
    }
#elif _WIN32
    std::vector<std::string> candidates{
        "libovms_mediapipe_runtime_shared.dll",
        ".\\libovms_mediapipe_runtime_shared.dll",
        "src\\libovms_mediapipe_runtime_shared.dll",
        ".\\src\\libovms_mediapipe_runtime_shared.dll",
        "bazel-bin\\src\\libovms_mediapipe_runtime_shared.dll",
        ".\\bazel-bin\\src\\libovms_mediapipe_runtime_shared.dll"};

    for (const auto& candidate : candidates) {
        api->handle = LoadLibraryA(candidate.c_str());
        if (api->handle != nullptr) {
            SPDLOG_INFO("MediaPipe runtime API loaded from: {}", candidate);
            break;
        }
    }
#endif
    }

    if (!loadedFromInProcessSymbols && api->handle == nullptr) {
        SPDLOG_ERROR("MediaPipe runtime API library is unavailable");
        return;
    }

    if (!loadedFromInProcessSymbols) {
        api->create = reinterpret_cast<ApiSymbols::CreateFn>(resolveSymbol(api->handle, "OVMS_MPFactoryCreate"));
        api->destroy = reinterpret_cast<ApiSymbols::DestroyFn>(resolveSymbol(api->handle, "OVMS_MPFactoryDestroy"));
        api->lastError = reinterpret_cast<ApiSymbols::LastErrorFn>(resolveSymbol(api->handle, "OVMS_MPFactoryGetLastError"));
        api->processConfig = reinterpret_cast<ApiSymbols::ProcessConfigFn>(resolveSymbol(api->handle, "OVMS_MPFactoryProcessConfig"));
        api->createExecutor = reinterpret_cast<ApiSymbols::CreateExecutorFn>(resolveSymbol(api->handle, "OVMS_MPFactoryCreateExecutor"));
        api->createExecutorHandle = reinterpret_cast<ApiSymbols::CreateExecutorHandleFn>(resolveSymbol(api->handle, "OVMS_MPFactoryCreateExecutorHandle"));
        api->definitionExists = reinterpret_cast<ApiSymbols::DefinitionExistsFn>(resolveSymbol(api->handle, "OVMS_MPFactoryDefinitionExists"));
        api->aliasesConflictExcluding = reinterpret_cast<ApiSymbols::AliasesConflictExcludingFn>(resolveSymbol(api->handle, "OVMS_MPFactoryAliasesConflictExcluding"));
        api->retireOtherThan = reinterpret_cast<ApiSymbols::RetireOtherThanFn>(resolveSymbol(api->handle, "OVMS_MPFactoryRetireOtherThan"));
        api->getNames = reinterpret_cast<ApiSymbols::GetNamesFn>(resolveSymbol(api->handle, "OVMS_MPFactoryGetNames"));
        api->findServableDefinition = reinterpret_cast<ApiSymbols::FindServableDefinitionFn>(resolveSymbol(api->handle, "OVMS_MPFactoryFindServableDefinitionByName"));
        api->createServableConfig = reinterpret_cast<ApiSymbols::CreateServableConfigFn>(resolveSymbol(api->handle, "OVMS_MPGraphExportCreateServableConfig"));
    }

    if (api->create == nullptr ||
        api->destroy == nullptr ||
        api->lastError == nullptr ||
        api->processConfig == nullptr ||
        api->createExecutor == nullptr ||
        api->createExecutorHandle == nullptr ||
        api->definitionExists == nullptr ||
        api->aliasesConflictExcluding == nullptr ||
        api->retireOtherThan == nullptr ||
        api->getNames == nullptr ||
        api->findServableDefinition == nullptr ||
        api->createServableConfig == nullptr) {
        SPDLOG_ERROR("MediaPipe runtime API symbols could not be resolved");
        return;
    }

    api->factoryHandle = api->create(static_cast<void*>(pythonBackend));
    if (api->factoryHandle == nullptr) {
        SPDLOG_ERROR("MediaPipe runtime API factory creation failed");
        return;
    }
}

MediapipeRuntimeApi::~MediapipeRuntimeApi() {
    if (api == nullptr) {
        return;
    }

    if (api->factoryHandle != nullptr && api->destroy != nullptr) {
        api->destroy(api->factoryHandle);
        api->factoryHandle = nullptr;
    }

#ifdef __linux__
    if (api->handle != nullptr) {
        dlclose(api->handle);
    }
#elif _WIN32
    if (api->handle != nullptr) {
        FreeLibrary(api->handle);
    }
#endif
    api->handle = nullptr;
}

bool MediapipeRuntimeApi::isLoaded() const {
    return api != nullptr && api->factoryHandle != nullptr;
}

Status MediapipeRuntimeApi::processConfig(const MediapipeGraphConfig& config,
    MetricProvider& metrics,
    const ServableNameChecker& checker) {
    if (!isLoaded()) {
        return StatusCode::INTERNAL_ERROR;
    }
    int code = api->processConfig(api->factoryHandle, &config, &metrics, &checker);
    if (code == static_cast<int>(StatusCode::OK)) {
        return StatusCode::OK;
    }
    const char* details = api->lastError();
    if (details == nullptr) {
        return Status(static_cast<StatusCode>(code), "MediaPipe runtime API error");
    }
    return Status(static_cast<StatusCode>(code), details);
}

Status MediapipeRuntimeApi::create(std::unique_ptr<MediapipeGraphExecutor>& pipeline,
    const std::string& name) const {
    if (!isLoaded()) {
        return StatusCode::INTERNAL_ERROR;
    }
    int code = api->createExecutor(api->factoryHandle, name.c_str(), &pipeline);
    if (code != static_cast<int>(StatusCode::OK)) {
        const char* details = api->lastError();
        if (details == nullptr) {
            return Status(static_cast<StatusCode>(code), "MediaPipe runtime API error");
        }
        return Status(static_cast<StatusCode>(code), details);
    }
    return StatusCode::OK;
}

Status MediapipeRuntimeApi::createHandle(std::unique_ptr<MediapipeGraphExecutorInterface>& pipeline,
    const std::string& name) const {
    if (!isLoaded()) {
        return StatusCode::INTERNAL_ERROR;
    }
    int code = api->createExecutorHandle(api->factoryHandle, name.c_str(), &pipeline);
    if (code != static_cast<int>(StatusCode::OK)) {
        const char* details = api->lastError();
        if (details == nullptr) {
            return Status(static_cast<StatusCode>(code), "MediaPipe runtime API error");
        }
        return Status(static_cast<StatusCode>(code), details);
    }
    return StatusCode::OK;
}

bool MediapipeRuntimeApi::definitionExists(const std::string& name) const {
    if (!isLoaded()) {
        return false;
    }
    return api->definitionExists(api->factoryHandle, name.c_str()) != 0;
}

bool MediapipeRuntimeApi::aliasesConflictExcluding(const std::vector<std::string>& aliases, const std::string& ownGraphName) const {
    if (!isLoaded()) {
        return false;
    }
    std::string joinedAliases = joinWithNewlines(aliases);
    return api->aliasesConflictExcluding(api->factoryHandle, joinedAliases.c_str(), ownGraphName.c_str()) != 0;
}

void MediapipeRuntimeApi::retireOtherThan(const std::set<std::string>& graphsInConfigFile) {
    if (!isLoaded()) {
        return;
    }
    std::string joined = joinWithNewlines(graphsInConfigFile);
    api->retireOtherThan(api->factoryHandle, joined.c_str());
}

const std::vector<std::string> MediapipeRuntimeApi::getMediapipePipelinesNames() const {
    if (!isLoaded()) {
        return {};
    }
    const char* names = api->getNames(api->factoryHandle, 0);
    if (names == nullptr) {
        return {};
    }
    return splitNewlineDelimited(names);
}

const std::vector<std::string> MediapipeRuntimeApi::getNamesOfAvailableMediapipePipelines() const {
    if (!isLoaded()) {
        return {};
    }
    const char* names = api->getNames(api->factoryHandle, 1);
    if (names == nullptr) {
        return {};
    }
    return splitNewlineDelimited(names);
}

MediapipeGraphDefinition* MediapipeRuntimeApi::findDefinitionByName(const std::string& name) const {
    return reinterpret_cast<MediapipeGraphDefinition*>(findServableDefinitionByName(name));
}

ServableDefinition* MediapipeRuntimeApi::findServableDefinitionByName(const std::string& name) const {
    if (!isLoaded()) {
        return nullptr;
    }
    return reinterpret_cast<ServableDefinition*>(api->findServableDefinition(api->factoryHandle, name.c_str()));
}

Status MediapipeRuntimeApi::createServableConfig(const std::string& directoryPath,
    const HFSettingsImpl& hfSettings,
    bool writeToFile) const {
    if (api == nullptr || api->handle == nullptr || api->createServableConfig == nullptr) {
        return StatusCode::INTERNAL_ERROR;
    }
    int code = api->createServableConfig(directoryPath.c_str(), &hfSettings, writeToFile ? 1 : 0);
    if (code == static_cast<int>(StatusCode::OK)) {
        return StatusCode::OK;
    }
    const char* details = api->lastError ? api->lastError() : nullptr;
    if (details == nullptr) {
        return Status(static_cast<StatusCode>(code), "MediaPipe runtime API error");
    }
    return Status(static_cast<StatusCode>(code), details);
}

std::string MediapipeRuntimeApi::joinWithNewlines(const std::vector<std::string>& values) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        joined += values[i];
        if (i + 1 < values.size()) {
            joined += '\n';
        }
    }
    return joined;
}

std::string MediapipeRuntimeApi::joinWithNewlines(const std::set<std::string>& values) {
    std::string joined;
    size_t index = 0;
    for (const auto& value : values) {
        joined += value;
        if (index + 1 < values.size()) {
            joined += '\n';
        }
        ++index;
    }
    return joined;
}

std::vector<std::string> MediapipeRuntimeApi::splitNewlineDelimited(const std::string& data) {
    std::vector<std::string> values;
    if (data.empty()) {
        return values;
    }

    size_t start = 0;
    while (start <= data.size()) {
        size_t end = data.find('\n', start);
        std::string value = (end == std::string::npos) ? data.substr(start) : data.substr(start, end - start);
        if (!value.empty()) {
            values.push_back(std::move(value));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

}  // namespace ovms
