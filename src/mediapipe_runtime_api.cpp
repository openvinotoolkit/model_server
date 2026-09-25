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
//*****************************************************************************/

#include "mediapipe_runtime_api.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#elif _WIN32
#include <windows.h>
#endif

#include "config.hpp"
#include "kfs_python_tensor_bridge.hpp"
#include "logging.hpp"
#include "mediapipe_internal/mediapipe_graph_executor_interface.hpp"
#include "utils/newline_delimited.hpp"

struct OVMS_Server_;
using OVMS_Server = OVMS_Server_;
struct OVMS_Status_;
using OVMS_Status = OVMS_Status_;
extern "C" OVMS_Status* OVMS_ServerNew(OVMS_Server** server);
extern "C" void OVMS_StatusDelete(OVMS_Status* status);

#ifdef __linux__
extern "C" void* OVMS_MPFactoryCreate(void*) __attribute__((weak));
extern "C" void OVMS_MPFactoryDestroy(void*) __attribute__((weak));
extern "C" const char* OVMS_MPFactoryGetLastError() __attribute__((weak));
extern "C" int OVMS_MPFactoryProcessConfig(void*, const ovms::MediapipeGraphConfig*, ovms::MetricProvider*, const ovms::ServableNameChecker*, int) __attribute__((weak));
extern "C" int OVMS_MPFactoryCreateExecutor(void*, const char*, std::unique_ptr<ovms::MediapipeGraphExecutor>*) __attribute__((weak));
extern "C" int OVMS_MPFactoryCreateExecutorHandle(void*, const char*, std::unique_ptr<ovms::MediapipeGraphExecutorInterface>*) __attribute__((weak));
extern "C" int OVMS_MPFactoryDefinitionExists(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryWakeUpDefinition(void*, const char*, const ovms::ServableNameChecker*) __attribute__((weak));
extern "C" int OVMS_MPFactoryPutToSleepDefinition(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryRetireDefinition(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryIsDefinitionRetired(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryIsDefinitionAvailable(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryShouldUnloadDefinitionDueToIdle(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryHasActiveInference(void*, const char*) __attribute__((weak));
extern "C" const char* OVMS_MPFactoryGetDefinitionGroupName(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPFactoryAliasesConflictExcluding(void*, const char*, const char*) __attribute__((weak));
extern "C" const char* OVMS_MPFactoryGetNames(void*, int) __attribute__((weak));
extern "C" void* OVMS_MPFactoryFindServableDefinitionByName(void*, const char*) __attribute__((weak));
extern "C" int OVMS_MPGraphExportCreateServableConfig(const char*, const ovms::HFSettingsImpl*) __attribute__((weak));
extern "C" int OVMS_MPGraphExportCreateServableConfigInMemory(const char*, const ovms::HFSettingsImpl*, char**) __attribute__((weak));
extern "C" const ovms::KfsPyTensorBridgeVTable* OVMS_getKfsPyTensorBridgeVTable() __attribute__((weak));
extern "C" void OVMS_MPSetExternalServerHandle(void*) __attribute__((weak));
extern "C" void OVMS_MPFactoryConfigureLogging(const char*, const char*) __attribute__((weak));
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
    using ProcessConfigFn = int (*)(void*, const MediapipeGraphConfig*, MetricProvider*, const ServableNameChecker*, int);
    using CreateExecutorFn = int (*)(void*, const char*, std::unique_ptr<MediapipeGraphExecutor>*);
    using CreateExecutorHandleFn = int (*)(void*, const char*, std::unique_ptr<MediapipeGraphExecutorInterface>*);
    using DefinitionExistsFn = int (*)(void*, const char*);
    using WakeUpDefinitionFn = int (*)(void*, const char*, const ServableNameChecker*);
    using PutToSleepDefinitionFn = int (*)(void*, const char*);
    using RetireDefinitionFn = int (*)(void*, const char*);
    using IsDefinitionRetiredFn = int (*)(void*, const char*);
    using IsDefinitionAvailableFn = int (*)(void*, const char*);
    using ShouldUnloadDefinitionDueToIdleFn = int (*)(void*, const char*);
    using HasActiveInferenceFn = int (*)(void*, const char*);
    using GetDefinitionGroupNameFn = const char* (*)(void*, const char*);
    using AliasesConflictExcludingFn = int (*)(void*, const char*, const char*);
    using GetNamesFn = const char* (*)(void*, int);
    using FindServableDefinitionFn = void* (*)(void*, const char*);
    using CreateServableConfigFn = int (*)(const char*, const HFSettingsImpl*);
    using CreateServableConfigInMemoryFn = int (*)(const char*, const HFSettingsImpl*, char**);
    using SetExternalServerHandleFn = void (*)(void*);
    using ConfigureLoggingFn = void (*)(const char*, const char*);

    LibraryHandle handle = nullptr;
    void* factoryHandle = nullptr;
    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    LastErrorFn lastError = nullptr;
    ProcessConfigFn processConfig = nullptr;
    CreateExecutorFn createExecutor = nullptr;
    CreateExecutorHandleFn createExecutorHandle = nullptr;
    DefinitionExistsFn definitionExists = nullptr;
    WakeUpDefinitionFn wakeUpDefinition = nullptr;
    PutToSleepDefinitionFn putToSleepDefinition = nullptr;
    RetireDefinitionFn retireDefinition = nullptr;
    IsDefinitionRetiredFn isDefinitionRetired = nullptr;
    IsDefinitionAvailableFn isDefinitionAvailable = nullptr;
    ShouldUnloadDefinitionDueToIdleFn shouldUnloadDefinitionDueToIdle = nullptr;
    HasActiveInferenceFn hasActiveInference = nullptr;
    GetDefinitionGroupNameFn getDefinitionGroupName = nullptr;
    AliasesConflictExcludingFn aliasesConflictExcluding = nullptr;
    GetNamesFn getNames = nullptr;
    FindServableDefinitionFn findServableDefinition = nullptr;
    CreateServableConfigFn createServableConfig = nullptr;
    CreateServableConfigInMemoryFn createServableConfigInMemory = nullptr;
    SetExternalServerHandleFn setExternalServerHandle = nullptr;
    ConfigureLoggingFn configureLogging = nullptr;
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

namespace {
void tryActivateKfsPythonTensorBridgeFromRuntimeSymbols(
#ifdef __linux__
    void* handle
#elif _WIN32
    HMODULE handle
#endif
) {
    using GetKfsBridgeFn = const ovms::KfsPyTensorBridgeVTable* (*)();
    using SetKfsBridgeFn = int (*)(const ovms::KfsPyTensorBridgeVTable*);

    if (ovms::getKfsPyTensorBridgeVTable() != nullptr) {
        return;
    }

#ifdef __linux__
    auto* getBridgeFn = OVMS_getKfsPyTensorBridgeVTable != nullptr ? OVMS_getKfsPyTensorBridgeVTable : reinterpret_cast<GetKfsBridgeFn>(resolveSymbol(handle, "OVMS_getKfsPyTensorBridgeVTable"));
#elif _WIN32
    auto* getBridgeFn = reinterpret_cast<GetKfsBridgeFn>(resolveSymbol(handle, "OVMS_getKfsPyTensorBridgeVTable"));
#endif
    if (getBridgeFn == nullptr) {
        return;
    }

    if (auto* bridge = getBridgeFn(); bridge != nullptr) {
#ifdef __linux__
        auto* setBridgeFn = reinterpret_cast<SetKfsBridgeFn>(resolveSymbol(handle, "OVMS_setKfsPyTensorBridgeVTable"));
#elif _WIN32
        auto* setBridgeFn = reinterpret_cast<SetKfsBridgeFn>(resolveSymbol(handle, "OVMS_setKfsPyTensorBridgeVTable"));
#endif
        if (setBridgeFn != nullptr) {
            setBridgeFn(bridge);
        }
        ovms::setKfsPyTensorBridgeVTable(bridge);
        SPDLOG_TRACE("KFS Python tensor bridge activated from in-process MediaPipe runtime symbols");
    }
}
}  // namespace

MediapipeRuntimeApi::MediapipeRuntimeApi(PythonBackend* pythonBackend) :
    api(std::make_unique<ApiSymbols>()) {
#ifdef __linux__
    void* currentModule = RTLD_DEFAULT;
#elif _WIN32
    HMODULE currentModule = GetModuleHandleA(nullptr);
    if (currentModule == nullptr) {
        SPDLOG_ERROR("In-process MediaPipe runtime symbols are unavailable");
        return;
    }
#endif

    api->create = reinterpret_cast<ApiSymbols::CreateFn>(resolveSymbol(currentModule, "OVMS_MPFactoryCreate"));
    api->destroy = reinterpret_cast<ApiSymbols::DestroyFn>(resolveSymbol(currentModule, "OVMS_MPFactoryDestroy"));
    api->lastError = reinterpret_cast<ApiSymbols::LastErrorFn>(resolveSymbol(currentModule, "OVMS_MPFactoryGetLastError"));
    api->processConfig = reinterpret_cast<ApiSymbols::ProcessConfigFn>(resolveSymbol(currentModule, "OVMS_MPFactoryProcessConfig"));
    api->createExecutor = reinterpret_cast<ApiSymbols::CreateExecutorFn>(resolveSymbol(currentModule, "OVMS_MPFactoryCreateExecutor"));
    api->createExecutorHandle = reinterpret_cast<ApiSymbols::CreateExecutorHandleFn>(resolveSymbol(currentModule, "OVMS_MPFactoryCreateExecutorHandle"));
    api->definitionExists = reinterpret_cast<ApiSymbols::DefinitionExistsFn>(resolveSymbol(currentModule, "OVMS_MPFactoryDefinitionExists"));
    api->wakeUpDefinition = reinterpret_cast<ApiSymbols::WakeUpDefinitionFn>(resolveSymbol(currentModule, "OVMS_MPFactoryWakeUpDefinition"));
    api->putToSleepDefinition = reinterpret_cast<ApiSymbols::PutToSleepDefinitionFn>(resolveSymbol(currentModule, "OVMS_MPFactoryPutToSleepDefinition"));
    api->retireDefinition = reinterpret_cast<ApiSymbols::RetireDefinitionFn>(resolveSymbol(currentModule, "OVMS_MPFactoryRetireDefinition"));
    api->isDefinitionRetired = reinterpret_cast<ApiSymbols::IsDefinitionRetiredFn>(resolveSymbol(currentModule, "OVMS_MPFactoryIsDefinitionRetired"));
    api->isDefinitionAvailable = reinterpret_cast<ApiSymbols::IsDefinitionAvailableFn>(resolveSymbol(currentModule, "OVMS_MPFactoryIsDefinitionAvailable"));
    api->shouldUnloadDefinitionDueToIdle = reinterpret_cast<ApiSymbols::ShouldUnloadDefinitionDueToIdleFn>(resolveSymbol(currentModule, "OVMS_MPFactoryShouldUnloadDefinitionDueToIdle"));
    api->hasActiveInference = reinterpret_cast<ApiSymbols::HasActiveInferenceFn>(resolveSymbol(currentModule, "OVMS_MPFactoryHasActiveInference"));
    api->getDefinitionGroupName = reinterpret_cast<ApiSymbols::GetDefinitionGroupNameFn>(resolveSymbol(currentModule, "OVMS_MPFactoryGetDefinitionGroupName"));
    api->aliasesConflictExcluding = reinterpret_cast<ApiSymbols::AliasesConflictExcludingFn>(resolveSymbol(currentModule, "OVMS_MPFactoryAliasesConflictExcluding"));
    api->getNames = reinterpret_cast<ApiSymbols::GetNamesFn>(resolveSymbol(currentModule, "OVMS_MPFactoryGetNames"));
    api->findServableDefinition = reinterpret_cast<ApiSymbols::FindServableDefinitionFn>(resolveSymbol(currentModule, "OVMS_MPFactoryFindServableDefinitionByName"));
    api->createServableConfig = reinterpret_cast<ApiSymbols::CreateServableConfigFn>(resolveSymbol(currentModule, "OVMS_MPGraphExportCreateServableConfig"));
    api->createServableConfigInMemory = reinterpret_cast<ApiSymbols::CreateServableConfigInMemoryFn>(resolveSymbol(currentModule, "OVMS_MPGraphExportCreateServableConfigInMemory"));
    api->setExternalServerHandle = reinterpret_cast<ApiSymbols::SetExternalServerHandleFn>(resolveSymbol(currentModule, "OVMS_MPSetExternalServerHandle"));
    api->configureLogging = reinterpret_cast<ApiSymbols::ConfigureLoggingFn>(resolveSymbol(currentModule, "OVMS_MPFactoryConfigureLogging"));

    if (api->create == nullptr || api->destroy == nullptr || api->lastError == nullptr || api->processConfig == nullptr ||
        api->createExecutor == nullptr || api->createExecutorHandle == nullptr || api->definitionExists == nullptr ||
        api->wakeUpDefinition == nullptr || api->putToSleepDefinition == nullptr || api->retireDefinition == nullptr ||
        api->isDefinitionRetired == nullptr || api->isDefinitionAvailable == nullptr ||
        api->shouldUnloadDefinitionDueToIdle == nullptr || api->hasActiveInference == nullptr ||
        api->getDefinitionGroupName == nullptr || api->aliasesConflictExcluding == nullptr || api->getNames == nullptr ||
        api->findServableDefinition == nullptr || api->createServableConfig == nullptr ||
        api->createServableConfigInMemory == nullptr) {
        SPDLOG_ERROR("In-process MediaPipe runtime symbols are unavailable");
        return;
    }

    SPDLOG_TRACE("MediaPipe runtime API resolved from in-process symbols");
    tryActivateKfsPythonTensorBridgeFromRuntimeSymbols(currentModule);

    if (api->configureLogging != nullptr) {
        const auto& config = Config::instance();
        api->configureLogging(config.logLevel().c_str(), config.logPath().c_str());
    }

    api->factoryHandle = api->create(static_cast<void*>(pythonBackend));
    if (api->factoryHandle == nullptr) {
        SPDLOG_ERROR("MediaPipe runtime API factory creation failed");
        return;
    }

    if (api->setExternalServerHandle != nullptr) {
        OVMS_Server* serverHandle = nullptr;
        auto* status = OVMS_ServerNew(&serverHandle);
        if (status != nullptr) {
            OVMS_StatusDelete(status);
            SPDLOG_WARN("Failed to obtain OVMS server handle for MediaPipe runtime");
        } else {
            api->setExternalServerHandle(static_cast<void*>(serverHandle));
        }
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
}

bool MediapipeRuntimeApi::isLoaded() const {
    return api != nullptr && api->factoryHandle != nullptr;
}

#define OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED() \
    if (!isLoaded()) {                                \
        return StatusCode::INTERNAL_ERROR;            \
    }

Status MediapipeRuntimeApi::processConfig(const MediapipeGraphConfig& config,
    MetricProvider& metrics,
    const ServableNameChecker& checker,
    bool lazyLoad) {
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
    int code = api->processConfig(api->factoryHandle, &config, &metrics, &checker, lazyLoad ? 1 : 0);
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
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
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
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
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
    return isLoaded() && api->definitionExists(api->factoryHandle, name.c_str()) != 0;
}

Status MediapipeRuntimeApi::wakeUpDefinition(const std::string& name, const ServableNameChecker& checker) const {
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
    int code = api->wakeUpDefinition(api->factoryHandle, name.c_str(), &checker);
    if (code == static_cast<int>(StatusCode::OK))
        return StatusCode::OK;
    const char* details = api->lastError();
    return details == nullptr ? Status(static_cast<StatusCode>(code), "MediaPipe runtime API error") : Status(static_cast<StatusCode>(code), details);
}

Status MediapipeRuntimeApi::putToSleepDefinition(const std::string& name) const {
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
    int code = api->putToSleepDefinition(api->factoryHandle, name.c_str());
    if (code == static_cast<int>(StatusCode::OK))
        return StatusCode::OK;
    const char* details = api->lastError();
    return details == nullptr ? Status(static_cast<StatusCode>(code), "MediaPipe runtime API error") : Status(static_cast<StatusCode>(code), details);
}

Status MediapipeRuntimeApi::retireDefinition(const std::string& name) const {
    OVMS_RETURN_IF_MEDIAPIPE_RUNTIME_NOT_LOADED();
    int code = api->retireDefinition(api->factoryHandle, name.c_str());
    if (code == static_cast<int>(StatusCode::OK))
        return StatusCode::OK;
    const char* details = api->lastError();
    return details == nullptr ? Status(static_cast<StatusCode>(code), "MediaPipe runtime API error") : Status(static_cast<StatusCode>(code), details);
}

bool MediapipeRuntimeApi::isDefinitionRetired(const std::string& name) const {
    return isLoaded() && api->isDefinitionRetired(api->factoryHandle, name.c_str()) != 0;
}

bool MediapipeRuntimeApi::isDefinitionAvailable(const std::string& name) const {
    return isLoaded() && api->isDefinitionAvailable(api->factoryHandle, name.c_str()) != 0;
}

bool MediapipeRuntimeApi::shouldUnloadDefinitionDueToIdle(const std::string& name) const {
    return isLoaded() && api->shouldUnloadDefinitionDueToIdle(api->factoryHandle, name.c_str()) != 0;
}

bool MediapipeRuntimeApi::hasActiveInference(const std::string& name) const {
    return isLoaded() && api->hasActiveInference(api->factoryHandle, name.c_str()) != 0;
}

std::string MediapipeRuntimeApi::getDefinitionGroupName(const std::string& name) const {
    if (!isLoaded())
        return "";
    const char* groupName = api->getDefinitionGroupName(api->factoryHandle, name.c_str());
    return groupName == nullptr ? "" : groupName;
}

bool MediapipeRuntimeApi::aliasesConflictExcluding(const std::vector<std::string>& aliases, const std::string& ownGraphName) const {
    if (!isLoaded())
        return false;
    std::string joinedAliases = joinWithNewlines(aliases);
    return api->aliasesConflictExcluding(api->factoryHandle, joinedAliases.c_str(), ownGraphName.c_str()) != 0;
}

const std::vector<std::string> MediapipeRuntimeApi::getMediapipePipelinesNames() const {
    if (!isLoaded())
        return {};
    const char* names = api->getNames(api->factoryHandle, 0);
    return names == nullptr ? std::vector<std::string>{} : splitNewlineDelimited(names);
}

const std::vector<std::string> MediapipeRuntimeApi::getNamesOfAvailableMediapipePipelines() const {
    if (!isLoaded())
        return {};
    const char* names = api->getNames(api->factoryHandle, 1);
    return names == nullptr ? std::vector<std::string>{} : splitNewlineDelimited(names);
}

MediapipeGraphDefinition* MediapipeRuntimeApi::findDefinitionByName(const std::string& name) const {
    return reinterpret_cast<MediapipeGraphDefinition*>(findServableDefinitionByName(name));
}

ServableDefinition* MediapipeRuntimeApi::findServableDefinitionByName(const std::string& name) const {
    if (!isLoaded())
        return nullptr;
    return reinterpret_cast<ServableDefinition*>(api->findServableDefinition(api->factoryHandle, name.c_str()));
}

Status MediapipeRuntimeApi::createServableConfig(const std::string& directoryPath, const HFSettingsImpl& hfSettings) const {
    if (api == nullptr || api->createServableConfig == nullptr)
        return StatusCode::INTERNAL_ERROR;
    int code = api->createServableConfig(directoryPath.c_str(), &hfSettings);
    if (code == static_cast<int>(StatusCode::OK))
        return StatusCode::OK;
    const char* details = api->lastError ? api->lastError() : nullptr;
    return details == nullptr ? Status(static_cast<StatusCode>(code), "MediaPipe runtime API error") : Status(static_cast<StatusCode>(code), details);
}

Status MediapipeRuntimeApi::createServableConfigInMemory(const std::string& directoryPath,
    const HFSettingsImpl& hfSettings,
    std::string& outPbtxt) const {
    if (api == nullptr || api->createServableConfigInMemory == nullptr)
        return StatusCode::INTERNAL_ERROR;
    char* buffer = nullptr;
    int code = api->createServableConfigInMemory(directoryPath.c_str(), &hfSettings, &buffer);
    std::unique_ptr<char, decltype(&std::free)> bufferGuard(buffer, &std::free);
    if (code == static_cast<int>(StatusCode::OK)) {
        if (buffer != nullptr)
            outPbtxt.assign(buffer);
        return StatusCode::OK;
    }
    const char* details = api->lastError ? api->lastError() : nullptr;
    return details == nullptr ? Status(static_cast<StatusCode>(code), "MediaPipe runtime API error") : Status(static_cast<StatusCode>(code), details);
}

}  // namespace ovms
