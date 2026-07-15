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

#include "mediapipefactory.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../metrics/metric_provider.hpp"
#include "../graph_export/graph_export.hpp"
#include "../servable_definition.hpp"
#include "../servable_name_checker.hpp"
#include "../status.hpp"
#include "mediapipegraphconfig.hpp"
#include "mediapipegraphdefinition.hpp"
#include "../logging.hpp"

#if defined(_WIN32)
#define MEDIAPIPE_RUNTIME_EXPORT __declspec(dllexport)
#else
#define MEDIAPIPE_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string gLastError;

std::vector<std::string> splitNewlineDelimited(const char* values) {
    std::vector<std::string> parsed;
    if (values == nullptr || values[0] == '\0') {
        return parsed;
    }

    std::string data(values);
    size_t start = 0;
    while (start <= data.size()) {
        size_t end = data.find('\n', start);
        std::string value = (end == std::string::npos) ? data.substr(start) : data.substr(start, end - start);
        if (!value.empty()) {
            parsed.push_back(std::move(value));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return parsed;
}

std::string joinWithNewlines(const std::vector<std::string>& values) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        joined += values[i];
        if (i + 1 < values.size()) {
            joined += '\n';
        }
    }
    return joined;
}

ovms::Status processConfigInternal(ovms::MediapipeFactory* factory,
    const ovms::MediapipeGraphConfig& config,
    ovms::MetricProvider& metrics,
    const ovms::ServableNameChecker& checker) {
    auto* definition = factory->findDefinitionByName(config.getGraphName());
    if (definition == nullptr) {
        return factory->createDefinition(config.getGraphName(), config, metrics, checker);
    }
    if (definition->isReloadRequired(config)) {
        return factory->reloadDefinition(config.getGraphName(), config, checker);
    }
    return ovms::StatusCode::OK;
}

}  // namespace

extern "C" MEDIAPIPE_RUNTIME_EXPORT void* OVMS_MPFactoryCreate(void* pythonBackend) {
    ovms::initialize_named_loggers_from_default();
    auto* factory = new ovms::MediapipeFactory(static_cast<ovms::PythonBackend*>(pythonBackend));
    return static_cast<void*>(factory);
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT void OVMS_MPFactoryDestroy(void* factoryHandle) {
    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    delete factory;
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT const char* OVMS_MPFactoryGetLastError() {
    return gLastError.c_str();
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPFactoryProcessConfig(void* factoryHandle,
    const ovms::MediapipeGraphConfig* config,
    ovms::MetricProvider* metrics,
    const ovms::ServableNameChecker* checker) {
    if (factoryHandle == nullptr || config == nullptr || metrics == nullptr || checker == nullptr) {
        gLastError = "Invalid arguments for OVMS_MPFactoryProcessConfig";
        return static_cast<int>(ovms::StatusCode::INTERNAL_ERROR);
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    auto status = processConfigInternal(factory, *config, *metrics, *checker);
    gLastError = status.string();
    return static_cast<int>(status.getCode());
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPFactoryCreateExecutor(void* factoryHandle,
    const char* name,
    std::unique_ptr<ovms::MediapipeGraphExecutor>* outExecutor) {
    if (factoryHandle == nullptr || name == nullptr || outExecutor == nullptr) {
        gLastError = "Invalid arguments for OVMS_MPFactoryCreateExecutor";
        return static_cast<int>(ovms::StatusCode::INTERNAL_ERROR);
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    auto status = factory->create(*outExecutor, name);
    gLastError = status.string();
    if (!status.ok()) {
        outExecutor->reset();
        return static_cast<int>(status.getCode());
    }
    return static_cast<int>(ovms::StatusCode::OK);
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPFactoryCreateExecutorHandle(void* factoryHandle,
    const char* name,
    std::unique_ptr<ovms::MediapipeGraphExecutorInterface>* outExecutor) {
    if (factoryHandle == nullptr || name == nullptr || outExecutor == nullptr) {
        gLastError = "Invalid arguments for OVMS_MPFactoryCreateExecutorHandle";
        return static_cast<int>(ovms::StatusCode::INTERNAL_ERROR);
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    auto status = factory->createHandle(*outExecutor, name);
    gLastError = status.string();
    if (!status.ok()) {
        outExecutor->reset();
        return static_cast<int>(status.getCode());
    }
    return static_cast<int>(ovms::StatusCode::OK);
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPFactoryDefinitionExists(void* factoryHandle, const char* name) {
    if (factoryHandle == nullptr || name == nullptr) {
        return 0;
    }
    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    return factory->definitionExists(name) ? 1 : 0;
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPFactoryAliasesConflictExcluding(void* factoryHandle,
    const char* aliases,
    const char* ownGraphName) {
    if (factoryHandle == nullptr || ownGraphName == nullptr) {
        return 0;
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    auto parsedAliases = splitNewlineDelimited(aliases);
    return factory->aliasesConflictExcluding(parsedAliases, ownGraphName) ? 1 : 0;
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT void OVMS_MPFactoryRetireOtherThan(void* factoryHandle, const char* names) {
    if (factoryHandle == nullptr) {
        return;
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    auto parsedNames = splitNewlineDelimited(names);
    std::set<std::string> graphNames(parsedNames.begin(), parsedNames.end());
    factory->retireOtherThan(std::move(graphNames));
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT const char* OVMS_MPFactoryGetNames(void* factoryHandle, int availableOnly) {
    if (factoryHandle == nullptr) {
        gLastError = "";
        return gLastError.c_str();
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    std::vector<std::string> names;
    if (availableOnly != 0) {
        names = factory->getNamesOfAvailableMediapipePipelines();
    } else {
        names = factory->getMediapipePipelinesNames();
    }
    gLastError = joinWithNewlines(names);
    return gLastError.c_str();
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT void* OVMS_MPFactoryFindServableDefinitionByName(void* factoryHandle, const char* name) {
    if (factoryHandle == nullptr || name == nullptr) {
        return nullptr;
    }

    auto* factory = static_cast<ovms::MediapipeFactory*>(factoryHandle);
    return static_cast<void*>(factory->findDefinitionByName(name));
}

extern "C" MEDIAPIPE_RUNTIME_EXPORT int OVMS_MPGraphExportCreateServableConfig(const char* directoryPath,
    const ovms::HFSettingsImpl* hfSettings,
    int writeToFile) {
    if (directoryPath == nullptr || hfSettings == nullptr) {
        gLastError = "Invalid arguments for OVMS_MPGraphExportCreateServableConfig";
        return static_cast<int>(ovms::StatusCode::INTERNAL_ERROR);
    }

    ovms::GraphExport graphExporter;
    auto status = graphExporter.createServableConfig(directoryPath, *hfSettings, writeToFile != 0);
    gLastError = status.string();
    return static_cast<int>(status.getCode());
}
