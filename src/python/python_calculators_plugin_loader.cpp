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

#include "python_calculators_plugin_loader.hpp"
#include "python_calculators_plugin_api.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#elif _WIN32
#include <windows.h>
#include <system_error>
#endif

#include "src/logging.hpp"

namespace ovms {

namespace {

#ifdef __linux__
using PluginHandle = void*;
#elif _WIN32
using PluginHandle = HMODULE;
#endif

using GetPythonCalculatorsPluginApiFn = const PythonCalculatorsPluginApi* (*)();

static PluginHandle pythonCalculatorsHandle = nullptr;
static const PythonCalculatorsPluginApi* pythonCalculatorsPluginApi = nullptr;

bool activatePluginApi(GetPythonCalculatorsPluginApiFn getApi, const char* source) {
    if (getApi == nullptr) {
        return false;
    }
    const PythonCalculatorsPluginApi* api = getApi();
    if (api == nullptr || api->abiVersion != PYTHON_CALCULATORS_PLUGIN_ABI_VERSION) {
        SPDLOG_ERROR("Python calculators plugin API from {} has incompatible ABI version: {}, expected: {}",
            source, api == nullptr ? 0u : api->abiVersion, PYTHON_CALCULATORS_PLUGIN_ABI_VERSION);
        return false;
    }
    pythonCalculatorsPluginApi = api;
    SPDLOG_TRACE("Python calculators plugin API activated from {}", source);
    return true;
}

bool activateInProcessPluginApi() {
#ifdef __linux__
    // Global scope lookup: the loader is linked into both ovms and libovmspython, and the plugin is dlopened RTLD_GLOBAL.
    return activatePluginApi(reinterpret_cast<GetPythonCalculatorsPluginApiFn>(
                                 dlsym(RTLD_DEFAULT, "OVMS_getPythonCalculatorsPluginApi")),
        "process global scope");
#elif _WIN32
    // The loader is linked into both ovms.exe and libovmspython.dll; either may have loaded the plugin.
    for (const char* moduleName : {static_cast<const char*>(nullptr), "libpython_calculators.dll"}) {
        HMODULE module = GetModuleHandleA(moduleName);
        if (module != nullptr &&
            activatePluginApi(reinterpret_cast<GetPythonCalculatorsPluginApiFn>(
                                  GetProcAddress(module, "OVMS_getPythonCalculatorsPluginApi")),
                moduleName == nullptr ? "current process exports" : moduleName)) {
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

#ifdef _WIN32
std::string formatWindowsErrorMessage(DWORD errorCode) {
    LPSTR buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD length = FormatMessageA(
        flags,
        nullptr,
        errorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);
    if (length == 0 || buffer == nullptr) {
        return "Unknown Windows error";
    }
    std::string message(buffer, length);
    LocalFree(buffer);
    while (!message.empty() && (message.back() == '\r' || message.back() == '\n' || message.back() == ' ' || message.back() == '\t')) {
        message.pop_back();
    }
    return message;
}

void logLikelyMissingWindowsDependencies() {
    const std::vector<std::string> likelyDependencies = {
        "libpython_calculators.dll",  // The plugin itself
        "libovmspython.dll",          // Python runtime support
        "python312.dll",              // Python interpreter
        "openvino.dll",               // OpenVINO core
        "openvino_genai.dll",         // OpenVINO GenAI
    };
    for (const auto& dependency : likelyDependencies) {
        char resolvedPath[MAX_PATH] = {0};
        DWORD pathLen = SearchPathA(nullptr, dependency.c_str(), nullptr, MAX_PATH, resolvedPath, nullptr);
        if (pathLen == 0 || pathLen >= MAX_PATH) {
            SPDLOG_DEBUG("Python calculators plugin dependency not found in DLL search path: {}", dependency);
        } else {
            SPDLOG_DEBUG("Python calculators plugin dependency resolved: {} -> {}", dependency, resolvedPath);
        }
    }
}

#endif

}  // namespace

const PythonCalculatorsPluginApi* getPythonCalculatorsPluginApi() {
    if (pythonCalculatorsPluginApi == nullptr) {
        // Covers binaries linking the implementation in-process (e.g. ovms_test) before the plugin loader runs.
        activateInProcessPluginApi();
    }
    return pythonCalculatorsPluginApi;
}

bool loadPythonCalculatorsPlugin() {
    if (pythonCalculatorsPluginApi != nullptr) {
        SPDLOG_DEBUG("Python calculators plugin already loaded");
        return true;
    }

    if (activateInProcessPluginApi()) {
        SPDLOG_TRACE("Python calculators plugin already present in the process, skipping plugin load");
        return true;
    }

#ifdef __linux__
    constexpr const char* pluginName = "libpython_calculators.so";
    SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", pluginName);
    pythonCalculatorsHandle = dlopen(pluginName, RTLD_NOW | RTLD_GLOBAL);
    if (pythonCalculatorsHandle == nullptr) {
        const char* errorMessage = dlerror();
        SPDLOG_WARN("Python calculators plugin {} failed to load: {}. MediaPipe Python calculators will not be available.",
            pluginName, errorMessage != nullptr ? errorMessage : "unknown error");
        return false;
    }

    if (!activatePluginApi(reinterpret_cast<GetPythonCalculatorsPluginApiFn>(
                               dlsym(pythonCalculatorsHandle, "OVMS_getPythonCalculatorsPluginApi")),
            pluginName)) {
        const char* errorMessage = dlerror();
        SPDLOG_WARN("Python calculators plugin {} does not provide a usable OVMS_getPythonCalculatorsPluginApi: {}. MediaPipe Python calculators will not be available.",
            pluginName, errorMessage != nullptr ? errorMessage : "unknown error");
        dlclose(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }

#elif _WIN32
    constexpr const char* pluginName = "libpython_calculators.dll";
    SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", pluginName);
    pythonCalculatorsHandle = LoadLibraryA(pluginName);

    if (pythonCalculatorsHandle == nullptr) {
        DWORD error = GetLastError();
        logLikelyMissingWindowsDependencies();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll failed to load: {} ({}). "
                    "Possible causes: missing dependency (libovmspython.dll), "
                    "incompatible architecture (32-bit vs 64-bit), or export symbol conflict. "
                    "MediaPipe Python calculators will not be available.",
            error, formatWindowsErrorMessage(error));
        return false;
    }

    if (!activatePluginApi(reinterpret_cast<GetPythonCalculatorsPluginApiFn>(
                               GetProcAddress(pythonCalculatorsHandle, "OVMS_getPythonCalculatorsPluginApi")),
            "libpython_calculators.dll")) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll does not provide a usable OVMS_getPythonCalculatorsPluginApi: {} ({}). "
                    "MediaPipe Python calculators will not be available.",
            error, std::system_category().message(error));
        FreeLibrary(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }
#endif

    SPDLOG_TRACE("Python calculators plugin loaded successfully");
    return true;
}

}  // namespace ovms
