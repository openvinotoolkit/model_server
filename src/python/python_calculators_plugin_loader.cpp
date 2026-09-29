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
#include "src/kfs_python_tensor_bridge.hpp"

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

#ifdef __linux__
extern "C" void registerPythonCalculators() __attribute__((weak));
extern "C" const KfsPyTensorBridgeVTable* OVMS_getKfsPyTensorBridgeVTable() __attribute__((weak));
#endif

namespace {

#ifdef __linux__
using PluginHandle = void*;
#elif _WIN32
using PluginHandle = HMODULE;
#endif

using RegisterPythonCalculatorsFn = void (*)();
using GetKfsPyTensorBridgeVTableFn = const KfsPyTensorBridgeVTable* (*)();

static PluginHandle pythonCalculatorsHandle = nullptr;
static RegisterPythonCalculatorsFn registerPythonCalculatorsFn = nullptr;

void activateKfsBridge(const KfsPyTensorBridgeVTable* vtable, const char* source) {
    if (vtable == nullptr) {
        return;
    }

    setKfsPyTensorBridgeVTable(vtable);
    SPDLOG_TRACE("KFS Python tensor bridge activated from {}", source);
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
        "libpython_calculators.dll",          // The plugin itself
        "ovms_mediapipe_runtime_shared.dll",  // OVMS MediaPipe runtime integration library
        "libovmspython.dll",                  // Python runtime support
        "python312.dll",                      // Python interpreter
        "openvino.dll",                       // OpenVINO core
        "openvino_genai.dll",                 // OpenVINO GenAI
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

bool loadPythonCalculatorsPlugin() {
    if (registerPythonCalculatorsFn != nullptr) {
        SPDLOG_DEBUG("Python calculators plugin already loaded");
        return true;
    }

    bool hasInProcessKfsBridge = getKfsPyTensorBridgeVTable() != nullptr;

#ifdef __linux__
    if (registerPythonCalculators != nullptr) {
        registerPythonCalculatorsFn = registerPythonCalculators;
        if (getKfsPyTensorBridgeVTable() == nullptr && OVMS_getKfsPyTensorBridgeVTable != nullptr) {
            if (auto* vtable = OVMS_getKfsPyTensorBridgeVTable(); vtable != nullptr) {
                activateKfsBridge(vtable, "in-process weak symbol");
            }
        }
        if (getKfsPyTensorBridgeVTable() != nullptr) {
            SPDLOG_TRACE("Python calculators plugin entry point already linked in-process, skipping plugin dlopen");
            return true;
        }
        SPDLOG_WARN("Python calculators registration is available in-process but KFS Python tensor bridge is not initialized. Continuing without plugin dlopen; OVMS_PY_TENSOR bridge paths may be unavailable.");
        return true;
    }

    auto* alreadyLoadedRegisterFn = reinterpret_cast<RegisterPythonCalculatorsFn>(dlsym(RTLD_DEFAULT, "registerPythonCalculators"));
    if (alreadyLoadedRegisterFn != nullptr) {
        registerPythonCalculatorsFn = alreadyLoadedRegisterFn;

        auto* getKfsBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(dlsym(RTLD_DEFAULT, "OVMS_getKfsPyTensorBridgeVTable"));
        if (getKfsBridgeFn != nullptr) {
            if (auto* vtable = getKfsBridgeFn(); vtable != nullptr) {
                activateKfsBridge(vtable, "already loaded python calculators plugin");
            }
        }

        if (getKfsPyTensorBridgeVTable() != nullptr) {
            SPDLOG_TRACE("Python calculators plugin already present in the process, skipping dlopen");
            return true;
        }
        SPDLOG_WARN("Python calculators entry point is present in the process but KFS Python tensor bridge is not initialized. Continuing without plugin dlopen; OVMS_PY_TENSOR bridge paths may be unavailable.");
        return true;
    }

    // In runtime-separation mode, calculator registrations are expected to be
    // owned by libovms_mediapipe_runtime_shared.so. Eagerly dlopen-ing
    // libpython_calculators.so here can register MediaPipe framework handlers
    // twice (plugin first, runtime-shared second).
    SPDLOG_INFO("Skipping explicit libpython_calculators.so dlopen. "
                "Python calculators are expected from runtime-shared ownership.");
    return true;

#elif _WIN32
    // Windows equivalent of Linux RTLD_DEFAULT lookup:
    // if OVMS_getKfsPyTensorBridgeVTable is already linked into the current
    // process (e.g. ovms_test with python bridge runtime), use it directly
    // and avoid loading libpython_calculators.dll.
    if (getKfsPyTensorBridgeVTable() == nullptr) {
        HMODULE currentProcessModule = GetModuleHandleA(nullptr);
        if (currentProcessModule != nullptr) {
            auto* inProcessBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(
                GetProcAddress(currentProcessModule, "OVMS_getKfsPyTensorBridgeVTable"));
            if (inProcessBridgeFn != nullptr) {
                if (auto* vtable = inProcessBridgeFn(); vtable != nullptr) {
                    activateKfsBridge(vtable, "current process exports");
                    hasInProcessKfsBridge = true;
                }
            }
        }
    } else {
        hasInProcessKfsBridge = true;
    }

    constexpr const char* pluginName = "libpython_calculators.dll";
    SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", pluginName);
    pythonCalculatorsHandle = LoadLibraryA(pluginName);

    if (pythonCalculatorsHandle == nullptr) {
        DWORD error = GetLastError();
        logLikelyMissingWindowsDependencies();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll failed to load: {} ({}). "
                    "Possible causes: missing dependency (ovms_mediapipe_runtime_shared.dll, libovmspython.dll), "
                    "incompatible architecture (32-bit vs 64-bit), or export symbol conflict. "
                    "MediaPipe Python calculators will not be available.",
            error, formatWindowsErrorMessage(error));
        return false;
    }

    registerPythonCalculatorsFn = reinterpret_cast<RegisterPythonCalculatorsFn>(
        GetProcAddress(pythonCalculatorsHandle, "registerPythonCalculators"));
    if (registerPythonCalculatorsFn == nullptr) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll missing symbol registerPythonCalculators: {} ({}). "
                    "MediaPipe Python calculators will not be available.",
            error, std::system_category().message(error));
        FreeLibrary(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }
#endif

    registerPythonCalculatorsFn();

    SPDLOG_TRACE("Python calculators plugin loaded successfully");
    // Also load the KFS Python tensor bridge vtable from the same plugin.
    // This enables OVMS_PY_TENSOR deserialization/serialization in the KFS
    // graph executor without linking pybind11 into the main binary.
#ifdef __linux__
    auto getKfsBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(
        dlsym(pythonCalculatorsHandle, "OVMS_getKfsPyTensorBridgeVTable"));
#elif _WIN32
    auto getKfsBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(
        GetProcAddress(pythonCalculatorsHandle, "OVMS_getKfsPyTensorBridgeVTable"));
#endif
    if (getKfsBridgeFn != nullptr) {
        auto* vtable = getKfsBridgeFn();
        if (vtable != nullptr) {
            if (hasInProcessKfsBridge) {
                SPDLOG_TRACE("KFS Python tensor bridge already active from in-process exports; keeping existing bridge and skipping plugin bridge override");
            } else {
                activateKfsBridge(vtable, "python calculators plugin");
            }
        }
    }
    return true;
}

}  // namespace ovms
