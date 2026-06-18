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
#include "kfs_python_tensor_bridge.hpp"

#include <memory>
#include <vector>
#include <string>

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

using RegisterPythonCalculatorsFn = void (*)();
using GetKfsPyTensorBridgeVTableFn = const KfsPyTensorBridgeVTable* (*)();

static PluginHandle pythonCalculatorsHandle = nullptr;
static RegisterPythonCalculatorsFn registerPythonCalculatorsFn = nullptr;

}  // namespace

bool loadPythonCalculatorsPlugin() {
    if (registerPythonCalculatorsFn != nullptr) {
        SPDLOG_DEBUG("Python calculators plugin already loaded");
        return true;
    }

#ifdef __linux__
    std::vector<std::string> candidates{
        "libpython_calculators.so",
        "./libpython_calculators.so",
        "src/python/libpython_calculators.so",
        "./src/python/libpython_calculators.so",
        "bazel-bin/src/python/libpython_calculators.so",
        "./bazel-bin/src/python/libpython_calculators.so"};

    for (const auto& candidate : candidates) {
        pythonCalculatorsHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (pythonCalculatorsHandle != nullptr) {
            break;
        }
    }

    if (pythonCalculatorsHandle == nullptr) {
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so failed to load: {}. "
                    "MediaPipe Python calculators will not be available.",
            dlerror());
        return false;
    }

    registerPythonCalculatorsFn = reinterpret_cast<RegisterPythonCalculatorsFn>(
        dlsym(pythonCalculatorsHandle, "registerPythonCalculators"));
    if (registerPythonCalculatorsFn == nullptr) {
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so missing symbol registerPythonCalculators: {}. "
                    "MediaPipe Python calculators will not be available.",
            dlerror());
        dlclose(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }

#elif _WIN32
    std::vector<std::string> candidates{
        "libpython_calculators.dll",
        ".\\libpython_calculators.dll",
        "src\\python\\libpython_calculators.dll",
        ".\\src\\python\\libpython_calculators.dll",
        "bazel-bin\\src\\python\\libpython_calculators.dll",
        ".\\bazel-bin\\src\\python\\libpython_calculators.dll"};

    char executablePath[MAX_PATH] = {0};
    DWORD executablePathLength = GetModuleFileNameA(nullptr, executablePath, MAX_PATH);
    if (executablePathLength > 0 && executablePathLength < MAX_PATH) {
        std::string exePath(executablePath, executablePathLength);
        std::string exeDir = ".";
        size_t separatorPos = exePath.find_last_of("\\/");
        if (separatorPos != std::string::npos) {
            exeDir = exePath.substr(0, separatorPos);
        }

        std::vector<std::string> executableRelativeCandidates{
            exeDir + "\\libpython_calculators.dll",
            exeDir + "\\src\\python\\libpython_calculators.dll",
            exeDir + "\\..\\src\\python\\libpython_calculators.dll",
        };

        std::string runfilesRoot = exePath + ".runfiles";
        std::vector<std::string> runfilesCandidates{
            runfilesRoot + "\\src\\python\\libpython_calculators.dll",
            runfilesRoot + "\\_main\\src\\python\\libpython_calculators.dll",
            runfilesRoot + "\\model_server\\src\\python\\libpython_calculators.dll",
        };

        candidates.insert(candidates.end(), executableRelativeCandidates.begin(), executableRelativeCandidates.end());
        candidates.insert(candidates.end(), runfilesCandidates.begin(), runfilesCandidates.end());
    }

    for (const auto& candidate : candidates) {
        pythonCalculatorsHandle = LoadLibraryA(candidate.c_str());
        if (pythonCalculatorsHandle != nullptr) {
            break;
        }
    }

    if (pythonCalculatorsHandle == nullptr) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll failed to load: {} ({}). "
                    "MediaPipe Python calculators will not be available.",
            error, std::system_category().message(error));
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

    // Call the registration function. This function executes the static initializers
    // from the plugin, which register the MediaPipe calculators and node initializers
    registerPythonCalculatorsFn();

    SPDLOG_INFO("Python calculators plugin loaded successfully");
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
            setKfsPyTensorBridgeVTable(vtable);
            SPDLOG_INFO("KFS Python tensor bridge activated from python calculators plugin");
        }
    }
    return true;
}

}  // namespace ovms
