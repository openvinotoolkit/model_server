//*****************************************************************************
// Copyright 2023-2026 Intel Corporation
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
#include "python_runtime_loader.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
using PythonLibraryHandle = void*;
#elif _WIN32
#include <windows.h>
using PythonLibraryHandle = HMODULE;
#endif

#include "../logging.hpp"
#include "../module.hpp"
#include "../status.hpp"
#include "python_runtime_module_api.hpp"

namespace ovms {

namespace {
using CreatePythonInterpreterModuleFn = Module* (*)();
using ValidatePythonEnvironmentFn = bool (*)(const char** errorMessage);

PythonLibraryHandle pythonRuntimeHandle = nullptr;
CreatePythonInterpreterModuleFn createPythonInterpreterModuleFn = nullptr;
ValidatePythonEnvironmentFn validatePythonEnvironmentFn = nullptr;

void unloadPythonRuntimeLibrary() {
    createPythonInterpreterModuleFn = nullptr;
    validatePythonEnvironmentFn = nullptr;
    if (pythonRuntimeHandle == nullptr) {
        return;
    }
#ifdef __linux__
    dlclose(pythonRuntimeHandle);
#elif _WIN32
    FreeLibrary(pythonRuntimeHandle);
#endif
    pythonRuntimeHandle = nullptr;
}

class PythonRuntimeModuleProxy : public Module, public PythonRuntimeModuleApi {
    std::unique_ptr<Module> runtimeModule;
    PythonRuntimeModuleApi* runtimeApi;

public:
    explicit PythonRuntimeModuleProxy(Module* runtimeModule) :
        runtimeModule(runtimeModule),
        runtimeApi(dynamic_cast<PythonRuntimeModuleApi*>(this->runtimeModule.get())) {
        if (this->runtimeApi == nullptr) {
            throw std::runtime_error("Python runtime module does not implement PythonRuntimeModuleApi");
        }
    }

    ~PythonRuntimeModuleProxy() override {
        runtimeModule.reset();
        unloadPythonRuntimeLibrary();
    }

    Status start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        auto status = runtimeModule->start(config);
        state = runtimeModule->getState();
        return status;
    }

    void shutdown() override {
        runtimeModule->shutdown();
        state = runtimeModule->getState();
    }

    PythonBackend* getPythonBackend() const override {
        return runtimeApi->getPythonBackend();
    }

    bool ownsPythonInterpreter() const override {
        return runtimeApi->ownsPythonInterpreter();
    }

    void releaseGILFromThisThread() const override {
        runtimeApi->releaseGILFromThisThread();
    }
};
}  // namespace

Module* ensurePythonRuntimeLoaded() {
    if (createPythonInterpreterModuleFn != nullptr && validatePythonEnvironmentFn != nullptr) {
        return createPythonInterpreterModuleFn();
    }

    const bool preferInProcessPythonRuntime = []() {
        const char* value = std::getenv("OVMS_TEST_PYTHON_RUNTIME_INPROCESS");
        return value != nullptr && std::string(value) == "1";
    }();

    if (preferInProcessPythonRuntime) {
#ifdef __linux__
        createPythonInterpreterModuleFn = reinterpret_cast<CreatePythonInterpreterModuleFn>(dlsym(RTLD_DEFAULT, "OVMS_createPythonInterpreterModule"));
        validatePythonEnvironmentFn = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(RTLD_DEFAULT, "OVMS_validatePythonEnvironment"));
        if (createPythonInterpreterModuleFn != nullptr && validatePythonEnvironmentFn != nullptr) {
            const char* pythonRuntimeValidationError = nullptr;
            if (!validatePythonEnvironmentFn(&pythonRuntimeValidationError)) {
                SPDLOG_WARN("In-process python runtime environment validation failed. Details: {}",
                    pythonRuntimeValidationError != nullptr ? pythonRuntimeValidationError : "Unknown error");
                createPythonInterpreterModuleFn = nullptr;
                validatePythonEnvironmentFn = nullptr;
                return nullptr;
            }
            SPDLOG_INFO("Python runtime entry points resolved from in-process symbols");
            return new PythonRuntimeModuleProxy(createPythonInterpreterModuleFn());
        }
#elif _WIN32
        HMODULE currentProcess = GetModuleHandleA(nullptr);
        if (currentProcess != nullptr) {
            createPythonInterpreterModuleFn = reinterpret_cast<CreatePythonInterpreterModuleFn>(GetProcAddress(currentProcess, "OVMS_createPythonInterpreterModule"));
            validatePythonEnvironmentFn = reinterpret_cast<ValidatePythonEnvironmentFn>(GetProcAddress(currentProcess, "OVMS_validatePythonEnvironment"));
        }
        if (createPythonInterpreterModuleFn != nullptr && validatePythonEnvironmentFn != nullptr) {
            const char* pythonRuntimeValidationError = nullptr;
            if (!validatePythonEnvironmentFn(&pythonRuntimeValidationError)) {
                SPDLOG_WARN("In-process python runtime environment validation failed. Details: {}",
                    pythonRuntimeValidationError != nullptr ? pythonRuntimeValidationError : "Unknown error");
                createPythonInterpreterModuleFn = nullptr;
                validatePythonEnvironmentFn = nullptr;
                return nullptr;
            }
            SPDLOG_INFO("Python runtime entry points resolved from in-process symbols");
            return new PythonRuntimeModuleProxy(createPythonInterpreterModuleFn());
        }
#endif
    }

#ifdef __linux__
    std::vector<std::string> candidates{
        "libovmspython.so",
        "./libovmspython.so",
        "src/python/libovmspython.so",
        "./src/python/libovmspython.so",
        "bazel-bin/src/python/libovmspython.so",
        "./bazel-bin/src/python/libovmspython.so"};

    for (const auto& candidate : candidates) {
        pythonRuntimeHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (pythonRuntimeHandle != nullptr) {
            break;
        }
    }

    if (pythonRuntimeHandle == nullptr) {
        SPDLOG_WARN("Python runtime library libovmspython.so failed to load: {}", dlerror());
        return nullptr;
    }
    createPythonInterpreterModuleFn = reinterpret_cast<CreatePythonInterpreterModuleFn>(dlsym(pythonRuntimeHandle, "OVMS_createPythonInterpreterModule"));
    if (createPythonInterpreterModuleFn == nullptr) {
        SPDLOG_WARN("Python runtime library libovmspython.so missing symbol OVMS_createPythonInterpreterModule: {}", dlerror());
        dlclose(pythonRuntimeHandle);
        pythonRuntimeHandle = nullptr;
        return nullptr;
    }
    validatePythonEnvironmentFn = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(pythonRuntimeHandle, "OVMS_validatePythonEnvironment"));
    if (validatePythonEnvironmentFn == nullptr) {
        SPDLOG_WARN("Python runtime library libovmspython.so missing symbol OVMS_validatePythonEnvironment: {}", dlerror());
        createPythonInterpreterModuleFn = nullptr;
        dlclose(pythonRuntimeHandle);
        pythonRuntimeHandle = nullptr;
        return nullptr;
    }
#elif _WIN32
    std::vector<std::string> candidates{
        "libovmspython.dll",
        ".\\libovmspython.dll",
        "src\\python\\libovmspython.dll",
        ".\\src\\python\\libovmspython.dll",
        "bazel-bin\\src\\python\\libovmspython.dll",
        ".\\bazel-bin\\src\\python\\libovmspython.dll"};

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
            exeDir + "\\libovmspython.dll",
            exeDir + "\\src\\python\\libovmspython.dll",
            exeDir + "\\..\\src\\python\\libovmspython.dll",
        };

        std::string runfilesRoot = exePath + ".runfiles";
        std::vector<std::string> runfilesCandidates{
            runfilesRoot + "\\src\\python\\libovmspython.dll",
            runfilesRoot + "\\_main\\src\\python\\libovmspython.dll",
            runfilesRoot + "\\model_server\\src\\python\\libovmspython.dll",
        };

        candidates.insert(candidates.end(), executableRelativeCandidates.begin(), executableRelativeCandidates.end());
        candidates.insert(candidates.end(), runfilesCandidates.begin(), runfilesCandidates.end());
    }

    for (const auto& candidate : candidates) {
        pythonRuntimeHandle = LoadLibraryA(candidate.c_str());
        if (pythonRuntimeHandle != nullptr) {
            break;
        }
    }

    if (pythonRuntimeHandle == nullptr) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python runtime library libovmspython.dll failed to load: {} ({})", error, std::system_category().message(error));
        return nullptr;
    }
    createPythonInterpreterModuleFn = reinterpret_cast<CreatePythonInterpreterModuleFn>(GetProcAddress(pythonRuntimeHandle, "OVMS_createPythonInterpreterModule"));
    if (createPythonInterpreterModuleFn == nullptr) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python runtime library libovmspython.dll missing symbol OVMS_createPythonInterpreterModule: {} ({})", error, std::system_category().message(error));
        FreeLibrary(pythonRuntimeHandle);
        pythonRuntimeHandle = nullptr;
        return nullptr;
    }
    validatePythonEnvironmentFn = reinterpret_cast<ValidatePythonEnvironmentFn>(GetProcAddress(pythonRuntimeHandle, "OVMS_validatePythonEnvironment"));
    if (validatePythonEnvironmentFn == nullptr) {
        DWORD error = GetLastError();
        SPDLOG_WARN("Python runtime library libovmspython.dll missing symbol OVMS_validatePythonEnvironment: {} ({})", error, std::system_category().message(error));
        createPythonInterpreterModuleFn = nullptr;
        FreeLibrary(pythonRuntimeHandle);
        pythonRuntimeHandle = nullptr;
        return nullptr;
    }
#endif

    const char* pythonRuntimeValidationError = nullptr;
    if (!validatePythonEnvironmentFn(&pythonRuntimeValidationError)) {
        SPDLOG_WARN("Python runtime environment validation failed. Ensure Python dependencies and PYTHONPATH are configured. Details: {}",
            pythonRuntimeValidationError != nullptr ? pythonRuntimeValidationError : "Unknown error");
        createPythonInterpreterModuleFn = nullptr;
        validatePythonEnvironmentFn = nullptr;
#ifdef __linux__
        dlclose(pythonRuntimeHandle);
#elif _WIN32
        FreeLibrary(pythonRuntimeHandle);
#endif
        pythonRuntimeHandle = nullptr;
        return nullptr;
    }

    SPDLOG_INFO("Python runtime library loaded successfully");
    return new PythonRuntimeModuleProxy(createPythonInterpreterModuleFn());
}

}  // namespace ovms
