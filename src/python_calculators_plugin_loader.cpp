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

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <cerrno>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
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
static PluginHandle pythonCalculatorsRuntimeHandle = nullptr;

#ifdef __linux__
std::string safeDlerror() {
    if (const char* error = dlerror(); error != nullptr) {
        return std::string(error);
    }
    return std::string("dlerror returned null");
}

bool ensurePythonCalculatorsRuntimeLoaded() {
    if (pythonCalculatorsRuntimeHandle != nullptr) {
        return true;
    }

    std::vector<std::string> candidates{
        "libovms_mediapipe_runtime_shared.so",
        "/ovms/lib/libovms_mediapipe_runtime_shared.so",
        "./libovms_mediapipe_runtime_shared.so",
        "src/libovms_mediapipe_runtime_shared.so",
        "./src/libovms_mediapipe_runtime_shared.so",
        "bazel-bin/src/libovms_mediapipe_runtime_shared.so",
        "./bazel-bin/src/libovms_mediapipe_runtime_shared.so"};

    for (const auto& candidate : candidates) {
        pythonCalculatorsRuntimeHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (pythonCalculatorsRuntimeHandle != nullptr) {
            SPDLOG_INFO("Python calculators runtime loaded from: {}", candidate);
            return true;
        }
    }

    SPDLOG_DEBUG("Python calculators runtime library not preloaded: {}", safeDlerror());
    return false;
}
#endif

#ifdef __linux__
// Weak reference allows using in-process bridge when linked into the binary
// (e.g. ovms_test) without requiring -rdynamic for RTLD_DEFAULT lookups.
extern "C" const KfsPyTensorBridgeVTable* OVMS_getKfsPyTensorBridgeVTable() __attribute__((weak));

bool probePluginLoadInChildProcess(const std::string& pluginPath) {
    // Some plugin failures are process-fatal (for example LOG(FATAL)/abort in
    // static initializers, as seen in MediaPipe type-map registration conflicts).
    // Probe in a short-lived child process so the parent OVMS process can
    // continue startup and gracefully disable Python calculators instead of
    // terminating the whole server.
    pid_t probePid = fork();
    if (probePid < 0) {
        SPDLOG_WARN("Could not start Python calculators plugin probe for {}. Will try loading directly.", pluginPath);
        return true;
    }

    if (probePid == 0) {
        // Ensure child process can find shared libraries like libmediapipe_framework.so
        // Add /ovms/lib to LD_LIBRARY_PATH for plugin loading
        const char* existing_ld = std::getenv("LD_LIBRARY_PATH");
        std::string ld_lib_path = "/ovms/lib";
        if (existing_ld != nullptr && existing_ld[0] != '\0') {
            ld_lib_path = ld_lib_path + ":" + existing_ld;
        }
        setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);

        // Keep process symbols globally visible in the probe process as well.
        // This mirrors the parent behavior and helps resolve plugin dependencies
        // that expect OVMS symbols to be available from the main executable.
        void* probeMainSymbols = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
        if (probeMainSymbols == nullptr) {
            const char* dlopen_main_error = dlerror();
            if (dlopen_main_error) {
                fprintf(stderr, "[PROBE] dlopen(NULL) failed: %s\n", dlopen_main_error);
            }
        }

        // Use RTLD_GLOBAL to ensure plugin symbols are available to the main binary.
        // Both main binary and shared MediaPipe library have protobuf, but since
        // the main binary no longer links to the shared library directly, protobuf
        // conflicts are avoided.
        void* probeHandle = dlopen(pluginPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (probeHandle != nullptr) {
            dlclose(probeHandle);
            _exit(0);
        }
        // Probe failed - log details before exiting
        const char* dlopen_error = dlerror();
        if (dlopen_error) {
            // Write to stderr since this is a child process
            fprintf(stderr, "[PROBE] dlopen failed for %s: %s\n", pluginPath.c_str(), dlopen_error);
        }
        _exit(1);
    }

    int waitStatus = 0;
    while (waitpid(probePid, &waitStatus, 0) == -1) {
        if (errno == EINTR) {
            continue;
        }
        SPDLOG_WARN("Could not wait for Python calculators plugin probe for {}. Will try loading directly.", pluginPath);
        return true;
    }

    if (WIFEXITED(waitStatus) && WEXITSTATUS(waitStatus) == 0) {
        return true;
    }

    if (WIFSIGNALED(waitStatus)) {
        SPDLOG_WARN("Skipping Python calculators plugin candidate {} because probe process terminated with signal {}.", pluginPath, WTERMSIG(waitStatus));
    } else if (WIFEXITED(waitStatus)) {
        SPDLOG_DEBUG("Python calculators plugin probe failed for {} with exit code {}.", pluginPath, WEXITSTATUS(waitStatus));
    } else {
        SPDLOG_DEBUG("Python calculators plugin probe failed for {}.", pluginPath);
    }
    return false;
}
#endif

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

std::string toAbsolutePath(const std::string& candidate) {
    char absPath[MAX_PATH] = {0};
    DWORD pathLen = GetFullPathNameA(candidate.c_str(), MAX_PATH, absPath, nullptr);
    if (pathLen == 0 || pathLen >= MAX_PATH) {
        return candidate;
    }
    return std::string(absPath, pathLen);
}

void logLikelyMissingWindowsDependencies() {
    const std::vector<std::string> likelyDependencies = {
        "libpython_calculators.dll",          // The plugin itself
        "libmediapipe_framework_shared.dll",  // Shared MediaPipe library (critical for type-id resolution)
        "libovmspython.dll",                  // Python runtime support
        "python312.dll",                      // Python interpreter
        "openvino.dll",                       // OpenVINO core
        "openvino_genai.dll",                 // OpenVINO GenAI
    };
    for (const auto& dependency : likelyDependencies) {
        char resolvedPath[MAX_PATH] = {0};
        DWORD pathLen = SearchPathA(nullptr, dependency.c_str(), nullptr, MAX_PATH, resolvedPath, nullptr);
        if (pathLen == 0 || pathLen >= MAX_PATH) {
            SPDLOG_WARN("Python calculators plugin dependency not found in DLL search path: {}", dependency);
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

#ifdef __linux__
    ensurePythonCalculatorsRuntimeLoaded();

    auto* alreadyLoadedRegisterFn = reinterpret_cast<RegisterPythonCalculatorsFn>(dlsym(RTLD_DEFAULT, "registerPythonCalculators"));
    if (alreadyLoadedRegisterFn != nullptr) {
        registerPythonCalculatorsFn = alreadyLoadedRegisterFn;

        auto* getKfsBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(dlsym(RTLD_DEFAULT, "OVMS_getKfsPyTensorBridgeVTable"));
        if (getKfsBridgeFn != nullptr) {
            if (auto* vtable = getKfsBridgeFn(); vtable != nullptr) {
                setKfsPyTensorBridgeVTable(vtable);
                SPDLOG_INFO("KFS Python tensor bridge activated from already loaded python calculators plugin");
            }
        }

        SPDLOG_INFO("Python calculators plugin already present in the process, skipping dlopen");
        return true;
    }

    std::vector<std::string> candidates{
        "libpython_calculators.so",
        "./libpython_calculators.so",
        "/ovms/lib/libpython_calculators.so",
        "src/python/libpython_calculators.so",
        "./src/python/libpython_calculators.so",
        "bazel-bin/src/python/libpython_calculators.so",
        "./bazel-bin/src/python/libpython_calculators.so"};

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr && testSrcDir[0] != '\0') {
        const std::string srcDir(testSrcDir);
        const char* testWorkspace = std::getenv("TEST_WORKSPACE");
        if (testWorkspace != nullptr && testWorkspace[0] != '\0') {
            candidates.emplace_back(srcDir + "/" + testWorkspace + "/src/python/libpython_calculators.so");
            candidates.emplace_back(srcDir + "/" + testWorkspace + "/bazel-bin/src/python/libpython_calculators.so");
        }
        candidates.emplace_back(srcDir + "/_main/src/python/libpython_calculators.so");
        candidates.emplace_back(srcDir + "/_main/bazel-bin/src/python/libpython_calculators.so");
        candidates.emplace_back(srcDir + "/model_server/src/python/libpython_calculators.so");
        candidates.emplace_back(srcDir + "/model_server/bazel-bin/src/python/libpython_calculators.so");
    }

    try {
        const auto testBinaryPath = std::filesystem::canonical("/proc/self/exe");
        const auto runfilesDir = testBinaryPath.string() + ".runfiles";
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "ovms/src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "_main/src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "model_server/src/python/libpython_calculators.so");
    } catch (...) {
    }

    // CRITICAL: Expose main process symbols to plugin before loading it.
    // The plugin will link to a shared MediaPipe library that contains undefined
    // OVMS symbols (from geti calculators in the external MediaPipe fork).
    // By calling dlopen(NULL, RTLD_NOW | RTLD_GLOBAL) on the main process, we make all
    // OVMS and geti symbols available via the main process's symbol table.
    // When the plugin's dlopen tries to resolve undefined symbols, it will find
    // them in the main process instead of failing with "undefined symbol" errors.
    // This allows the plugin to load even though the shared MediaPipe library
    // has forward references to OVMS code that's only available in the main binary.
    void* mainProcessSymbols = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
    if (mainProcessSymbols == nullptr) {
        const std::string errorDetails = safeDlerror();
        SPDLOG_WARN("Failed to expose main process symbols: {}. "
                    "Plugin loading may fail if shared libraries have unresolved symbols.",
            errorDetails);
    }

    for (const auto& candidate : candidates) {
        // Try loading candidate in a child process first. If probe indicates a
        // crash/non-zero exit, skip direct dlopen in the main process to avoid
        // bringing down OVMS during optional plugin initialization.
        if (!probePluginLoadInChildProcess(candidate)) {
            continue;
        }

        // Use RTLD_GLOBAL to share plugin symbols with the main binary.
        // This allows the main binary to call plugin functions like registerPythonCalculators.
        SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", candidate);

        pythonCalculatorsHandle = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);

        if (pythonCalculatorsHandle != nullptr) {
            SPDLOG_INFO("Successfully loaded Python calculators plugin from: {}", candidate);
            break;
        } else {
            const std::string errorDetails = safeDlerror();
            SPDLOG_DEBUG("Failed to load Python calculators plugin candidate {}: {}", candidate, errorDetails);
        }
    }

    if (pythonCalculatorsHandle == nullptr) {
        const std::string errorDetails = safeDlerror();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so failed to load: {}. "
                    "MediaPipe Python calculators will not be available.",
            errorDetails);
        return false;
    }

    registerPythonCalculatorsFn = reinterpret_cast<RegisterPythonCalculatorsFn>(
        dlsym(pythonCalculatorsHandle, "registerPythonCalculators"));
    if (registerPythonCalculatorsFn == nullptr) {
        const std::string errorDetails = safeDlerror();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so missing symbol registerPythonCalculators: {}. "
                    "MediaPipe Python calculators will not be available.",
            errorDetails);
        dlclose(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }

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
                    setKfsPyTensorBridgeVTable(vtable);
                    SPDLOG_INFO("KFS Python tensor bridge activated from current process exports");
                }
            }
        }
    }

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

    DWORD lastLoadError = ERROR_SUCCESS;
    for (const auto& candidate : candidates) {
        SetLastError(ERROR_SUCCESS);

        // On Windows, unlike dlopen(NULL, RTLD_GLOBAL) on Linux, the operating system
        // automatically makes all symbols from the current process available to any DLL
        // that LoadLibraryA loads. The DLL's import table is resolved against:
        // 1. The DLL itself
        // 2. DLLs it explicitly links to
        // 3. The main executable's exported symbols
        // 4. System libraries
        // This automatic symbol resolution means the plugin will find undefined OVMS
        // symbols from the shared MediaPipe library without requiring an explicit call.
        // No dlopen(NULL, RTLD_GLOBAL) equivalent is needed on Windows.

        SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", toAbsolutePath(candidate));
        pythonCalculatorsHandle = LoadLibraryA(candidate.c_str());
        if (pythonCalculatorsHandle != nullptr) {
            SPDLOG_INFO("Python calculators plugin loaded from candidate: {}", toAbsolutePath(candidate));
            break;
        }

        lastLoadError = GetLastError();
        const bool candidateExists = std::filesystem::exists(candidate);
        SPDLOG_DEBUG(
            "Failed to load python calculators candidate: {} (absolute: {}, exists: {}), error: {} ({})",
            candidate,
            toAbsolutePath(candidate),
            candidateExists,
            lastLoadError,
            formatWindowsErrorMessage(lastLoadError));
    }

    if (pythonCalculatorsHandle == nullptr) {
        DWORD error = lastLoadError != ERROR_SUCCESS ? lastLoadError : GetLastError();
        SPDLOG_WARN("Python calculators plugin candidates attempted: {}", candidates.size());
        for (const auto& candidate : candidates) {
            SPDLOG_WARN("  candidate: {}", toAbsolutePath(candidate));
        }
        logLikelyMissingWindowsDependencies();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.dll failed to load: {} ({}). "
                    "Possible causes: missing dependency (libmediapipe_framework_shared.dll, libovmspython.dll), "
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
