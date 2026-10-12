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

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <cerrno>
#include <csignal>
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

#ifdef __linux__
int pythonPluginDlopenFlags() {
    int flags = RTLD_LAZY | RTLD_GLOBAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    return flags;
}

std::string safeDlerror() {
    if (const char* error = dlerror(); error != nullptr) {
        return std::string(error);
    }
    return std::string("dlerror returned null");
}

// Weak reference allows using in-process bridge when linked into the binary
// (e.g. ovms_test) without requiring -rdynamic for RTLD_DEFAULT lookups.
bool probePluginLoadInChildProcess(const std::string& pluginPath) {
    // Some plugin failures are process-fatal (for example LOG(FATAL)/abort in
    // static initializers, as seen in MediaPipe type-map registration conflicts).
    // Probe in a short-lived child process so the parent OVMS process can
    // continue startup and gracefully disable Python calculators instead of
    // terminating the whole server.
    pid_t probePid = fork();
    if (probePid < 0) {
        SPDLOG_DEBUG("Could not start Python calculators plugin probe for {}. Will try loading directly.", pluginPath);
        return true;
    }

    if (probePid == 0) {
        // Ensure child process can find shared libraries like libmediapipe_framework.so
        // Add /ovms/lib to LD_LIBRARY_PATH for plugin loading
        const char* existingLd = std::getenv("LD_LIBRARY_PATH");
        std::string ldLibPath = "/ovms/lib";
        if (existingLd != nullptr && existingLd[0] != '\0') {
            ldLibPath = ldLibPath + ":" + existingLd;
        }
        setenv("LD_LIBRARY_PATH", ldLibPath.c_str(), 1);

        // Keep process symbols globally visible in the probe process as well.
        // This mirrors the parent behavior and helps resolve plugin dependencies
        // that expect OVMS symbols to be available from the main executable.
        void* probeMainSymbols = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
        if (probeMainSymbols == nullptr) {
            const char* dlopenMainError = dlerror();
            if (dlopenMainError) {
                fprintf(stderr, "[PROBE] dlopen(NULL) failed: %s\n", dlopenMainError);
            }
        }

        // Use RTLD_GLOBAL to ensure plugin symbols are available to the main binary.
        // Both main binary and shared MediaPipe library have protobuf, but since
        // the main binary no longer links to the shared library directly, protobuf
        // conflicts are avoided.
        void* probeHandle = dlopen(pluginPath.c_str(), pythonPluginDlopenFlags());
        if (probeHandle != nullptr) {
            dlclose(probeHandle);
            _exit(0);
        }
        // Probe failed - log details before exiting
        const char* dlopenError = dlerror();
        if (dlopenError) {
            // Write to stderr since this is a child process
            fprintf(stderr, "[PROBE] dlopen failed for %s: %s\n", pluginPath.c_str(), dlopenError);
        }
        _exit(1);
    }

    // Bound the EINTR retry count so a pathological signal storm can never
    // trap the parent OVMS process in an infinite wait loop. A dlopen probe
    // is expected to finish in milliseconds; 100 retries is orders of
    // magnitude beyond any realistic signal delivery rate.
    constexpr int kMaxWaitRetries = 100;
    int waitStatus = 0;
    int retries = 0;
    while (waitpid(probePid, &waitStatus, 0) == -1) {
        if (errno == EINTR && ++retries < kMaxWaitRetries) {
            continue;
        }
        SPDLOG_DEBUG("Could not wait for Python calculators plugin probe for {} (errno={}, retries={}). Will try loading directly.",
            pluginPath, errno, retries);
        // Best-effort cleanup so the probe child does not linger as a zombie.
        kill(probePid, SIGKILL);
        waitpid(probePid, nullptr, WNOHANG);
        return true;
    }

    if (WIFEXITED(waitStatus) && WEXITSTATUS(waitStatus) == 0) {
        return true;
    }

    if (WIFSIGNALED(waitStatus)) {
        SPDLOG_DEBUG("Skipping Python calculators plugin candidate {} because probe process terminated with signal {}.", pluginPath, WTERMSIG(waitStatus));
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

#ifdef __linux__
    if (activateInProcessPluginApi()) {
        SPDLOG_TRACE("Python calculators implementation linked in-process, skipping plugin dlopen");
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
        candidates.insert(candidates.begin(), (testBinaryPath.parent_path().parent_path() / "lib/libpython_calculators.so").string());
        candidates.emplace_back((testBinaryPath.parent_path() / "python/libpython_calculators.so").string());
        const auto runfilesDir = testBinaryPath.string() + ".runfiles";
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "ovms/src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "_main/src/python/libpython_calculators.so");
        candidates.emplace_back(std::filesystem::path(runfilesDir) / "model_server/src/python/libpython_calculators.so");
    } catch (...) {
    }

    for (const auto& candidate : candidates) {
        // Try loading candidate in a child process first. If probe indicates a
        // crash/non-zero exit, skip direct dlopen in the main process to avoid
        // bringing down OVMS during optional plugin initialization.
        if (!probePluginLoadInChildProcess(candidate)) {
            continue;
        }

        SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", candidate);

        pythonCalculatorsHandle = dlopen(candidate.c_str(), pythonPluginDlopenFlags());

        if (pythonCalculatorsHandle != nullptr) {
            SPDLOG_TRACE("Successfully loaded Python calculators plugin from: {}", candidate);
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

    if (!activatePluginApi(reinterpret_cast<GetPythonCalculatorsPluginApiFn>(
                               dlsym(pythonCalculatorsHandle, "OVMS_getPythonCalculatorsPluginApi")),
            "libpython_calculators.so")) {
        const std::string errorDetails = safeDlerror();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so does not provide a usable OVMS_getPythonCalculatorsPluginApi: {}. "
                    "MediaPipe Python calculators will not be available.",
            errorDetails);
        dlclose(pythonCalculatorsHandle);
        pythonCalculatorsHandle = nullptr;
        return false;
    }

#elif _WIN32
    if (activateInProcessPluginApi()) {
        SPDLOG_TRACE("Python calculators implementation already present in the process, skipping plugin load");
        return true;
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

        SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", toAbsolutePath(candidate));
        pythonCalculatorsHandle = LoadLibraryA(candidate.c_str());
        if (pythonCalculatorsHandle != nullptr) {
            SPDLOG_TRACE("Python calculators plugin loaded from candidate: {}", toAbsolutePath(candidate));
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
        SPDLOG_TRACE("Python calculators plugin candidates attempted: {}", candidates.size());
        for (const auto& candidate : candidates) {
            SPDLOG_TRACE("Python calculators plugin candidate: {}", toAbsolutePath(candidate));
        }
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
