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

#include <cstdlib>
#include <cstdio>
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
    const bool forceInProcessForTests = []() {
        const char* value = std::getenv("OVMS_TEST_PYTHON_CALCULATORS_INPROCESS");
        return value != nullptr && std::string(value) == "1";
    }();
    bool hasInProcessRegisterFn = false;

    if (registerPythonCalculators != nullptr) {
        registerPythonCalculatorsFn = registerPythonCalculators;
        hasInProcessRegisterFn = true;
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

    if (forceInProcessForTests) {
        auto* inProcessRegisterFn = reinterpret_cast<RegisterPythonCalculatorsFn>(dlsym(RTLD_DEFAULT, "registerPythonCalculators"));
        if (inProcessRegisterFn == nullptr) {
            SPDLOG_ERROR("OVMS_TEST_PYTHON_CALCULATORS_INPROCESS=1 but registerPythonCalculators is not available in-process.");
            SPDLOG_ERROR("Refusing to fall back to libpython_calculators.so dlopen in strict test in-process mode.");
            return false;
        } else {
            registerPythonCalculatorsFn = inProcessRegisterFn;
            hasInProcessRegisterFn = true;
            auto* getKfsBridgeFn = reinterpret_cast<GetKfsPyTensorBridgeVTableFn>(dlsym(RTLD_DEFAULT, "OVMS_getKfsPyTensorBridgeVTable"));
            if (getKfsBridgeFn != nullptr) {
                if (auto* vtable = getKfsBridgeFn(); vtable != nullptr) {
                    activateKfsBridge(vtable, "in-process symbol (test mode)");
                }
            }

            if (getKfsPyTensorBridgeVTable() != nullptr) {
                SPDLOG_TRACE("OVMS_TEST_PYTHON_CALCULATORS_INPROCESS=1 set; using in-process Python calculators symbols and skipping plugin dlopen");
                return true;
            }
            SPDLOG_ERROR("OVMS_TEST_PYTHON_CALCULATORS_INPROCESS=1 set and in-process register function was found, but KFS bridge is missing.");
            SPDLOG_ERROR("Refusing to fall back to libpython_calculators.so dlopen in strict test in-process mode.");
            return false;
        }
    }

    auto* alreadyLoadedRegisterFn = reinterpret_cast<RegisterPythonCalculatorsFn>(dlsym(RTLD_DEFAULT, "registerPythonCalculators"));
    if (alreadyLoadedRegisterFn != nullptr) {
        registerPythonCalculatorsFn = alreadyLoadedRegisterFn;
        hasInProcessRegisterFn = true;

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

    const bool forcePluginDlopen = []() {
        const char* value = std::getenv("OVMS_PYTHON_CALCULATORS_PLUGIN_DLOPEN");
        return value != nullptr && std::string(value) == "1";
    }();

    // In runtime-separation mode, calculator registrations are expected to be
    // owned by libovms_mediapipe_runtime_shared.so. Eagerly dlopen-ing
    // libpython_calculators.so here can register MediaPipe framework handlers
    // twice (plugin first, runtime-shared second).
    if (!forceInProcessForTests && !forcePluginDlopen) {
        SPDLOG_INFO("Skipping explicit libpython_calculators.so dlopen. "
                    "Python calculators are expected from runtime-shared ownership; "
                    "set OVMS_PYTHON_CALCULATORS_PLUGIN_DLOPEN=1 to force plugin loading.");
        return true;
    }

    // Avoid eager preloading of runtime-shared calculators here.
    // In split-runtime deployments, preloading can trigger duplicate MediaPipe
    // registrations (for example OpenVINOInferenceCalculator) before plugin
    // fallback logic has a chance to run.

    constexpr const char* pluginName = "libpython_calculators.so";

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

    // Try loading the plugin in a child process first. If the probe crashes or
    // fails, avoid bringing down OVMS during optional plugin initialization.
    if (probePluginLoadInChildProcess(pluginName)) {
        // Use RTLD_GLOBAL to share plugin symbols with the main binary.
        // This allows the main binary to call plugin functions like registerPythonCalculators.
        SPDLOG_DEBUG("Attempting to load Python calculators plugin: {}", pluginName);

        pythonCalculatorsHandle = dlopen(pluginName, pythonPluginDlopenFlags());

        if (pythonCalculatorsHandle != nullptr) {
            SPDLOG_TRACE("Successfully loaded Python calculators plugin: {}", pluginName);
        } else {
            const std::string errorDetails = safeDlerror();
            SPDLOG_DEBUG("Failed to load Python calculators plugin {}: {}", pluginName, errorDetails);
        }
    }

    if (pythonCalculatorsHandle == nullptr) {
        const std::string errorDetails = safeDlerror();
        SPDLOG_WARN("Python calculators plugin libpython_calculators.so failed to load: {}. "
                    "MediaPipe Python calculators will not be available.",
            errorDetails);
        if (hasInProcessRegisterFn) {
            SPDLOG_TRACE("Proceeding with in-process python calculators registration without KFS Python tensor bridge.");
            return true;
        }
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
