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
#include "opencv_calculators_plugin_loader.hpp"

#include <mutex>
#include <string>

#include "src/logging.hpp"
#include "src/utils/env_guard.hpp"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/subgraph.h"

#ifdef __linux__
#include <array>
#include <climits>
#include <filesystem>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>
#endif

namespace ovms {
namespace {

struct PluginState {
    std::mutex mutex;
    void* handle = nullptr;
    std::string error = "OpenCV calculator plugin is not loaded";
};

PluginState& pluginState() {
    static PluginState state;
    return state;
}

#ifdef __linux__
std::vector<std::string> pluginCandidates() {
    const std::string overridePath = GetEnvVar("OVMS_OPENCV_CALCULATORS_PLUGIN_PATH");
    if (!overridePath.empty()) {
        return {overridePath};
    }
    std::vector<std::string> candidates;
    // Also support a C API host whose executable is outside the OVMS package.
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&loadOpenCvCalculatorsPlugin), &info) && info.dli_fname) {
        const auto directory = std::filesystem::absolute(info.dli_fname).parent_path();
        candidates.emplace_back((directory / "libovms_opencv_calculators.so").string());
        candidates.emplace_back((directory / "lib" / "libovms_opencv_calculators.so").string());
    }
    std::array<char, PATH_MAX> executable{};
    const ssize_t length = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length > 0) {
        const auto directory = std::filesystem::path(std::string(executable.data(), length)).parent_path();
        candidates.emplace_back((directory / "lib" / "libovms_opencv_calculators.so").string());
        candidates.emplace_back((directory.parent_path() / "lib" / "libovms_opencv_calculators.so").string());
    }
    return candidates;
}

bool exposeHostSymbols(std::string& error) {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(&loadOpenCvCalculatorsPlugin), &info) || !info.dli_fname) {
        error = "cannot locate the OpenCV calculator plugin host with dladdr";
        return false;
    }
    std::array<char, PATH_MAX> executable{};
    const ssize_t length = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    std::error_code filesystemError;
    if (length > 0 &&
        std::filesystem::equivalent(info.dli_fname, std::string(executable.data(), length), filesystemError)) {
        return true;
    }
    // ctypes and embedded C API clients normally load OVMS with RTLD_LOCAL.
    // Promote only the already-loaded module containing this loader, never an
    // environment-supplied path. Keep the reference for the plugin's lifetime.
    if (dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL) == nullptr) {
        const char* loaderError = dlerror();
        error = "cannot expose OpenCV calculator host symbols from " + std::string(info.dli_fname) +
                ": " + (loaderError ? loaderError : "dynamic loader failed");
        return false;
    }
    return true;
}
#endif

}  // namespace

bool loadOpenCvCalculatorsPlugin() {
    auto& state = pluginState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.handle != nullptr) {
        return true;
    }
#ifdef __linux__
    state.error.clear();
    if (!exposeHostSymbols(state.error)) {
        SPDLOG_ERROR("OpenCV calculators plugin host promotion failed: {}", state.error);
        return false;
    }
    for (const auto& candidate : pluginCandidates()) {
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            using RegisterFactories = bool (*)(void*, void*);
            auto registerFactories = reinterpret_cast<RegisterFactories>(
                dlsym(handle, "OVMS_RegisterOpenCvCalculators"));
            if (registerFactories == nullptr ||
                !registerFactories(mediapipe::CalculatorBaseRegistry::functions(),
                    mediapipe::SubgraphRegistry::functions())) {
                state.error = candidate + ": missing or incompatible OpenCV calculator registration entry point";
                // Static protobuf registration has process lifetime, so never
                // unload even a plugin with an incompatible entry point.
                continue;
            }
            state.handle = handle;
            state.error.clear();
            SPDLOG_DEBUG("OpenCV calculators plugin loaded from: {}", candidate);
            return true;
        }
        const char* error = dlerror();
        if (!state.error.empty()) {
            state.error += "; ";
        }
        state.error += candidate + ": " + (error ? error : "dynamic loader failed");
    }
#else
    state.error = "runtime OpenCV calculator plugins are not supported on this platform";
#endif
    SPDLOG_DEBUG("OpenCV calculators plugin unavailable: {}", state.error);
    return false;
}

std::string getOpenCvCalculatorsPluginDiagnostic() {
    auto& state = pluginState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.handle != nullptr) {
        return {};
    }
    return " OpenCV-dependent calculator/operation or options require the optional "
           "libovms_opencv_calculators.so plugin and OpenCV libraries. Missing OpenCV/plugin: " +
           state.error;
}

}  // namespace ovms
