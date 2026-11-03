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

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_base.h"
#include "src/logging.hpp"

#ifdef __linux__
#include <array>
#include <climits>

#include <dlfcn.h>
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#include "mediapipe/calculators/image/image_cropping_calculator.pb.h"
#include "mediapipe/calculators/image/image_transformation_calculator.pb.h"
#include "mediapipe/calculators/image/opencv_encoded_image_to_image_frame_calculator.pb.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/calculators/image/segmentation_smoothing_calculator.pb.h"
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/util/annotation_overlay_calculator.pb.h"
#endif

#include "mediapipe/framework/subgraph.h"

namespace ovms {
namespace {

struct PluginState {
    std::mutex mutex;
    void* handle = nullptr;
    bool loaded = false;
    std::string testCandidatePath;
    std::string error = "OpenCV calculator plugin is not loaded";
};

PluginState& pluginState() {
    static PluginState state;
    return state;
}

#ifdef _WIN32
void retainOpenCvCalculatorOptionDescriptors() {
    (void)mediapipe::ImageCroppingCalculatorOptions::ext.number();
    (void)mediapipe::ImageTransformationCalculatorOptions::ext.number();
    (void)mediapipe::OpenCvEncodedImageToImageFrameCalculatorOptions::ext.number();
    (void)mediapipe::OpenCvImageEncoderCalculatorOptions::ext.number();
    (void)mediapipe::SegmentationSmoothingCalculatorOptions::ext.number();
    (void)mediapipe::WarpAffineCalculatorOptions::ext.number();
    (void)mediapipe::ImageToTensorCalculatorOptions::ext.number();
    (void)mediapipe::AnnotationOverlayCalculatorOptions::ext.number();
}
#endif

#ifdef __linux__
std::vector<std::string> pluginCandidates() {
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
#elif _WIN32
std::vector<std::string> pluginCandidates() {
    std::vector<std::string> candidates;
    std::vector<HMODULE> modules;
    HMODULE loaderModule = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&loadOpenCvCalculatorsPlugin), &loaderModule)) {
        modules.emplace_back(loaderModule);
    }
    if (HMODULE executableModule = GetModuleHandleA(nullptr); executableModule != nullptr) {
        modules.emplace_back(executableModule);
    }

    std::vector<char> modulePath(32768);
    for (const auto module : modules) {
        const DWORD pathLength = GetModuleFileNameA(module, modulePath.data(), static_cast<DWORD>(modulePath.size()));
        if (pathLength == 0 || static_cast<size_t>(pathLength) >= modulePath.size()) {
            continue;
        }
        const auto directory = std::filesystem::path(std::string(modulePath.data(), pathLength)).parent_path();
        candidates.emplace_back((directory / "ovms_opencv_calculators.dll").string());
        candidates.emplace_back((directory / "lib" / "ovms_opencv_calculators.dll").string());

        const auto runfilesDirectory = std::filesystem::path(std::string(modulePath.data(), pathLength) + ".runfiles");
        for (const char* workspace : {"ovms", "_main", "model_server"}) {
            candidates.emplace_back((runfilesDirectory / workspace / "src/lib/ovms_opencv_calculators.dll").string());
        }
    }

    return candidates;
}
#endif

bool loadOpenCvCalculatorsPluginFromCandidates(const std::vector<std::string>& candidates) {
    auto& state = pluginState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.loaded) {
        return true;
    }
#ifdef __linux__
    state.error.clear();
    if (!exposeHostSymbols(state.error)) {
        SPDLOG_ERROR("OpenCV calculators plugin host promotion failed: {}", state.error);
        return false;
    }
    for (const auto& candidate : candidates) {
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
            state.loaded = true;
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
#elif _WIN32
    state.error.clear();
    retainOpenCvCalculatorOptionDescriptors();
    if (candidates.empty()) {
        state.error = "cannot locate the OVMS module directory or OpenCV calculator plugin";
    }
    for (const auto& candidate : candidates) {
        HMODULE handle = LoadLibraryExA(candidate.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (handle == nullptr) {
            if (!state.error.empty()) {
                state.error += "; ";
            }
            state.error += candidate + ": Windows loader error " + std::to_string(GetLastError());
            continue;
        }

        using RegisterFactories = bool (*)(void*, void*);
        auto registerFactories = reinterpret_cast<RegisterFactories>(GetProcAddress(handle, "OVMS_RegisterOpenCvCalculators"));
        if (registerFactories == nullptr ||
            !registerFactories(mediapipe::CalculatorBaseRegistry::functions(), mediapipe::SubgraphRegistry::functions())) {
            if (!state.error.empty()) {
                state.error += "; ";
            }
            state.error += candidate + ": missing or incompatible OVMS_RegisterOpenCvCalculators entry point";
            continue;
        }

        state.handle = handle;
        state.loaded = true;
        state.error.clear();
        SPDLOG_DEBUG("OpenCV calculators plugin loaded from: {}", candidate);
        return true;
    }
#endif
    SPDLOG_DEBUG("OpenCV calculators plugin unavailable: {}", state.error);
    return false;
}

}  // namespace

bool loadOpenCvCalculatorsPlugin() {
    auto& state = pluginState();
    std::vector<std::string> candidates;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.loaded) {
            return true;
        }
        if (!state.testCandidatePath.empty()) {
            candidates.emplace_back(state.testCandidatePath);
        }
    }
    if (candidates.empty()) {
        candidates = pluginCandidates();
    }
    return loadOpenCvCalculatorsPluginFromCandidates(candidates);
}

bool loadOpenCvCalculatorsPluginForTesting(const std::string& candidatePath) {
    auto& state = pluginState();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.loaded) {
            state.error = "cannot override the OpenCV calculator plugin after it has loaded";
            return false;
        }
        state.testCandidatePath = candidatePath;
    }
    return loadOpenCvCalculatorsPlugin();
}

void clearOpenCvCalculatorsPluginTestCandidate() {
    auto& state = pluginState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.testCandidatePath.clear();
}

std::string getOpenCvCalculatorsPluginDiagnostic() {
    auto& state = pluginState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.loaded) {
        return {};
    }
#ifdef _WIN32
    return " OpenCV-dependent calculator/operation or options require the optional "
           "ovms_opencv_calculators.dll plugin and OpenCV libraries. Missing OpenCV/plugin: " + state.error;
#else
    return " OpenCV-dependent calculator/operation or options require the optional "
           "libovms_opencv_calculators.so plugin and OpenCV libraries. Missing OpenCV/plugin: " +
           state.error;
#endif
}

}  // namespace ovms
