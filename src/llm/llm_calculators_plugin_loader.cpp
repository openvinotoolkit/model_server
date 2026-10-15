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

#include "llm_calculators_plugin_loader.hpp"

#include <array>
#include <climits>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#ifdef __linux__

#include <dlfcn.h>
#include <unistd.h>

#include "src/logging.hpp"

namespace ovms {
namespace {

std::vector<std::string> getLlmPluginCandidates() {
    std::vector<std::string> candidates;

    std::array<char, PATH_MAX> exePath{};
    ssize_t exePathLength = readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
    if (exePathLength > 0) {
        exePath[exePathLength] = '\0';
        std::filesystem::path exeDir = std::filesystem::path(exePath.data()).parent_path();
        candidates.emplace_back((exeDir / "lib" / "libovms_llm_calculators.so").string());
        candidates.emplace_back((exeDir.parent_path() / "lib" / "libovms_llm_calculators.so").string());
    }
    return candidates;
}

}  // namespace

void* loadLlmCalculatorsPlugin() {
    static std::mutex loadMutex;
    static void* llmPluginHandle = nullptr;
    std::lock_guard<std::mutex> lock(loadMutex);
    if (llmPluginHandle != nullptr) {
        return llmPluginHandle;
    }

    for (const auto& candidate : getLlmPluginCandidates()) {
        llmPluginHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (llmPluginHandle != nullptr) {
            SPDLOG_TRACE("LLM calculators plugin loaded from: {}", candidate);
            return llmPluginHandle;
        }
    }

    SPDLOG_DEBUG("LLM calculators plugin is unavailable");
    return nullptr;
}

void* getLlmCalculatorsPluginSymbol(const char* symbolName) {
    void* handle = loadLlmCalculatorsPlugin();
    if (handle == nullptr) {
        return nullptr;
    }
    return dlsym(handle, symbolName);
}

}  // namespace ovms

#else

namespace ovms {

// The plugin relies on the dynamic linker binding its unresolved references back to the host
// executable. Windows has no equivalent, so there the runtime calls the servable directly.
void* loadLlmCalculatorsPlugin() {
    return nullptr;
}

void* getLlmCalculatorsPluginSymbol(const char*) {
    return nullptr;
}

}  // namespace ovms

#endif
