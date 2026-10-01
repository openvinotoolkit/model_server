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
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include "src/logging.hpp"

namespace ovms {
namespace {

#ifdef __linux__
void* llmPluginHandle = nullptr;
#elif _WIN32
HMODULE llmPluginHandle = nullptr;
#endif

std::vector<std::string> getLlmPluginCandidates() {
    // Bazel keeps the ".so" name of the shared library target on every platform.
    std::vector<std::string> candidates{
        "libovms_llm_calculators.so",
#ifdef _WIN32
        ".\\libovms_llm_calculators.so",
        "src\\llm\\libovms_llm_calculators.so",
        ".\\src\\llm\\libovms_llm_calculators.so",
        "bazel-bin\\src\\llm\\libovms_llm_calculators.so",
        ".\\bazel-bin\\src\\llm\\libovms_llm_calculators.so",
#else
        "/ovms/lib/libovms_llm_calculators.so",
        "./libovms_llm_calculators.so",
        "src/llm/libovms_llm_calculators.so",
        "./src/llm/libovms_llm_calculators.so",
        "bazel-bin/src/llm/libovms_llm_calculators.so",
        "./bazel-bin/src/llm/libovms_llm_calculators.so",
#endif
    };

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr) {
        candidates.emplace_back(std::string(testSrcDir) + "/_main/src/llm/libovms_llm_calculators.so");
        candidates.emplace_back(std::string(testSrcDir) + "/ovms/src/llm/libovms_llm_calculators.so");
    }

    std::filesystem::path exeDir;
#ifdef _WIN32
    std::array<char, MAX_PATH> exePath{};
    DWORD exePathLength = GetModuleFileNameA(nullptr, exePath.data(), static_cast<DWORD>(exePath.size()));
    if (exePathLength > 0 && exePathLength < static_cast<DWORD>(exePath.size())) {
        exeDir = std::filesystem::path(exePath.data()).parent_path();
    }
#else
    std::array<char, PATH_MAX> exePath{};
    ssize_t exePathLength = readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
    if (exePathLength > 0) {
        exePath[exePathLength] = '\0';
        exeDir = std::filesystem::path(exePath.data()).parent_path();
    }
#endif
    if (!exeDir.empty()) {
        candidates.emplace_back((exeDir / "libovms_llm_calculators.so").string());
        candidates.emplace_back((exeDir / "src/llm/libovms_llm_calculators.so").string());
        candidates.emplace_back((exeDir / "llm/libovms_llm_calculators.so").string());
    }
    return candidates;
}

}  // namespace

bool loadLlmCalculatorsPlugin() {
    if (llmPluginHandle != nullptr) {
        return true;
    }

    for (const auto& candidate : getLlmPluginCandidates()) {
#ifdef _WIN32
        llmPluginHandle = LoadLibraryA(candidate.c_str());
#else
        llmPluginHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
        if (llmPluginHandle != nullptr) {
            SPDLOG_TRACE("LLM calculators plugin loaded from: {}", candidate);
            return true;
        }
    }

    SPDLOG_DEBUG("LLM calculators plugin is unavailable");
    return false;
}

void* getLlmCalculatorsPluginSymbol(const char* symbolName) {
    if (llmPluginHandle == nullptr && !loadLlmCalculatorsPlugin()) {
        return nullptr;
    }
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(llmPluginHandle, symbolName));
#else
    return dlsym(llmPluginHandle, symbolName);
#endif
}

}  // namespace ovms
