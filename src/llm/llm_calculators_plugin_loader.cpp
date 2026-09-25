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
#ifdef _WIN32
    return {
        "ovms_llm_calculators.dll",
        ".\\ovms_llm_calculators.dll",
        "src\\llm\\ovms_llm_calculators.dll",
        ".\\src\\llm\\ovms_llm_calculators.dll",
        "bazel-bin\\src\\llm\\ovms_llm_calculators.dll",
        ".\\bazel-bin\\src\\llm\\ovms_llm_calculators.dll"};
#else
    std::vector<std::string> candidates{
        "libovms_llm_calculators.so",
        "/ovms/lib/libovms_llm_calculators.so",
        "./libovms_llm_calculators.so",
        "src/llm/libovms_llm_calculators.so",
        "./src/llm/libovms_llm_calculators.so",
        "bazel-bin/src/llm/libovms_llm_calculators.so",
        "./bazel-bin/src/llm/libovms_llm_calculators.so"};

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr) {
        candidates.emplace_back(std::string(testSrcDir) + "/_main/src/llm/libovms_llm_calculators.so");
        candidates.emplace_back(std::string(testSrcDir) + "/ovms/src/llm/libovms_llm_calculators.so");
    }

    std::array<char, PATH_MAX> exePath{};
    ssize_t exePathLength = readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
    if (exePathLength > 0) {
        exePath[exePathLength] = '\0';
        std::filesystem::path exeDir = std::filesystem::path(exePath.data()).parent_path();
        candidates.emplace_back((exeDir / "libovms_llm_calculators.so").string());
        candidates.emplace_back((exeDir / "src/llm/libovms_llm_calculators.so").string());
        candidates.emplace_back((exeDir / "llm/libovms_llm_calculators.so").string());
    }
    return candidates;
#endif
}

}  // namespace

bool loadLlmCalculatorsPlugin() {
    const char* enabled = std::getenv("OVMS_LLM_CALCULATORS_PLUGIN_DLOPEN");
    if (enabled == nullptr || std::string(enabled) != "1") {
        SPDLOG_DEBUG("LLM calculators plugin loading is disabled; set OVMS_LLM_CALCULATORS_PLUGIN_DLOPEN=1 to enable the experimental plugin");
        return false;
    }

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

}  // namespace ovms