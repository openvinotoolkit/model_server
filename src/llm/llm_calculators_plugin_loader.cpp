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

#ifdef __linux__

#include <array>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "src/logging.hpp"

namespace ovms {
namespace {

void* llmPluginHandle = nullptr;

std::vector<std::string> getLlmPluginCandidates() {
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
}

}  // namespace

bool loadLlmCalculatorsPlugin() {
    if (llmPluginHandle != nullptr) {
        return true;
    }

    for (const auto& candidate : getLlmPluginCandidates()) {
        llmPluginHandle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (llmPluginHandle != nullptr) {
            SPDLOG_TRACE("LLM calculators plugin loaded from: {}", candidate);
            return true;
        }
    }

    SPDLOG_DEBUG("LLM calculators plugin is unavailable");
    return false;
}

void* getLlmCalculatorsPluginSymbol(const char* symbolName) {
    if (!loadLlmCalculatorsPlugin()) {
        return nullptr;
    }
    void* handle = llmPluginHandle;
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
bool loadLlmCalculatorsPlugin() {
    return false;
}

void* getLlmCalculatorsPluginSymbol(const char*) {
    return nullptr;
}

}  // namespace ovms

#endif
