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

#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

namespace ovms {

// Requires an initialized interpreter. Existing search paths retain precedence.
inline void addPackagedPythonPath(const std::filesystem::path& executablePath) {
#ifdef _WIN32
    const auto pythonDirectory = executablePath.parent_path() / "python";
#else
    const auto pythonDirectory = (executablePath.parent_path() / "../lib/python").lexically_normal();
#endif
    if (!std::filesystem::is_directory(pythonDirectory)) {
        return;
    }

    pybind11::gil_scoped_acquire acquire;
    pybind11::list paths = pybind11::module_::import("sys").attr("path");
    const pybind11::str pythonPath(pythonDirectory.string());
    if (!paths.contains(pythonPath)) {
        paths.append(pythonPath);
    }
}

// Add Python modules from the package-relative directory for the current platform.
inline void addPackagedPythonPath() {
#ifdef __linux__
    addPackagedPythonPath(std::filesystem::canonical("/proc/self/exe"));
#elif defined(_WIN32)
    std::wstring executablePath(MAX_PATH, L'\0');
    for (;;) {
        const DWORD pathLength = GetModuleFileNameW(nullptr, executablePath.data(), static_cast<DWORD>(executablePath.size()));
        if (pathLength == 0) {
            return;
        }
        if (pathLength < executablePath.size()) {
            executablePath.resize(pathLength);
            addPackagedPythonPath(std::filesystem::path(executablePath));
            return;
        }
        if (executablePath.size() >= 32768) {
            return;
        }
        executablePath.resize(executablePath.size() * 2);
    }
#endif
}

}  // namespace ovms
