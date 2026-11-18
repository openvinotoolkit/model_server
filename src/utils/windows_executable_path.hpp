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

#ifdef _WIN32
#include <filesystem>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

namespace ovms {
inline std::filesystem::path getWindowsExecutablePath() {
    std::wstring executablePath(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, executablePath.data(), static_cast<DWORD>(executablePath.size()));
        if (length == 0) {
            return {};
        }
        if (length < executablePath.size()) {
            executablePath.resize(length);
            return std::filesystem::path(executablePath);
        }
        if (executablePath.size() >= 32768) {
            return {};
        }
        executablePath.resize(executablePath.size() * 2);
    }
}
}  // namespace ovms
#endif