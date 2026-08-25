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
#include "python_runtime_version.hpp"

#include <cctype>
#include <cstdlib>
#include <filesystem>

namespace ovms {

namespace {

// Collects a run of digits from the start of `text`, allowing '.' separators
// between digit groups (e.g. "3.13" -> "313", "313\\Lib" -> "313").
std::string extractVersionDigits(const std::string& text) {
    std::string digits;
    bool sawDigit = false;
    for (char c : text) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digits += c;
            sawDigit = true;
        } else if (c == '.' && sawDigit) {
            continue;
        } else {
            break;
        }
    }
    return digits;
}

}  // namespace

std::string detectPythonAbiTag() {
    if (const char* overrideTag = std::getenv("OVMS_PYTHON_ABI"); overrideTag != nullptr && overrideTag[0] != '\0') {
        return std::string(overrideTag);
    }

    if (const char* pythonHome = std::getenv("PYTHONHOME"); pythonHome != nullptr && pythonHome[0] != '\0') {
        const std::string home(pythonHome);
        std::string lowerHome(home);
        for (char& c : lowerHome) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const size_t markerPos = lowerHome.rfind("python");
        if (markerPos != std::string::npos) {
            const std::string tail = home.substr(markerPos + 6);  // skip past "python"
            const std::string digits = extractVersionDigits(tail);
            if (digits.size() >= 2) {
                return digits;
            }
        }
    }

    return std::string();
}

std::vector<std::string> withAbiVersionedCandidates(const std::vector<std::string>& baseCandidates) {
    const std::string abiTag = detectPythonAbiTag();
    if (abiTag.empty()) {
        return baseCandidates;
    }

    std::vector<std::string> versioned;
    versioned.reserve(baseCandidates.size());
    for (const auto& candidatePath : baseCandidates) {
        const std::filesystem::path original(candidatePath);
        const std::filesystem::path parent = original.parent_path();
        const std::string versionedName = original.stem().string() + "-cp" + abiTag + original.extension().string();
        const std::filesystem::path versionedPath = parent.empty() ? std::filesystem::path(versionedName) : parent / versionedName;
        versioned.push_back(versionedPath.string());
    }

    // Versioned candidates take priority; fall back to the unsuffixed names for
    // backward compatibility with single-ABI builds/packages.
    std::vector<std::string> result;
    result.reserve(versioned.size() + baseCandidates.size());
    result.insert(result.end(), versioned.begin(), versioned.end());
    result.insert(result.end(), baseCandidates.begin(), baseCandidates.end());
    return result;
}

}  // namespace ovms
