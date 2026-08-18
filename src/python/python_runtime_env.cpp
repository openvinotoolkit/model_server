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

#include "python_runtime_env.hpp"

#include <cstdlib>
#include <filesystem>

namespace ovms {

bool existsExecutableInPath(const std::string& executableName) {
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr || pathEnv[0] == '\0') {
        return false;
    }

#ifdef _WIN32
    const char separator = ';';
#else
    const char separator = ':';
#endif

    std::string pathValue(pathEnv);
    size_t start = 0;
    while (start <= pathValue.size()) {
        size_t end = pathValue.find(separator, start);
        std::string directory = (end == std::string::npos) ? pathValue.substr(start) : pathValue.substr(start, end - start);
        if (!directory.empty()) {
            auto candidate = std::filesystem::path(directory) / executableName;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec) && !ec) {
                return true;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

}  // namespace ovms
