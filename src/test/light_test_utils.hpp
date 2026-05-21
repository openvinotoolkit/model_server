#pragma once
//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <string>
std::string GetFileContents(const std::string& filePath);
bool createConfigFileWithContent(const std::string& content, std::string filename = "/tmp/ovms_config_file.json");

// Removes generated graph header lines (version and optional queue size directive)
// which differ across build/runtime setup.
inline std::string removeGeneratedGraphHeaders(std::string input) {
    auto firstLineEnd = input.find("\n");
    if (firstLineEnd == std::string::npos) {
        return "";
    }
    input.erase(0, firstLineEnd + 1);

    const std::string queueLinePrefix = "# OVMS_GRAPH_QUEUE_MAX_SIZE:";
    if (input.rfind(queueLinePrefix, 0) == 0) {
        auto secondLineEnd = input.find("\n");
        if (secondLineEnd == std::string::npos) {
            return "";
        }
        input.erase(0, secondLineEnd + 1);
    }
    return input;
}
