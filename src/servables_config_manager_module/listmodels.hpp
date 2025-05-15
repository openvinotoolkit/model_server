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
#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ovms {
enum ServableType_t {
    SERVABLE_TYPE_MODEL,
    SERVABLE_TYPE_MEDIAPIPEGRAPH
};

bool isVersionDir(const std::string& path);
bool isMediapipeGraphDir(const std::string& path);
std::string getPartialPath(const std::filesystem::path& path, int depth);

template <std::size_t N>
static bool hasRequiredExtensions(const std::string& directoryPath, const std::array<const char*, N>& extensions) {
    std::unordered_set<std::string> foundExtensions;
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                foundExtensions.insert(ext);
            }
        }
        if (foundExtensions.size() == extensions.size()) {
            return true;
        }
    }
    return false;
}
std::unordered_map<std::string, ServableType_t> listServables(const std::string directoryPath);
void addEntryAndReturnIfContainsModel(const std::filesystem::path& directoryPath, std::unordered_map<std::string, ovms::ServableType_t>& servablesList, std::string& dirName, bool& retFlag);
void NewFunction(const std::filesystem::directory_entry& entry, std::unordered_map<std::string, ovms::ServableType_t>& servablesList, std::string& dirName, bool& retFlag);
}  // namespace ovms
