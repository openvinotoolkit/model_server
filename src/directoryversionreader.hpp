//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <spdlog/spdlog.h>

#include "modelconfig.hpp"
#include "status.hpp"

namespace ovms {
/**
 * @brief Interface for reading versions for model manager
 */
class IVersionReader {
public:
    IVersionReader() = default;
    virtual ~IVersionReader() {}
    virtual Status readAvailableVersions(model_versions_t& versions) = 0;
};

/**
 * @brief Version reader based on directory structure
 */
class DirectoryVersionReader : public IVersionReader {
public:
    DirectoryVersionReader(const std::string& path) :
        path(path) {}
    virtual ~DirectoryVersionReader() {}
    virtual Status readAvailableVersions(model_versions_t& versions) {
        std::filesystem::directory_iterator it;
        try {
            it = std::filesystem::directory_iterator(path);
        } catch (const std::filesystem::filesystem_error&) {
            spdlog::error("Specified model directory does not exist:{}", path);
            return StatusCode::PATH_INVALID;
        }
        for (const auto& entry : it) {
            if (!entry.is_directory()) {
                spdlog::warn("Expected version directory in models path:{}. Found file:{}", path, entry.path().filename().string());
                continue;
            }
            std::string version_string = entry.path().filename().string();
            try {
                ovms::model_version_t version = std::stoll(version_string);
                versions.push_back(version);
            } catch (const std::invalid_argument& e) {
                spdlog::warn("Expected version directory to be in number format. Got:{}", version_string);
            } catch (const std::out_of_range& e) {
                spdlog::error("Directory name is out of range for supported version format. Got:{}", version_string);
            }
        }
        if (0 == versions.size()) {
            spdlog::error("No version found for model in path:{}", path);
            return StatusCode::NO_MODEL_VERSION_AVAILABLE;
        }
        return StatusCode::OK;
    }

private:
    const std::string path;
};
}  // namespace ovms
