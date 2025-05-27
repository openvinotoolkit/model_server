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
#include "optimum_export.hpp"

#include <string>
#include <memory>

#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../stringutils.hpp"
#include "../status.hpp"

namespace ovms {

Op

OptimumDownloader::OptimumDownloader() {
    this->sourceModel = "";
    this->downloadPath = "";
    this->hfEndpoint = "";
    this->hfToken = "";
    this->httpProxy = "";
    this->overwriteModels = false;
}

std::string OptimumDownloader::getGraphDirectory() {
    return this->downloadPath;
}

OptimumDownloader::OptimumDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const std::string& inHfEndpoint, const std::string& inHfToken, const std::string& inHttpProxy, bool inOverwrite) {
    this->sourceModel = inSourceModel;
    this->downloadPath = HfDownloader::getGraphDirectory(inDownloadPath, inSourceModel);
    this->hfEndpoint = inHfEndpoint;
    this->hfToken = inHfToken;
    this->httpProxy = inHttpProxy;
    this->overwriteModels = inOverwrite;
}

Status OptimumDownloader::cloneRepository() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        SPDLOG_DEBUG("Path already exists on local filesystem. Not downloading to path: {}", this->downloadPath);
        return StatusCode::OK;
    }

    auto status = checkIfOverwriteAndRemove(this->downloadPath);
    if (!status.ok()) {
        return status;
    }


    return StatusCode::OK;
}

}  // namespace ovms
