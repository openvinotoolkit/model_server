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

#include "model_downloader.hpp"

#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"

namespace ovms {

IModelDownloader::IModelDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const bool inOverwriteModels) :
    sourceModel(inSourceModel),
    downloadPath(inDownloadPath),
    overwriteModels(inOverwriteModels) {}

Status IModelDownloader::checkIfOverwriteAndRemove() {
    auto lfstatus = StatusCode::OK;
    if (this->overwriteModels && std::filesystem::is_directory(this->downloadPath)) {
        LocalFileSystem lfs;
        lfstatus = lfs.deleteFileFolder(this->downloadPath);
        if (lfstatus != StatusCode::OK) {
            SPDLOG_ERROR("Error occurred while deleting path: {} reason: {}",
                this->downloadPath,
                lfstatus);
        } else {
            SPDLOG_DEBUG("Path deleted: {}", this->downloadPath);
        }
    }

    return lfstatus;
}

std::string IModelDownloader::getGraphDirectory(const std::string& inDownloadPath, const std::string& inSourceModel) {
    std::string fullPath = FileSystem::joinPath({inDownloadPath, inSourceModel});
    return fullPath;
}

std::string IModelDownloader::getGraphDirectory() {
    return this->downloadPath;
}
}  // namespace ovms
