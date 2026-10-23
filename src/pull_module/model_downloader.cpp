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
#include <iostream>
#include <string>

#include "model_downloader.hpp"

#include "src/filesystem/filesystem.hpp"
#include "src/filesystem/localfilesystem.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../logging.hpp"

namespace ovms {

IModelDownloader::IModelDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const bool inOverwriteModels) :
    sourceModel(inSourceModel),
    downloadPath(inDownloadPath),
    overwriteModels(inOverwriteModels) {}

Status IModelDownloader::validateGraphDirectory() const {
    return StatusCode::OK;
}

void IModelDownloader::onDownloadComplete(HFSettingsImpl&) const {
    std::cout << "Model: " << this->sourceModel << " downloaded to: " << this->getGraphDirectory() << std::endl;
}

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
    if (inSourceModel.empty()) {
        return inDownloadPath;
    }

    // localModelDirectoryName() is the identity for HuggingFace repo ids; it
    // only rewrites OCI references, whose "oci://" scheme and ':' tag
    // separator are not usable as a directory name.
    const std::string normalizedSourceModel = localModelDirectoryName(inSourceModel);
    std::filesystem::path downloadPath(inDownloadPath);
    std::filesystem::path sourceModelPath(normalizedSourceModel);
    const std::string downloadNorm = downloadPath.lexically_normal().generic_string();
    const std::string sourceNorm = sourceModelPath.lexically_normal().generic_string();

    if (downloadNorm == sourceNorm) {
        return downloadNorm;
    }

    const auto containsSourcePathSegment = [&sourceNorm](const std::string& path) {
        if (sourceNorm.empty() || path.empty()) {
            return false;
        }

        std::string normalized = path;
        std::replace(normalized.begin(), normalized.end(), '\\', '/');

        if (normalized == sourceNorm) {
            return true;
        }
        if (normalized.rfind(sourceNorm + "/", 0) == 0) {
            return true;
        }

        const std::string segment = "/" + sourceNorm + "/";
        const std::string segmentAtEnd = "/" + sourceNorm;
        if (normalized.size() >= segment.size() && normalized.find(segment) != std::string::npos) {
            return true;
        }
        if (normalized.size() >= segmentAtEnd.size() &&
            normalized.compare(normalized.size() - segmentAtEnd.size(), segmentAtEnd.size(), segmentAtEnd) == 0) {
            return true;
        }
        return false;
    };

    if (containsSourcePathSegment(downloadNorm)) {
        return downloadNorm;
    }

    std::string fullPath = FileSystem::joinPath({inDownloadPath, normalizedSourceModel});
    return std::filesystem::path(fullPath).lexically_normal().generic_string();
}

std::string IModelDownloader::getGraphDirectory() const {
    return this->downloadPath;
}
}  // namespace ovms
