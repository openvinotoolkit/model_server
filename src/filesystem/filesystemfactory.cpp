//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "filesystemfactory.hpp"

#include <memory>
#include <utility>

#include "filesystem.hpp"
#include "localfilesystem.hpp"
#if (CLOUD_DISABLE == 0)
#include "azurefilesystem.hpp"
#include "gcsfilesystem.hpp"
#include "s3filesystem.hpp"
#endif

namespace ovms {
std::shared_ptr<FileSystem> getFilesystem(const std::string& basePath) {
#if (CLOUD_DISABLE == 0)
    if (basePath.rfind(FileSystem::S3_URL_PREFIX, 0) == 0) {
        Aws::SDKOptions options;
        Aws::InitAPI(options);
        return std::make_shared<S3FileSystem>(options, basePath);
    }
    if (basePath.rfind(FileSystem::GCS_URL_PREFIX, 0) == 0) {
        return std::make_shared<ovms::GCSFileSystem>();
    }
    if (basePath.rfind(FileSystem::AZURE_URL_FILE_PREFIX, 0) == 0) {
        return std::make_shared<ovms::AzureFileSystem>();
    }
    if (basePath.rfind(FileSystem::AZURE_URL_BLOB_PREFIX, 0) == 0) {
        return std::make_shared<ovms::AzureFileSystem>();
    }
#endif
    return std::make_shared<LocalFileSystem>();
}
}  // namespace ovms
