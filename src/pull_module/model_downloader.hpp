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

namespace ovms {
class Status;
class IModelDownloader {
public:
    IModelDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const bool inOverwriteModels);
    virtual ~IModelDownloader() = default;
    virtual Status downloadModel() = 0;
    static std::string getGraphDirectory(const std::string& inDownloadPath, const std::string& inSourceModel);
    std::string getGraphDirectory();

protected:
    Status checkIfOverwriteAndRemove();
    const std::string sourceModel;
    const std::string downloadPath;
    const bool overwriteModels;
};

}  // namespace ovms
