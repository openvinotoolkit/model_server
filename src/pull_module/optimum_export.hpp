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

#include "libgit2.hpp"
#include "../capi_frontend/server_settings.hpp"

namespace ovms {
class Status;
enum GraphExportType : unsigned int;

class OptimumDownloader : public HfDownloader {
public:
    OptimumDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const HFSettingsImpl& hfSettings, bool inOverwrite);
    Status cloneRepository();
    std::string getGraphDirectory();

protected:
    std::string sourceModel;
    std::string downloadPath;
    HFSettingsImpl hfSettings;
    bool overwriteModels;

    OptimumDownloader();
    Status checkRequiredToolsArePresent();
};
}  // namespace ovms
