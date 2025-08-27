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
    OptimumDownloader(const HFSettingsImpl& hfSettings, const std::string& cliExportCmd = "optimum-cli export openvino ", const std::string& cliCheckCmd = "optimum-cli -h");
    Status cloneRepository();
    std::string getGraphDirectory();

protected:
    std::string sourceModel;
    std::string downloadPath;
    HFSettingsImpl hfSettings;
    bool overwriteModels;
    std::string OPTIMUM_CLI_EXPORT_COMMAND;
    std::string OPTIMUM_CLI_CHECK_COMMAND;

    OptimumDownloader();
    Status checkRequiredToolsArePresent();
    std::string getExportCmd();
    std::string getExportCmdText();
    std::string getExportCmdEmbeddings();
    std::string getExportCmdRerank();
    std::string getExportCmdImageGeneration();
};
}  // namespace ovms
