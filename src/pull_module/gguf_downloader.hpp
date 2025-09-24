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
#include <variant>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "model_downloader.hpp"

namespace ovms {
class Status;
enum GraphExportType : unsigned int;

class GGUFDownloader : public IModelDownloader {
public:
    GGUFDownloader(const std::string& sourceModel, const std::string& downloadPath, bool inOverwrite, const std::optional<std::string> ggufFilename, const std::string& hfEndpoint);
    Status downloadModel() override;
    Status checkIfOverwriteAndRemove();
    static Status downloadWithCurl(const std::string& hfEndpoint, const std::string& modelName, const std::string& filenamePrefix, const std::string& ggufFilename, const std::string& downloadPath);
    static std::variant<Status, std::string> preparePartFilename(const std::string& ggufFilename, int part, int totalParts);
    static std::variant<Status, std::vector<std::string>> createGGUFFilenamesToDownload(const std::string& ggufFilename);
    static std::variant<Status, bool> checkIfAlreadyExists(const std::optional<std::string> ggufFilename, const std::string& path);

protected:
    const std::optional<std::string> ggufFilename;
    const std::string hfEndpoint;
};
}  // namespace ovms
