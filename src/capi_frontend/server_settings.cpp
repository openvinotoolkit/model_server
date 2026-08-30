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
#include <map>
#include <string>

#include "server_settings.hpp"
#include "../stringutils.hpp"

namespace ovms {
std::string enumToString(ConfigExportType type) {
    auto it = configExportTypeToString.find(type);
    return (it != configExportTypeToString.end()) ? it->second : "UNKNOWN_MODEL";
}

ConfigExportType stringToConfigExportEnum(const std::string& inString) {
    auto it = stringToConfigExportType.find(inString);
    return (it != stringToConfigExportType.end()) ? it->second : UNKNOWN_MODEL;
}

std::string enumToString(GraphExportType type) {
    auto it = typeToString.find(type);
    return (it != typeToString.end()) ? it->second : "unknown_graph";
}

GraphExportType stringToEnum(const std::string& inString) {
    auto it = stringToType.find(inString);
    return (it != stringToType.end()) ? it->second : UNKNOWN_GRAPH;
}
bool isOptimumCliDownload(const std::string& sourceModel, std::optional<std::string> ggufFilename) {
    // OCI references are resolved by the OciDownloader, which decides on its
    // own whether the resolved payload still needs an optimum-cli conversion.
    return !isOciDownload(sourceModel) && !startsWith(toLower(sourceModel), "openvino/") && (ggufFilename == std::nullopt);
}

bool isOciDownload(const std::string& sourceModel) {
    return startsWith(toLower(sourceModel), OCI_SCHEME);
}

std::string stripOciScheme(const std::string& sourceModel) {
    if (!isOciDownload(sourceModel)) {
        return sourceModel;
    }
    return sourceModel.substr(std::string(OCI_SCHEME).size());
}

std::string localModelDirectoryName(const std::string& sourceModel) {
    if (!isOciDownload(sourceModel)) {
        return sourceModel;
    }
    std::string name = stripOciScheme(sourceModel);
    // ':' separates the tag (and, for a non-default registry port, the port).
    // It is not a legal filename character on Windows.
    std::replace(name.begin(), name.end(), ':', '_');
    return name;
}

}  // namespace ovms
