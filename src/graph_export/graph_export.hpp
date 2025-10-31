//****************************************************************************
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
#pragma once
#include <optional>
#include <string>
#include <variant>

namespace ovms {
struct PluginConfigSettingsImpl;
struct HFSettingsImpl;
struct ExportSettings;
class Status;

class GraphExport {
public:
    GraphExport();
    Status createServableConfig(const std::string& directoryPath, const HFSettingsImpl& graphSettings);
    static std::variant<std::string, Status> createPluginString(const PluginConfigSettingsImpl& pluginConfig, const ExportSettings& exportSettings);
    static std::string getDraftModelDirectoryName(std::string draftModel);
    static std::string getDraftModelDirectoryPath(const std::string& directoryPath, const std::string& draftModel);
};
}  // namespace ovms
