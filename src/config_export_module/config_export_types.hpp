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
#include <map>
#include <string>
#pragma once
namespace ovms {
enum ConfigExportType : int {
    ENABLE_MODEL,
    DISABLE_MODEL,
    DELETE_MODEL,
    UNKNOWN_MODEL
};

const std::map<ConfigExportType, std::string> configExportTypeToString = {
    {ENABLE_MODEL, "ENABLE_MODEL"},
    {DISABLE_MODEL, "DISABLE_MODEL"},
    {DELETE_MODEL, "DELETE_MODEL"},
    {UNKNOWN_MODEL, "UNKNOWN_MODEL"}};

const std::map<std::string, ConfigExportType> stringToConfigExportType = {
    {"ENABLE_MODEL", ENABLE_MODEL},
    {"DISABLE_MODEL", DISABLE_MODEL},
    {"DELETE_MODEL", DELETE_MODEL},
    {"UNKNOWN_MODEL", UNKNOWN_MODEL}};

std::string enumToString(ConfigExportType type);
ConfigExportType stringToConfigExportEnum(std::string inString);

}  // namespace ovms
