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

#include "graph_export_types.hpp"
namespace ovms {

std::string enumToString(ExportType type) {
    auto it = typeToString.find(type);
    return (it != typeToString.end()) ? it->second : "unknown";
}

ExportType stringToEnum(std::string inString) {
    auto it = stringToType.find(inString);
    return (it != stringToType.end()) ? it->second : unknown;
}
}  // namespace ovms
