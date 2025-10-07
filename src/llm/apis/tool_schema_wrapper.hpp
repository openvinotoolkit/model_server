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

// Type that holds vector of pairs where first element is chat turn index and second is image tensor
// this way we store information about which image is associated with which chat turn
#pragma once
#include <map>
#include <string>

#include "src/port/rapidjson_document.hpp"

namespace ovms {
struct ToolSchemaWrapper {
    rapidjson::Value* rapidjsonRepr;
    std::string stringRepr;
};
using ToolsSchemas_t = std::map<std::string, ToolSchemaWrapper>;
}  // namespace ovms
