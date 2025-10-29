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
#pragma once
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

#include "base_output_parser.hpp"

namespace ovms {
// Generates random alphanumeric string of length 9 for tool call ID
std::string generateRandomId();

// Common function to parse tool calls from a JSON array
// Returns true if parsing was successful, false otherwise
// The toolCalls vector will be populated with parsed tool calls
bool parseToolCallsFromJsonArray(const rapidjson::Document& toolsDoc, ToolCalls_t& toolCalls);
}  // namespace ovms
