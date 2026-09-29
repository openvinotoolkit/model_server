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

#include <optional>
#include <string>
#include <vector>

namespace ovms {

bool graphPbtxtExists(const std::string& modelPath);
bool hasTaskSpecificParameters(const std::vector<std::string>& unmatchedOptions);

// Returns the detected task name, or std::nullopt if the task could not be
// determined. Logs the reason at INFO level (stdout) when returning nullopt.
std::optional<std::string> determineDefaultTaskParameter(const std::optional<std::string>& modelPath, const std::optional<std::string>& sourceModel, const std::optional<std::string>& modelRepositoryPath);

}  // namespace ovms
