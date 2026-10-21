//*****************************************************************************
// Copyright 2026 Intel Corporation
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

namespace ovms {

// Loads upstream calculator and option registrations into the host registries.
// The handle is retained for the process lifetime; absence is not a server error.
bool loadOpenCvCalculatorsPlugin();

// Empty after successful loading, otherwise explains the missing OpenCV plugin.
std::string getOpenCvCalculatorsPluginDiagnostic();

}  // namespace ovms
