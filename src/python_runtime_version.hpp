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
#include <vector>

namespace ovms {

// Detects the Python ABI tag (e.g. "312", "313") that should be used to select
// version-suffixed Python runtime libraries such as libovmspython-cp313.dll,
// libpython_calculators-cp313.dll or pyovms-cp313.pyd.
//
// Detection order:
//   1. OVMS_PYTHON_ABI environment variable override (e.g. "313").
//   2. PYTHONHOME environment variable, parsed for a "python"/"Python" version
//      marker (e.g. C:\opt\Python313 -> "313", /usr/lib/python3.13 -> "313").
//
// Returns an empty string when no version marker could be determined; callers
// should then rely solely on the unsuffixed (version-agnostic) library names.
std::string detectPythonAbiTag();

// Given a list of candidate library paths (e.g. "libovmspython.dll",
// "src\\python\\libovmspython.dll"), returns a new list with version-suffixed
// variants (e.g. "libovmspython-cp313.dll") inserted ahead of each original
// entry when a Python ABI tag is detected via detectPythonAbiTag(). When no
// tag is detected, the original list is returned unchanged so single-ABI
// builds/packages keep working without modification.
std::vector<std::string> withAbiVersionedCandidates(const std::vector<std::string>& baseCandidates);

}  // namespace ovms
