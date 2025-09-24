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

const std::string& getGenericFullPathForSrcTest(const std::string& linuxPath, bool logChange = true);
const std::string& getGenericFullPathForSrcTest(const char* linuxPath, bool logChange = true);
const std::string& getGenericFullPathForTmp(const std::string& linuxPath, bool logChange = true);
const std::string& getGenericFullPathForTmp(const char* linuxPath, bool logChange = true);
const std::string& getGenericFullPathForBazelOut(const std::string& linuxPath, bool logChange = true);
std::string getOvmsTestExecutablePath();

#ifdef _WIN32
const std::string getWindowsRepoRootPath();
#endif
void adjustConfigForTargetPlatform(std::string& input);
const std::string& adjustConfigForTargetPlatformReturn(std::string& input);
std::string adjustConfigForTargetPlatformCStr(const char* input);
