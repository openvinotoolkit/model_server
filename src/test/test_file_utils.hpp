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

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

// Create a unique temporary directory inside the system temp directory.
fs::path createTempDir();

fs::path writeFile(const fs::path& dir, const std::string& name, const std::string& content);

// A helper for writing test files.
fs::path writeTempFile(const std::string& filename,
    const std::string& content);

void mkdirs(const fs::path& p);

// A simple RAII for a temp directory
class TempDir {
public:
    fs::path dir;
    TempDir();
    ~TempDir();
};
