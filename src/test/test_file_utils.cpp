//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include "test_file_utils.hpp"

namespace fs = std::filesystem;

// Create a unique temporary directory inside the system temp directory.
fs::path createTempDir() {
    const fs::path base = fs::temp_directory_path();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    // Try a reasonable number of times to avoid rare collisions
    for (int attempt = 0; attempt < 100; ++attempt) {
        auto candidate = base / ("lfs_kw_tests_" + std::to_string(dist(gen)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
        // If creation failed due to existing path, loop and try another name
        // Otherwise (e.g., permissions), fall through and try again up to limit
    }

    throw std::runtime_error("Failed to create a unique temporary directory");
}

fs::path writeFile(const fs::path& dir, const std::string& name, const std::string& content) {
    fs::path p = dir / name;
    std::ofstream out(p, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to create file: " + p.string());
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return p;
}

void mkdirs(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
}

// A helper for writing test files.
fs::path writeTempFile(const std::string& filename,
    const std::string& content) {
    fs::path p = fs::temp_directory_path() / filename;
    std::ofstream out(p, std::ios::binary);
    out << content;
    return p;
}

// A simple RAII for a temp directory
TempDir::TempDir() :
    dir(createTempDir()) {
    if (dir.empty())
        throw std::runtime_error("Failed to create temp directory");
}

TempDir::~TempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
}
