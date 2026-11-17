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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace ovms {
class GgufMetadataReader {
    std::ifstream file;
    uint64_t remaining;

public:
    explicit GgufMetadataReader(const std::string& path);

    void skip(uint64_t bytes);
    uint64_t readInteger(std::size_t bytes);
    std::string readString();
    void skipValue(uint64_t type);
    std::string readArchitecture();
};
}  // namespace ovms
