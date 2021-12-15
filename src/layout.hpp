//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <unordered_map>

#include "status.hpp"

namespace ovms {

class LayoutConfiguration {
    std::string tensor;
    std::string model;

public:
    LayoutConfiguration() = default;
    LayoutConfiguration(const char* modelLayout);
    LayoutConfiguration(const std::string& modelLayout);
    LayoutConfiguration(const std::string& tensorLayout, const std::string& modelLayout);

    const std::string& getTensorLayout() const;
    const std::string& getModelLayout() const;

    bool isSet() const;

    bool operator==(const LayoutConfiguration& rhs) const;
    bool operator!=(const LayoutConfiguration& rhs) const;

    static Status fromString(const std::string& configuration, LayoutConfiguration& configOut);
};

using layouts_map_2_t = std::unordered_map<std::string, LayoutConfiguration>;

}  // namespace ovms
