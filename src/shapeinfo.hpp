//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <vector>

namespace ovms {

enum Mode { FIXED,
    AUTO };
using shape_t = std::vector<size_t>;

struct ShapeInfo {
    Mode shapeMode = FIXED;
    shape_t shape;

    operator std::string() const;

    bool operator==(const ShapeInfo& rhs) const {
        return this->shapeMode == rhs.shapeMode && this->shape == rhs.shape;
    }

    bool operator!=(const ShapeInfo& rhs) const {
        return !(*this == rhs);
    }
};

using shapes_map_t = std::unordered_map<std::string, ShapeInfo>;
}  // namespace ovms
