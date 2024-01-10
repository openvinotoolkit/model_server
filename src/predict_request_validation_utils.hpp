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

#include <limits>
#include <set>
#include <string>
#include <vector>

#include <google/protobuf/map.h>

#include "modelversion.hpp"
#include "shape.hpp"
#include "tensorinfo.hpp"

namespace ovms {
class Status;
namespace request_validation_utils {

template <typename RequestType>
Status validate(
    const RequestType& request,
    const tensor_map_t& inputsInfo,
    const std::string& servableName,
    const model_version_t servableVersion,
    const std::set<std::string>& optionalAllowedInputNames = {},
    const Mode batchingMode = Mode::FIXED,
    const shapes_info_map_t& shapeInfo = shapes_info_map_t());

// This function is expected to be called with already validated shape that does not contain negative dimensions
template <typename T>
static bool computeExpectedBufferSizeReturnFalseIfOverflow(const std::vector<T>& shape, const size_t& itemsize, size_t& expectedBufferSize) {
    expectedBufferSize = 1;
    for (const T& dim : shape) {
        if (expectedBufferSize > std::numeric_limits<size_t>::max() / dim)
            return false;
        expectedBufferSize *= dim;
    }
    if (expectedBufferSize > std::numeric_limits<size_t>::max() / itemsize)
        return false;
    expectedBufferSize *= itemsize;
    return true;
}

}  // namespace request_validation_utils
}  // namespace ovms
