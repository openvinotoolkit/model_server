//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "capi_utils.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../logging.hpp"
#include "../pocapi.hpp"
#include "../precision.hpp"
#include "../profiler.hpp"
#include "../shape.hpp"
#include "../status.hpp"

namespace ovms {

size_t DataTypeSize(const OVMS_DataType& datatype) {
    static std::unordered_map<OVMS_DataType, size_t> datatypeSizeMap{
        {OVMS_DATATYPE_BOOL, 1},
        {OVMS_DATATYPE_U8, 1},
        {OVMS_DATATYPE_U16, 2},
        {OVMS_DATATYPE_U32, 4},
        {OVMS_DATATYPE_U64, 8},
        {OVMS_DATATYPE_I8, 1},
        {OVMS_DATATYPE_I16, 2},
        {OVMS_DATATYPE_I32, 4},
        {OVMS_DATATYPE_I64, 8},
        {OVMS_DATATYPE_FP16, 2},
        {OVMS_DATATYPE_FP32, 4},
        {OVMS_DATATYPE_FP64, 8}
        // {"BYTES", },
    };
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

std::string tensorShapeToString(const Shape& shape) {
    std::ostringstream oss;
    oss << "(";
    size_t i = 0;
    if (shape.size() > 0) {
        for (; i < shape.size() - 1; i++) {
            oss << shape[i].toString() << ",";
        }
        oss << shape[i].toString();
    }
    oss << ")";

    return oss.str();
}

}  // namespace ovms
