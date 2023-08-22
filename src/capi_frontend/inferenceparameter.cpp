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

#include "inferenceparameter.hpp"

#include <stdexcept>

#include "../ovms.h"  // NOLINT
namespace ovms {
//
size_t DataTypeToByteSize(OVMS_DataType datatype) {
    static std::unordered_map<OVMS_DataType, size_t> datatypeSizeMap{
        {OVMS_DATATYPE_BOOL, 1},
        {OVMS_DATATYPE_U1, 1},
        {OVMS_DATATYPE_U4, 1},
        {OVMS_DATATYPE_U8, 1},
        {OVMS_DATATYPE_U16, 2},
        {OVMS_DATATYPE_U32, 4},
        {OVMS_DATATYPE_U64, 8},
        {OVMS_DATATYPE_I4, 1},
        {OVMS_DATATYPE_I8, 1},
        {OVMS_DATATYPE_I16, 2},
        {OVMS_DATATYPE_I32, 4},
        {OVMS_DATATYPE_I64, 8},
        {OVMS_DATATYPE_FP16, 2},
        {OVMS_DATATYPE_FP32, 4},
        {OVMS_DATATYPE_FP64, 8},
        {OVMS_DATATYPE_BF16, 2},
        // {"BYTES", },
    };
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

InferenceParameter::InferenceParameter(const char* name, OVMS_DataType datatype, const void* data) :
    name(name),
    datatype(datatype),
    data(reinterpret_cast<const char*>(data), DataTypeToByteSize(datatype)) {
}

const std::string& InferenceParameter::getName() const {
    return this->name;
}
OVMS_DataType InferenceParameter::getDataType() const {
    return this->datatype;
}

const void* InferenceParameter::getData() const {
    return reinterpret_cast<const void*>(this->data.c_str());
}
}  // namespace ovms
