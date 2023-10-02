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
#include "capi_utils.hpp"
namespace ovms {
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
