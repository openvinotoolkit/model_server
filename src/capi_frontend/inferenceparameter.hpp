#pragma once
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
#include <string>
#include <unordered_map>

#include "../ovms.h"  // NOLINT

namespace ovms {
size_t DataTypeToByteSize(OVMS_DataType datatype);

class InferenceParameter {
    const std::string name;
    OVMS_DataType datatype;
    const std::string data;

public:
    InferenceParameter(const char* name, OVMS_DataType datatype, const void* data);
    InferenceParameter(const char* name, OVMS_DataType datatype, const void* data, size_t byteSize);
    const std::string& getName() const;
    OVMS_DataType getDataType() const;
    const void* getData() const;
};
}  // namespace ovms
