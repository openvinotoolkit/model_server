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

#include "pocapi.hpp"

namespace ovms {

// TODO should we own our own copy of value?
class InferenceParameter {
    const std::string name;
    DataType datatype;
    const std::string data;

public:
    InferenceParameter(const char* name, DataType datatype, const void* data);
    InferenceParameter(const char* name, DataType datatype, const void* data, size_t byteSize);
    const std::string& getName() const;
    DataType getDataType() const;
    size_t getByteSize() const;
    const void* getData() const;
};
}  // namespace ovms
