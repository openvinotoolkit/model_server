//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ovms {

// KServe API defines data types
// https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-types
// Struct string-syntax for buffer fromat description
// https://docs.python.org/3/library/struct.html#format-characters

const std::unordered_map<std::string, std::string> datatypeToBufferFormat{
    {"BOOL", "?"},
    {"UINT8", "B"},
    {"UINT16", "H"},
    {"UINT32", "I"},
    {"UINT64", "Q"},
    {"INT8", "b"},
    {"INT16", "h"},
    {"INT32", "i"},
    {"INT64", "q"},
    {"FP16", "e"},
    {"FP32", "f"},
    {"FP64", "d"},
    // {"BF16", X} to be considered, for now it shall be treated as a custom datatype
};

const std::unordered_map<std::string, std::string> bufferFormatToDatatype{
    {"?", "BOOL"},
    {"B", "UINT8"},
    {"H", "UINT16"},
    {"I", "UINT32"},
    {"L", "UINT32"},  // additional entry for unsigned long type
    {"Q", "UINT64"},
    {"b", "INT8"},
    {"h", "INT16"},
    {"i", "INT32"},
    {"l", "INT32"},  // additional entry for long type
    {"q", "INT64"},
    {"e", "FP16"},
    {"f", "FP32"},
    {"d", "FP64"},
    // {X, "BF16"} to be considered, for now it shall be treated as a custom datatype
};

// TO DO: Note that for numpy for example np.int64 gets translates to "l" not "q" on 64 bit linux systems.
// We should consider an alternative to hardcoding those characters if it becomes an issue.

const std::unordered_map<std::string, py::ssize_t> bufferFormatToItemsize{
    {"?", 1},
    {"B", 1},
    {"H", 2},
    {"I", 4},
    {"Q", 8},
    {"b", 1},
    {"h", 2},
    {"i", 4},
    {"q", 8},
    {"e", 2},
    {"f", 4},
    {"d", 8},
    // {"BF16", X} to be considered, for now it shall be treated as a custom datatype
};

const std::string RAW_BINARY_FORMAT = "B";

struct OvmsPyTensor {
    std::string name;
    // Can be one of predefined types (like int8, float32 etc.) or totally custom like numpy (for example "<U83")
    std::string datatype;
    // User defined shape read from the request
    std::vector<py::ssize_t> userShape;
    // Binary size of the input data
    size_t size;

    // Buffer protocol fields
    void* ptr;
    std::vector<py::ssize_t> bufferShape;
    py::ssize_t ndim;
    std::string format;  // Struct-syntax buffer format
    py::ssize_t itemsize;
    std::vector<py::ssize_t> strides;

    // ---

    // Construct object from request contents
    OvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size);

    // Construct object from buffer info
    OvmsPyTensor(const std::string& name, py::buffer_info bufferInfo);
};
}  // namespace ovms
