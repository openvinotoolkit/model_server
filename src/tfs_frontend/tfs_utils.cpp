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
#include "tfs_utils.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../logging.hpp"

namespace ovms {

tensorflow::DataType getPrecisionAsDataType(Precision precision) {
    static std::unordered_map<Precision, tensorflow::DataType> precisionMap{
        {Precision::FP32, tensorflow::DataType::DT_FLOAT},
        {Precision::FP64, tensorflow::DataType::DT_DOUBLE},
        {Precision::FP16, tensorflow::DataType::DT_HALF},
        {Precision::I64, tensorflow::DataType::DT_INT64},
        {Precision::I32, tensorflow::DataType::DT_INT32},
        {Precision::I16, tensorflow::DataType::DT_INT16},
        {Precision::I8, tensorflow::DataType::DT_INT8},
        {Precision::U64, tensorflow::DataType::DT_UINT64},
        {Precision::U16, tensorflow::DataType::DT_UINT16},
        {Precision::U8, tensorflow::DataType::DT_UINT8},
        //    {Precision::MIXED, tensorflow::DataType::DT_INVALID},
        //    {Precision::Q78, tensorflow::DataType::DT_INVALID},
        //    {Precision::BIN, tensorflow::DataType::DT_INVALID},
        {Precision::BOOL, tensorflow::DataType::DT_BOOL}
        //    {Precision::CUSTOM, tensorflow::DataType::DT_INVALID}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return tensorflow::DataType::DT_INVALID;
    }
    return it->second;
}

std::string getDataTypeAsString(tensorflow::DataType dataType) {
    switch (dataType) {
    case tensorflow::DataType::DT_FLOAT:
        return "FP32";
    case tensorflow::DataType::DT_DOUBLE:
        return "FP64";
    case tensorflow::DataType::DT_INT32:
        return "I32";
    case tensorflow::DataType::DT_INT8:
        return "I8";
    case tensorflow::DataType::DT_UINT8:
        return "U8";
    case tensorflow::DataType::DT_HALF:
        return "FP16";
    case tensorflow::DataType::DT_INT16:
        return "I16";
    case tensorflow::DataType::DT_UINT16:
        return "U16";
    case tensorflow::DataType::DT_UINT64:
        return "U64";
    case tensorflow::DataType::DT_INT64:
        return "I64";
    case tensorflow::DataType::DT_BOOL:
        return "BOOL";
    case tensorflow::DataType::DT_STRING:
        return "STRING";
    default:
        return "INVALID";
    }
}

std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape) {
    std::ostringstream oss;
    oss << "(";
    int i = 0;
    if (tensorShape.dim_size() > 0) {
        for (; i < tensorShape.dim_size() - 1; i++) {
            oss << tensorShape.dim(i).size() << ",";
        }
        oss << tensorShape.dim(i).size();
    }
    oss << ")";

    return oss.str();
}
}  // namespace ovms
