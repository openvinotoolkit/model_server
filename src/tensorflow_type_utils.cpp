//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "tensorflow_type_utils.hpp"

#include <unordered_map>

#include "precision.hpp"

namespace ovms {

Precision TFSPrecisionToOvmsPrecision(const TFSDataType& datatype) {
    static std::unordered_map<TFSDataType, Precision> precisionMap{
        {TFSDataType::DT_FLOAT, Precision::FP32},
        {TFSDataType::DT_DOUBLE, Precision::FP64},
        {TFSDataType::DT_HALF, Precision::FP16},
        {TFSDataType::DT_INT64, Precision::I64},
        {TFSDataType::DT_INT32, Precision::I32},
        {TFSDataType::DT_INT16, Precision::I16},
        {TFSDataType::DT_INT8, Precision::I8},
        {TFSDataType::DT_UINT64, Precision::U64},
        {TFSDataType::DT_UINT16, Precision::U16},
        {TFSDataType::DT_UINT8, Precision::U8},
        {TFSDataType::DT_STRING, Precision::STRING},
        {TFSDataType::DT_BOOL, Precision::BOOL}};
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

TFSDataType getPrecisionAsDataType(Precision precision) {
    static std::unordered_map<Precision, TFSDataType> precisionMap{
        {Precision::FP64, TFSDataType::DT_DOUBLE},
        {Precision::FP32, TFSDataType::DT_FLOAT},
        {Precision::FP16, TFSDataType::DT_HALF},
        {Precision::I16, TFSDataType::DT_INT16},
        {Precision::U8, TFSDataType::DT_UINT8},
        {Precision::I8, TFSDataType::DT_INT8},
        {Precision::U16, TFSDataType::DT_UINT16},
        {Precision::I32, TFSDataType::DT_INT32},
        {Precision::I64, TFSDataType::DT_INT64},
        {Precision::U32, TFSDataType::DT_UINT32},
        {Precision::U64, TFSDataType::DT_UINT64},
        {Precision::BOOL, TFSDataType::DT_BOOL},
        {Precision::STRING, TFSDataType::DT_STRING}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return TFSDataType::DT_INVALID;
    }
    return it->second;
}

}  // namespace ovms
