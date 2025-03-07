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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "../logging.hpp"
#include "../status.hpp"
#include "buffer.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"

namespace ovms {
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

template <>
std::string tensorShapeToString(const signed_shape_t& shape) {
    return shapeToString(shape);
}

OVMS_DataType getPrecisionAsOVMSDataType(Precision precision) {
    switch (precision) {
    case Precision::BF16:
        return OVMS_DATATYPE_BF16;
    case Precision::FP64:
        return OVMS_DATATYPE_FP64;
    case Precision::FP32:
        return OVMS_DATATYPE_FP32;
    case Precision::FP16:
        return OVMS_DATATYPE_FP16;
    case Precision::I64:
        return OVMS_DATATYPE_I64;
    case Precision::I32:
        return OVMS_DATATYPE_I32;
    case Precision::I16:
        return OVMS_DATATYPE_I16;
    case Precision::I8:
        return OVMS_DATATYPE_I8;
    case Precision::I4:
        return OVMS_DATATYPE_I4;
    case Precision::U64:
        return OVMS_DATATYPE_U64;
    case Precision::U32:
        return OVMS_DATATYPE_U32;
    case Precision::U16:
        return OVMS_DATATYPE_U16;
    case Precision::U8:
        return OVMS_DATATYPE_U8;
    case Precision::U4:
        return OVMS_DATATYPE_U4;
    case Precision::U1:
        return OVMS_DATATYPE_U1;
    case Precision::BOOL:
        return OVMS_DATATYPE_BOOL;
    case Precision::CUSTOM:
        return OVMS_DATATYPE_CUSTOM;
    case Precision::UNDEFINED:
        return OVMS_DATATYPE_UNDEFINED;
    case Precision::DYNAMIC:
        return OVMS_DATATYPE_DYNAMIC;
    case Precision::MIXED:
        return OVMS_DATATYPE_MIXED;
    case Precision::Q78:
        return OVMS_DATATYPE_Q78;
    case Precision::STRING:
        return OVMS_DATATYPE_STRING;
    case Precision::BIN:
        return OVMS_DATATYPE_BIN;
    default:
        return OVMS_DATATYPE_UNDEFINED;
    }
}
Precision getOVMSDataTypeAsPrecision(OVMS_DataType datatype) {
    switch (datatype) {
    case OVMS_DATATYPE_BF16:
        return Precision::BF16;
    case OVMS_DATATYPE_FP64:
        return Precision::FP64;
    case OVMS_DATATYPE_FP32:
        return Precision::FP32;
    case OVMS_DATATYPE_FP16:
        return Precision::FP16;
    case OVMS_DATATYPE_I64:
        return Precision::I64;
    case OVMS_DATATYPE_I32:
        return Precision::I32;
    case OVMS_DATATYPE_I16:
        return Precision::I16;
    case OVMS_DATATYPE_I8:
        return Precision::I8;
    case OVMS_DATATYPE_I4:
        return Precision::I4;
    case OVMS_DATATYPE_U64:
        return Precision::U64;
    case OVMS_DATATYPE_U32:
        return Precision::U32;
    case OVMS_DATATYPE_U16:
        return Precision::U16;
    case OVMS_DATATYPE_U8:
        return Precision::U8;
    case OVMS_DATATYPE_U4:
        return Precision::U4;
    case OVMS_DATATYPE_U1:
        return Precision::U1;
    case OVMS_DATATYPE_STRING:
        return Precision::STRING;
    case OVMS_DATATYPE_BOOL:
        return Precision::BOOL;
    case OVMS_DATATYPE_CUSTOM:
        return Precision::CUSTOM;
    case OVMS_DATATYPE_UNDEFINED:
        return Precision::UNDEFINED;
    case OVMS_DATATYPE_DYNAMIC:
        return Precision::DYNAMIC;
    case OVMS_DATATYPE_MIXED:
        return Precision::MIXED;
    case OVMS_DATATYPE_Q78:
        return Precision::Q78;
    case OVMS_DATATYPE_BIN:
        return Precision::BIN;
    default:
        return Precision::UNDEFINED;
    }
}
Status isNativeFileFormatUsed(const InferenceRequest& request, const std::string& name, bool& nativeFileFormatUsed) {
    nativeFileFormatUsed = false;
    return StatusCode::OK;
}
const std::string& getRequestServableName(const ovms::InferenceRequest& request) {
    return request.getServableName();
}
bool requiresPreProcessing(const InferenceTensor& tensor) {
    return false;
}
int getBinaryInputsSize(const InferenceTensor& tensor) {
    throw std::runtime_error("Not implemented");
}
const std::string& getBinaryInput(const InferenceTensor& tensor, size_t i) {
    throw std::runtime_error("Not implemented");
}
Status validateTensor(const TensorInfo& tensorInfo,
    const InferenceTensor& src,
    const std::string* buffer) {
    return StatusCode::NOT_IMPLEMENTED;
}
}  // namespace ovms
