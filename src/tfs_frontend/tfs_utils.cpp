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
#include <utility>

#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"

namespace ovms {

TFSDataType getPrecisionAsDataType(Precision precision) {
    static std::unordered_map<Precision, TFSDataType> precisionMap{
        {Precision::FP32, TFSDataType::DT_FLOAT},
        {Precision::FP64, TFSDataType::DT_DOUBLE},
        {Precision::FP16, TFSDataType::DT_HALF},
        {Precision::I64, TFSDataType::DT_INT64},
        {Precision::I32, TFSDataType::DT_INT32},
        {Precision::I16, TFSDataType::DT_INT16},
        {Precision::I8, TFSDataType::DT_INT8},
        {Precision::U64, TFSDataType::DT_UINT64},
        {Precision::U16, TFSDataType::DT_UINT16},
        {Precision::U8, TFSDataType::DT_UINT8},
        //    {Precision::MIXED, TFSDataType::DT_INVALID},
        //    {Precision::Q78, TFSDataType::DT_INVALID},
        //    {Precision::BIN, TFSDataType::DT_INVALID},
        {Precision::BOOL, TFSDataType::DT_BOOL}
        //    {Precision::CUSTOM, TFSDataType::DT_INVALID}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return TFSDataType::DT_INVALID;
    }
    return it->second;
}

std::string getDataTypeAsString(TFSDataType dataType) {
    switch (dataType) {
    case TFSDataType::DT_FLOAT:
        return "FP32";
    case TFSDataType::DT_DOUBLE:
        return "FP64";
    case TFSDataType::DT_INT32:
        return "I32";
    case TFSDataType::DT_INT8:
        return "I8";
    case TFSDataType::DT_UINT8:
        return "U8";
    case TFSDataType::DT_HALF:
        return "FP16";
    case TFSDataType::DT_INT16:
        return "I16";
    case TFSDataType::DT_UINT16:
        return "U16";
    case TFSDataType::DT_UINT64:
        return "U64";
    case TFSDataType::DT_INT64:
        return "I64";
    case TFSDataType::DT_BOOL:
        return "BOOL";
    case TFSDataType::DT_STRING:
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
        {TFSDataType::DT_BOOL, Precision::BOOL}};
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

Status prepareConsolidatedTensorImpl(TFSPredictResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& bufferOut, size_t size) {
    OVMS_PROFILE_FUNCTION();
    tensorflow::TensorProto tensorProto;
    auto [it, isInserted] = response->mutable_outputs()->insert(google::protobuf::MapPair<std::string, tensorflow::TensorProto>(name, std::move(tensorProto)));
    if (!isInserted) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, tensor with name {} already prepared", name);
        return StatusCode::INTERNAL_ERROR;
    }
    it->second.mutable_tensor_content()->resize(size);
    bufferOut = it->second.mutable_tensor_content()->data();
    return StatusCode::OK;
}
const std::string& getRequestServableName(const TFSPredictRequest& request) {
    return request.model_spec().name();
}
Status isNativeFileFormatUsed(const TFSPredictRequest& request, const std::string& name, bool& nativeFileFormatUsed) {
    auto it = request.inputs().find(name);
    if (it == request.inputs().end()) {
        SPDLOG_DEBUG("Error during checking binary input; input: {} does not exist in request for: {}", name, getRequestServableName(request));
        return StatusCode::INTERNAL_ERROR;
    }
    nativeFileFormatUsed = isNativeFileFormatUsed(it->second);
    return StatusCode::OK;
}

bool isNativeFileFormatUsed(const TFSInputTensorType& proto) {
    return proto.dtype() == TFSDataType::DT_STRING;
    // return request.string_val_size() > 0;
}

bool requiresPreProcessing(const TFSInputTensorType& proto) {
    return proto.dtype() == tensorflow::DataType::DT_STRING;
}

std::string& createOrGetString(TFSInputTensorType& proto, int index) {
    while (proto.string_val_size() <= index) {
        proto.add_string_val();
    }
    return *proto.mutable_string_val(index);
}

void setBatchSize(TFSInputTensorType& proto, int64_t batch) {
    if (proto.tensor_shape().dim_size() == 0) {
        proto.mutable_tensor_shape()->add_dim();
    }
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(batch);
}
void setStringPrecision(TFSInputTensorType& proto) {
    proto.set_dtype(TFSDataType::DT_STRING);
}
}  // namespace ovms
