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
#include "kfs_utils.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
Precision KFSPrecisionToOvmsPrecision(const KFSDataType& datatype) {
    static std::unordered_map<KFSDataType, Precision> precisionMap{
        {"BOOL", Precision::BOOL},
        {"FP64", Precision::FP64},
        {"FP32", Precision::FP32},
        {"FP16", Precision::FP16},
        {"INT64", Precision::I64},
        {"INT32", Precision::I32},
        {"INT16", Precision::I16},
        {"INT8", Precision::I8},
        {"UINT64", Precision::U64},
        {"UINT32", Precision::U32},
        {"UINT16", Precision::U16},
        // {"BYTES", Precision::??},
        {"UINT8", Precision::U8}};
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

size_t KFSDataTypeSize(const KFSDataType& datatype) {
    static std::unordered_map<KFSDataType, size_t> datatypeSizeMap{
        {"BOOL", 1},
        {"UINT8", 1},
        {"UINT16", 2},
        {"UINT32", 4},
        {"UINT64", 8},
        {"INT8", 1},
        {"INT16", 2},
        {"INT32", 4},
        {"INT64", 8},
        {"FP16", 2},
        {"FP32", 4},
        {"FP64", 8},
        {"BYTES", 1}};
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

const KFSDataType& ovmsPrecisionToKFSPrecision(Precision precision) {
    static std::unordered_map<Precision, KFSDataType> precisionMap{
        {Precision::FP64, "FP64"},
        {Precision::FP32, "FP32"},
        {Precision::FP16, "FP16"},
        {Precision::I64, "INT64"},
        {Precision::I32, "INT32"},
        {Precision::I16, "INT16"},
        {Precision::I8, "INT8"},
        {Precision::U64, "UINT64"},
        {Precision::U32, "UINT32"},
        {Precision::U16, "UINT16"},
        {Precision::U8, "UINT8"},
        {Precision::BOOL, "BOOL"}};
    // {Precision::BF16, ""},
    // {Precision::U4, ""},
    // {Precision::U1, ""},
    // {Precision::CUSTOM, ""},
    // {Precision::DYNAMIC, ""},
    // {Precision::MIXED, ""},
    // {Precision::Q78, ""},
    // {Precision::BIN, ""},
    // {Precision::I4, ""},
    // {Precision::UNDEFINED, "UNDEFINED"}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        static const std::string invalid{"INVALID"};
        return invalid;
    }
    return it->second;
}

std::string tensorShapeToString(const KFSShapeType& shape) {
    std::ostringstream oss;
    oss << "(";
    int i = 0;
    if (shape.size() > 0) {
        for (; i < shape.size() - 1; i++) {
            oss << shape[i] << ",";
        }
        oss << shape[i];
    }
    oss << ")";

    return oss.str();
}

Status prepareConsolidatedTensorImpl(KFSResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& bufferOut, size_t size) {
    OVMS_PROFILE_FUNCTION();
    for (int i = 0; i < response->outputs_size(); i++) {
        if (response->mutable_outputs(i)->name() == name) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, tensor with name {} already prepared", name);
            return StatusCode::INTERNAL_ERROR;
        }
    }
    auto* proto = response->add_outputs();
    proto->set_name(name);
    auto* content = response->add_raw_output_contents();
    content->resize(size);
    bufferOut = content->data();
    return StatusCode::OK;
}
const std::string& getRequestServableName(const KFSRequest& request) {
    return request.model_name();
}
Status isNativeFileFormatUsed(const KFSRequest& request, const std::string& name, bool& nativeFileFormatUsed) {
    auto it = request.inputs().begin();
    while (it != request.inputs().end()) {
        if (it->name() == name) {
            break;
        }
        ++it;
    }
    if (it == request.inputs().end()) {
        SPDLOG_ERROR("Error during checking binary input; input: {} does not exist for request: {}", name, getRequestServableName(request));
        return StatusCode::INTERNAL_ERROR;
    }
    nativeFileFormatUsed = isNativeFileFormatUsed(*it);
    return StatusCode::OK;
}

bool isNativeFileFormatUsed(const KFSTensorInputProto& proto) {
    return proto.datatype() == "BYTES";
}

bool isStringFormatUsed(const KFSTensorInputProto& proto, const TensorInfo& tensorInfo) {
    return proto.datatype() == "BYTES" && tensorInfo.getProcessingHint() == TensorInfo::ProcessingHint::STRING;
}

bool hasString(const KFSTensorInputProto& proto) {
    return proto.datatype() == "BYTES";
}
}  // namespace ovms
