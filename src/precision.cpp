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

#include "precision.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <unordered_map>

namespace ovms {

std::string toString(Precision precision) {
    static std::unordered_map<Precision, const char*> precisionMap{
        {Precision::BF16, "BF16"},
        {Precision::FP64, "FP64"},
        {Precision::FP32, "FP32"},
        {Precision::FP16, "FP16"},
        {Precision::I64, "I64"},
        {Precision::I32, "I32"},
        {Precision::I16, "I16"},
        {Precision::I8, "I8"},
        {Precision::I4, "I4"},
        {Precision::U64, "U64"},
        {Precision::U32, "U32"},
        {Precision::U16, "U16"},
        {Precision::U8, "U8"},
        {Precision::U4, "U4"},
        {Precision::U1, "U1"},
        {Precision::MIXED, "MIXED"},
        {Precision::Q78, "Q78"},
        {Precision::BIN, "BIN"},
        {Precision::BOOL, "BOOL"},
        {Precision::UNDEFINED, "UNDEFINED"},
        {Precision::CUSTOM, "CUSTOM"}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return "DT_INVALID";  // TODO other way? why translate it to TF equivalent maybe UNDEFINED?
    }
    return it->second;
}

Precision fromString(const std::string& s) {
    static std::unordered_map<std::string, Precision> precisionMap{
        {"BF16", Precision::BF16},
        {"FP64", Precision::FP64},
        {"FP32", Precision::FP32},
        {"FP16", Precision::FP16},
        {"I64", Precision::I64},
        {"I32", Precision::I32},
        {"I16", Precision::I16},
        {"I8", Precision::I8},
        {"I4", Precision::I4},
        {"U64", Precision::U64},
        {"U32", Precision::U32},
        {"U16", Precision::U16},
        {"U8", Precision::U8},
        {"U4", Precision::U4},
        {"U1", Precision::U1},
        {"MIXED", Precision::MIXED},
        {"Q78", Precision::Q78},
        {"BIN", Precision::BIN},
        {"BOOL", Precision::BOOL},
        {"UNDEFINED", Precision::UNDEFINED},
        {"CUSTOM", Precision::CUSTOM}};
    auto it = precisionMap.find(s);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

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
        // {"BYTES", Precision::??}, // TODO decide what to do with BYTES
        {"UINT8", Precision::U8}};
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
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
        // TODO missing precisions
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
        {"FP64", 8}
        // {"BYTES", }, // TODO decide what to do with BYTES
    };
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

KFSDataType ovmsPrecisionToKFSPrecision(Precision precision) {
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
    // {Precision::BF16, ""}, TODO
    // {Precision::U4, ""}, TODO
    // {Precision::U1, ""}, TODO
    // {Precision::CUSTOM, ""}, TODO
    // {Precision::DYNAMIC, ""}, TODO
    // {Precision::MIXED, ""}, TODO
    // {Precision::Q78, ""}, TODO
    // {Precision::BIN, ""}, TODO
    // {Precision::I4, ""}, TODO
    // {Precision::UNDEFINED, "UNDEFINED"}}; TODO
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return "INVALID";
    }
    return it->second;
}

ov::element::Type_t ovmsPrecisionToIE2Precision(Precision precision) {
    static std::unordered_map<Precision, ov::element::Type_t> precisionMap{
        {Precision::FP64, ov::element::Type_t::f64},
        {Precision::FP32, ov::element::Type_t::f32},
        {Precision::FP16, ov::element::Type_t::f16},
        {Precision::I64, ov::element::Type_t::i64},
        {Precision::I32, ov::element::Type_t::i32},
        {Precision::I16, ov::element::Type_t::i16},
        {Precision::I8, ov::element::Type_t::i8},
        {Precision::I4, ov::element::Type_t::i4},
        {Precision::U64, ov::element::Type_t::u64},
        {Precision::U32, ov::element::Type_t::u32},
        {Precision::U16, ov::element::Type_t::u16},
        {Precision::U8, ov::element::Type_t::u8},
        {Precision::U4, ov::element::Type_t::u4},
        {Precision::U1, ov::element::Type_t::u1},
        {Precision::BOOL, ov::element::Type_t::boolean},
        {Precision::UNDEFINED, ov::element::Type_t::undefined},  // TODO
        {Precision::DYNAMIC, ov::element::Type_t::dynamic}       // TODO
        //    {Precision::MIXED, ov::element::Type_t::MIXED}, // TODO
        //    {Precision::Q78, ov::element::Type_t::Q78}, // TODO
        //    {Precision::BIN, ov::element::Type_t::BIN}, // TODO
        //    {Precision::CUSTOM, ov::element::Type_t::CUSTOM // TODO
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::undefined;  // TODO other way?
    }
    return it->second;
}

Precision ovElementTypeToOvmsPrecision(ov::element::Type_t type) {
    static std::unordered_map<ov::element::Type_t, Precision> precisionMap{
        {ov::element::Type_t::f64, Precision::FP64},
        {ov::element::Type_t::f32, Precision::FP32},
        {ov::element::Type_t::f16, Precision::FP16},
        {ov::element::Type_t::bf16, Precision::BF16},
        {ov::element::Type_t::i64, Precision::I64},
        {ov::element::Type_t::i32, Precision::I32},
        {ov::element::Type_t::i16, Precision::I16},
        {ov::element::Type_t::i8, Precision::I8},
        {ov::element::Type_t::i4, Precision::I4},
        {ov::element::Type_t::u64, Precision::U64},
        {ov::element::Type_t::u32, Precision::U32},
        {ov::element::Type_t::u16, Precision::U16},
        {ov::element::Type_t::u8, Precision::U8},
        {ov::element::Type_t::u4, Precision::U4},
        {ov::element::Type_t::u1, Precision::U1},
        {ov::element::Type_t::undefined, Precision::UNDEFINED},
        {ov::element::Type_t::dynamic, Precision::DYNAMIC},
        //    {ov::element::Type_t::???, Precision::MIXED}, // TODO
        //    {ov::element::Type_t::???, Precision::Q78}, // TODO
        //    {ov::element::Type_t::???, Precision::BIN}, // TODO
        {ov::element::Type_t::boolean, Precision::BOOL}
        //    {ov::element::Type_t::CUSTOM, Precision::CUSTOM} // TODO
        /*
    undefined,
    dynamic,
*/
    };
    auto it = precisionMap.find(type);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;  // TODO other way?
    }
    return it->second;
}
}  // namespace ovms
