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
#pragma once

#include <map>  // TODO remove
#include <string>
#include <unordered_map>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>

namespace ovms {

enum class Precision {
    BF16,
    FP64,
    FP32,
    FP16,
    I64,
    I32,
    I16,
    I8,
    I4,
    U64,
    U32,
    U16,
    U8,
    U4,
    U1,
    BOOL,
    CUSTOM,
    UNDEFINED,
    DYNAMIC,
    MIXED,
    Q78,
    BIN
};

inline static std::string toString(Precision precision) {
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

inline static Precision fromString(const std::string& s) {
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

inline static InferenceEngine::Precision ovmsPrecisionToIE1Precision(Precision precision) {
    static std::unordered_map<Precision, InferenceEngine::Precision> precisionMap{
        {Precision::BF16, InferenceEngine::Precision::BF16},
        {Precision::FP64, InferenceEngine::Precision::FP64},
        {Precision::FP32, InferenceEngine::Precision::FP32},
        {Precision::FP16, InferenceEngine::Precision::FP16},
        {Precision::I64, InferenceEngine::Precision::I64},
        {Precision::I32, InferenceEngine::Precision::I32},
        {Precision::I16, InferenceEngine::Precision::I16},
        {Precision::I8, InferenceEngine::Precision::I8},
        {Precision::U64, InferenceEngine::Precision::U64},
        {Precision::U16, InferenceEngine::Precision::U16},
        {Precision::U8, InferenceEngine::Precision::U8},
        {Precision::BOOL, InferenceEngine::Precision::BOOL},
        {Precision::MIXED, InferenceEngine::Precision::MIXED},
        {Precision::Q78, InferenceEngine::Precision::Q78},
        {Precision::BIN, InferenceEngine::Precision::BIN},
        {Precision::UNDEFINED, InferenceEngine::Precision::UNSPECIFIED},
        {Precision::CUSTOM, InferenceEngine::Precision::CUSTOM}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return InferenceEngine::Precision::CUSTOM;  // TODO other way?
    }
    return it->second;
}

inline static ov::element::Type_t ovmsPrecisionToIE2Precision(Precision precision) {
    static std::unordered_map<Precision, ov::element::Type_t> precisionMap{
        {Precision::FP32, ov::element::Type_t::f32},
        {Precision::FP16, ov::element::Type_t::f16},
        {Precision::I64, ov::element::Type_t::i64},
        {Precision::I32, ov::element::Type_t::i32},
        {Precision::I16, ov::element::Type_t::i16},
        {Precision::I8, ov::element::Type_t::i8},
        {Precision::I4, ov::element::Type_t::i4},
        {Precision::U64, ov::element::Type_t::u64},
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
inline static Precision ovElementTypeToOvmsPrecision(ov::element::Type_t type) {
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
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u4,
    u8,
    u16,
    u32,
*/
    };
    auto it = precisionMap.find(type);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;  // TODO other way?
    }
    return it->second;
}
inline static Precision IE1PrecisionToOvmsPrecision(InferenceEngine::Precision precision) {
    static std::map<InferenceEngine::Precision, Precision> precisionMap{
        {InferenceEngine::Precision::BF16, Precision::BF16},
        {InferenceEngine::Precision::FP64, Precision::FP64},
        {InferenceEngine::Precision::FP32, Precision::FP32},
        {InferenceEngine::Precision::FP16, Precision::FP16},
        {InferenceEngine::Precision::I64, Precision::I64},
        {InferenceEngine::Precision::I32, Precision::I32},
        {InferenceEngine::Precision::I16, Precision::I16},
        {InferenceEngine::Precision::I8, Precision::I8},
        {InferenceEngine::Precision::U64, Precision::U64},
        {InferenceEngine::Precision::U16, Precision::U16},
        {InferenceEngine::Precision::U8, Precision::U8},
        {InferenceEngine::Precision::BOOL, Precision::BOOL},
        {InferenceEngine::Precision::MIXED, Precision::MIXED},
        {InferenceEngine::Precision::Q78, Precision::Q78},
        {InferenceEngine::Precision::BIN, Precision::BIN},
        {InferenceEngine::Precision::UNSPECIFIED, Precision::UNDEFINED},
        {InferenceEngine::Precision::CUSTOM, Precision::CUSTOM}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;  // TODO other way?
    }
    return it->second;
}
}  // namespace ovms
