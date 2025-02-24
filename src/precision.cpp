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

#include <typeinfo>
#include <unordered_map>

namespace ovms {

const std::string& toString(Precision precision) {
    static std::unordered_map<Precision, std::string> precisionMap{
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
        {Precision::STRING, "STRING"},
        {Precision::CUSTOM, "CUSTOM"}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        static const std::string UNKNOWN{"UNKNOWN"};
        return UNKNOWN;
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
        {"STRING", Precision::STRING},
        {"CUSTOM", Precision::CUSTOM}};
    auto it = precisionMap.find(s);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
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
        {Precision::BF16, ov::element::Type_t::bf16},
        {Precision::UNDEFINED, ov::element::Type_t::dynamic},
        {Precision::DYNAMIC, ov::element::Type_t::dynamic},
        {Precision::STRING, ov::element::Type_t::string}
        //    {Precision::MIXED, ov::element::Type_t::MIXED},
        //    {Precision::Q78, ov::element::Type_t::Q78},
        //    {Precision::BIN, ov::element::Type_t::BIN},
        //    {Precision::CUSTOM, ov::element::Type_t::CUSTOM
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::dynamic;
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
        {ov::element::Type_t::string, Precision::STRING},
        {ov::element::Type_t::dynamic, Precision::UNDEFINED},
        //    {ov::element::Type_t::???, Precision::MIXED},
        //    {ov::element::Type_t::???, Precision::Q78},
        //    {ov::element::Type_t::???, Precision::BIN},
        {ov::element::Type_t::boolean, Precision::BOOL}
        //    {ov::element::Type_t::CUSTOM, Precision::CUSTOM}
        /*
    undefined,
    dynamic,
*/
    };
    auto it = precisionMap.find(type);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}
}  // namespace ovms
