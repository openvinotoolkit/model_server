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

#include <inference_engine.hpp>
#include <unordered_map>

namespace ovms {

enum class Precision {
FP32,
I32,
I8,
U8,
I16,
FP16,
U16,
I64,
MIXED,
Q78,
BIN,
BOOL,
CUSTOM,
};
inline static InferenceEngine::Precision ovmsPrecisionToIE1Precision(Precision precision) {
    static std::unordered_map<Precision, InferenceEngine::Precision> precisionMap {
    {Precision::FP32, InferenceEngine::Precision::FP32},
    {Precision::I32, InferenceEngine::Precision::I32},
    {Precision::I8, InferenceEngine::Precision::I8},
    {Precision::U8, InferenceEngine::Precision::U8},
    {Precision::I16, InferenceEngine::Precision::I16},
    {Precision::FP16, InferenceEngine::Precision::FP16},
    {Precision::U16, InferenceEngine::Precision::U16},
    {Precision::I64, InferenceEngine::Precision::I64},
    {Precision::MIXED, InferenceEngine::Precision::MIXED},
    {Precision::Q78, InferenceEngine::Precision::Q78},
    {Precision::BIN, InferenceEngine::Precision::BIN},
    {Precision::BOOL, InferenceEngine::Precision::BOOL},
    {Precision::CUSTOM, InferenceEngine::Precision::CUSTOM}
};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return InferenceEngine::Precision::CUSTOM; // TODO other way?
    }
    return it->second;
}
}  // namespace ovms
