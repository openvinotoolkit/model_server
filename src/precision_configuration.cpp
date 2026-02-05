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
#include "precision_configuration.hpp"

#include <algorithm>
#include <vector>

namespace ovms {

const char PrecisionConfiguration::PRECISION_DELIMITER = ':';
const std::unordered_map<std::string, ov::element::Type> PrecisionConfiguration::precisionMap = {
    {"BF16", ov::element::bf16},
    {"FP64", ov::element::f64},
    {"FP32", ov::element::f32},
    {"FP16", ov::element::f16},
    {"INT64", ov::element::i64},
    {"INT32", ov::element::i32},
    {"INT16", ov::element::i16},
    {"INT8", ov::element::i8},
    {"INT4", ov::element::i4},
    {"UINT64", ov::element::u64},
    {"UINT32", ov::element::u32},
    {"UINT16", ov::element::u16},
    {"UINT8", ov::element::u8},
    {"UINT4", ov::element::u4},
    {"UINT1", ov::element::u1}
};

Status PrecisionConfiguration::stringToPrecision(const std::string& precisionStr, ov::element::Type& precisionOut) {
    auto it = precisionMap.find(precisionStr);
    if (it != precisionMap.end()) {
        precisionOut = it->second;
        return StatusCode::OK;
    } else {
        return StatusCode::PRECISION_WRONG_FORMAT;
    }
}

Status PrecisionConfiguration::fromString(const std::string& configurationStr, PrecisionConfiguration& configOut) {
    std::string upperConfigurationStr;
    std::transform(configurationStr.begin(), configurationStr.end(), std::back_inserter(upperConfigurationStr), ::toupper);

    size_t delimiterPos = upperConfigurationStr.find(PRECISION_DELIMITER);
    if (delimiterPos == std::string::npos) {
        SPDLOG_ERROR("Invalid precision configuration string: {}", configurationStr);
        return StatusCode::PRECISION_WRONG_FORMAT;
    }
    ov::element::Type targetPrecision;
    std::string targetPrecisionStr = upperConfigurationStr.substr(0, delimiterPos);

    Status status = stringToPrecision(targetPrecisionStr, targetPrecision);
    if (status != StatusCode::OK) {
        SPDLOG_ERROR("Invalid precision configuration string: {}", configurationStr);
        return status;
    }

    ov::element::Type sourcePrecision;
    std::string sourcePrecisionStr = upperConfigurationStr.substr(delimiterPos + 1);
    status = stringToPrecision(sourcePrecisionStr, sourcePrecision);
    if (status != StatusCode::OK) {
        SPDLOG_ERROR("Invalid precision configuration string: {}", configurationStr);
        return status;
    }
    configOut = PrecisionConfiguration(targetPrecision, sourcePrecision);

    return StatusCode::OK;
}
}  // namespace ovms
