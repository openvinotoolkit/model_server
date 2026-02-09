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
#pragma once
#include "status.hpp"

#include <openvino/openvino.hpp>
#include <unordered_map>
#include <string>

namespace ovms {
class PrecisionConfiguration {
private:
    static const std::unordered_map<std::string, ov::element::Type> precisionMap;

    ov::element::Type targetPrecision;
    ov::element::Type sourcePrecision;

public:
    static const char PRECISION_DELIMITER;
    PrecisionConfiguration() = default;
    PrecisionConfiguration(ov::element::Type target, ov::element::Type source) :
        targetPrecision(target),
        sourcePrecision(source) {}
    PrecisionConfiguration(const std::string& precision) { fromString(precision, *this); }
    static Status fromString(const std::string& configurationStr, PrecisionConfiguration& configOut);
    static Status stringToPrecision(const std::string& precisionStr, ov::element::Type& precisionOut);

    const ov::element::Type& getTargetPrecision() const {
        return targetPrecision;
    }
    const ov::element::Type& getSourcePrecision() const {
        return sourcePrecision;
    }

    PrecisionConfiguration& operator=(const PrecisionConfiguration& other) {
        this->sourcePrecision = other.sourcePrecision;
        this->targetPrecision = other.targetPrecision;
        return *this;
    }
};
}  // namespace ovms
