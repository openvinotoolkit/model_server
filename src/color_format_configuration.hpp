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
class ColorFormatConfiguration {
private:
    static const std::unordered_map<std::string, ov::preprocess::ColorFormat> colorFormatMap;

    ov::preprocess::ColorFormat targetColorFormat;
    ov::preprocess::ColorFormat sourceColorFormat;

public:
    static const char COLOR_FORMAT_DELIMITER;
    ColorFormatConfiguration() = default;
    ColorFormatConfiguration(ov::preprocess::ColorFormat target, ov::preprocess::ColorFormat source) :
        targetColorFormat(target),
        sourceColorFormat(source) {}
    ColorFormatConfiguration(const std::string& colorFormat) { fromString(colorFormat, *this); }
    static Status fromString(const std::string& configurationStr, ColorFormatConfiguration& configOut);
    static Status stringToColorFormat(const std::string& colorFormatStr, ov::preprocess::ColorFormat& colorFormatOut);

    const ov::preprocess::ColorFormat& getTargetColorFormat() const {
        return targetColorFormat;
    }
    const ov::preprocess::ColorFormat& getSourceColorFormat() const {
        return sourceColorFormat;
    }

    ColorFormatConfiguration& operator=(const ColorFormatConfiguration& other) {
        this->sourceColorFormat = other.sourceColorFormat;
        this->targetColorFormat = other.targetColorFormat;
        return *this;
    }
};
}  // namespace ovms
