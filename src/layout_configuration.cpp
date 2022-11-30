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
#include "layout_configuration.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

const char LayoutConfiguration::LAYOUT_CONFIGURATION_DELIMETER = ':';

LayoutConfiguration::LayoutConfiguration(const char* layout) :
    LayoutConfiguration(std::string(layout)) {
}

LayoutConfiguration::LayoutConfiguration(const std::string& layout) :
    LayoutConfiguration(layout, layout) {
}

LayoutConfiguration::LayoutConfiguration(const std::string& tensorLayout, const std::string& modelLayout) :
    tensor(Layout(tensorLayout)),
    model(Layout(modelLayout)) {
}

bool LayoutConfiguration::isSet() const {
    return !tensor.empty() || !model.empty();
}

Status LayoutConfiguration::fromString(const std::string& configurationStr, LayoutConfiguration& configOut) {
    std::string configurationCopy = configurationStr;
    erase_spaces(configurationCopy);

    std::transform(configurationCopy.begin(), configurationCopy.end(), configurationCopy.begin(), ::toupper);

    if (configurationCopy.find_first_not_of(Layout::ALLOWED_DIMENSION_LETTERS_AND_CHARS + LAYOUT_CONFIGURATION_DELIMETER) != std::string::npos)
        return StatusCode::LAYOUT_WRONG_FORMAT;

    size_t delimCount = std::count(configurationCopy.begin(), configurationCopy.end(), LAYOUT_CONFIGURATION_DELIMETER);
    if (delimCount > 1)
        return StatusCode::LAYOUT_WRONG_FORMAT;

    if (delimCount == 0) {
        configOut = LayoutConfiguration(configurationCopy);
    } else {
        std::vector<std::string> tokens = tokenize(configurationCopy, LAYOUT_CONFIGURATION_DELIMETER);
        if (tokens.size() > 2)
            return StatusCode::LAYOUT_WRONG_FORMAT;
        else if (tokens.size() == 2)
            configOut = LayoutConfiguration(tokens[0], tokens[1]);
        else if (tokens.size() == 1)
            configOut = LayoutConfiguration(tokens[0]);
        else
            return StatusCode::LAYOUT_WRONG_FORMAT;
    }
    return StatusCode::OK;
}

std::string LayoutConfiguration::toString() const {
    std::stringstream ss;
    if (tensor.empty()) {
        ss << model;
    } else {
        ss << tensor << LAYOUT_CONFIGURATION_DELIMETER << model;
    }
    return ss.str();
}

const Layout& LayoutConfiguration::getTensorLayout() const {
    return this->tensor;
}

const Layout& LayoutConfiguration::getModelLayout() const {
    return this->model;
}

bool LayoutConfiguration::operator==(const LayoutConfiguration& rhs) const {
    return this->tensor == rhs.tensor && this->model == rhs.model;
}

bool LayoutConfiguration::operator!=(const LayoutConfiguration& rhs) const {
    return !(this->operator==(rhs));
}

}  // namespace ovms
