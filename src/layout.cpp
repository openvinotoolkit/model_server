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
#include "layout.hpp"

#include <algorithm>

#include "stringutils.hpp"

namespace ovms {

LayoutConfiguration::LayoutConfiguration(const char* modelLayout) :
    model(modelLayout) {
}

LayoutConfiguration::LayoutConfiguration(const std::string& modelLayout) :
    model(modelLayout) {
}

LayoutConfiguration::LayoutConfiguration(const std::string& tensorLayout, const std::string& modelLayout) :
    tensor(tensorLayout), model(modelLayout) {
}

bool LayoutConfiguration::isSet() const {
    return !tensor.empty() || !model.empty();
}

// TODO: isConfigurationEqual layout comparision
// TODO: Reading from parameter.
// TODO: Unit tests
Status LayoutConfiguration::fromString(const std::string& configuration, LayoutConfiguration& configOut) {
    std::string configurationCopy = configuration;
    erase_spaces(configurationCopy);

    std::transform(configurationCopy.begin(), configurationCopy.end(), configurationCopy.begin(), ::toupper);

    if (configurationCopy.find_first_not_of("NCHWD?.:") != std::string::npos)
        return StatusCode::LAYOUT_WRONG_FORMAT;

    size_t delimCount = std::count(configurationCopy.begin(), configurationCopy.end(), ':');
    if (delimCount > 1)
        return StatusCode::LAYOUT_WRONG_FORMAT;
    
    if (delimCount == 0) {
        configOut = LayoutConfiguration("", configurationCopy);
    } else {
        std::vector<std::string> tokens = tokenize(configurationCopy, ':');
        if (tokens.size() > 2)
            return StatusCode::LAYOUT_WRONG_FORMAT;
        else if (tokens.size() == 2)
            configOut = LayoutConfiguration(tokens[0], tokens[1]);
        else if (tokens.size() == 1)
            configOut = LayoutConfiguration("", tokens[0]);
        else
            return StatusCode::LAYOUT_WRONG_FORMAT;
    }
    return StatusCode::OK;
}

const std::string& LayoutConfiguration::getTensorLayout() const {
    return this->tensor;
}

const std::string& LayoutConfiguration::getModelLayout() const {
    return this->model;
}

bool LayoutConfiguration::operator==(const LayoutConfiguration& rhs) const {
    return this->tensor == rhs.tensor && this->model == rhs.model;
}

bool LayoutConfiguration::operator!=(const LayoutConfiguration& rhs) const {
    return !(this->operator==(rhs));
}

// TODO: toString

}  // namespace ovms
