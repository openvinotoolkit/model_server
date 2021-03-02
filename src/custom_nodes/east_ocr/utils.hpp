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

#include <string>

#include "../../custom_node_interface.h"

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsLength, float defaultValue = 0.0f) {
    for (int i = 0; i < paramsLength; i++) {
        if (name == params[i].key) {
            return std::stof(params[i].value);
        }
    }
    return defaultValue;
}

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsLength, int defaultValue = 0) {
    for (int i = 0; i < paramsLength; i++) {
        if (name == params[i].key) {
            return std::stoi(params[i].value);
        }
    }
    return defaultValue;
}

std::string get_string_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsLength, const std::string& defaultValue = "") {
    for (int i = 0; i < paramsLength; i++) {
        if (name == params[i].key) {
            return params[i].value;
        }
    }
    return defaultValue;
}
