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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../custom_node_interface.h"

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }

#define NODE_EXPECT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
    }

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, int defaultValue = 0) {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stoi(params[i].value);
            } catch (std::invalid_argument&) {
                return defaultValue;
            } catch (std::out_of_range&) {
                return defaultValue;
            }
        }
    }
    return defaultValue;
}

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, float defaultValue = 0.0f) {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stof(params[i].value);
            } catch (std::invalid_argument&) {
                return defaultValue;
            } catch (std::out_of_range&) {
                return defaultValue;
            }
        }
    }
    return defaultValue;
}

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, bool& isDefined, float defaultValue = 0.0f) {
    isDefined = true;
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stof(params[i].value);
            } catch (std::invalid_argument&) {
                isDefined = false;
                return defaultValue;
            } catch (std::out_of_range&) {
                isDefined = false;
                return defaultValue;
            }
        }
    }
    isDefined = false;
    return defaultValue;
}

std::string get_string_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, const std::string& defaultValue = "") {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            return params[i].value;
        }
    }
    return defaultValue;
}

std::vector<float> get_float_list_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount) {
    std::string listStr;
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            listStr = params[i].value;
            break;
        }
    }

    if (listStr.length() < 2 || listStr.front() != '[' || listStr.back() != ']') {
        return {};
    }

    listStr = listStr.substr(1, listStr.size() - 2);

    std::vector<float> result;

    std::stringstream lineStream(listStr);
    std::string element;
    while (std::getline(lineStream, element, ',')) {
        try {
            float e = std::stof(element.c_str());
            result.push_back(e);
        } catch (std::invalid_argument&) {
            NODE_EXPECT(false, "error parsing list parameter");
            return {};
        } catch (std::out_of_range&) {
            NODE_EXPECT(false, "error parsing list parameter");
            return {};
        }
    }

    return result;
}

std::string floatListToString(const std::vector<float>& values) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            ss << ",";
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

void cleanup(CustomNodeTensor& tensor) {
    free(tensor.data);
    free(tensor.dims);
}
