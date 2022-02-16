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

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, int defaultValue = 0);

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, float defaultValue = 0.0f);

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, bool& isDefined, float defaultValue = 0.0f);

std::string get_string_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, const std::string& defaultValue = "");

std::vector<float> get_float_list_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount);

std::string floatListToString(const std::vector<float>& values);

void cleanup(CustomNodeTensor& tensor);
