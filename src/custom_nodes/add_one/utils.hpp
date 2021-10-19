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

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, int defaultValue = 0) {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stoi(params[i].value);
            } catch (std::invalid_argument& e) {
                return defaultValue;
            } catch (std::out_of_range& e) {
                return defaultValue;
            }
        }
    }
    return defaultValue;
}