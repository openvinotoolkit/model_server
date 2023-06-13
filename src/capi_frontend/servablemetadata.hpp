//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "../modelversion.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
using capi_tensor_shapes_map_t = std::unordered_map<std::string, std::vector<dimension_value_t>>;

class ServableMetadata {
    static const ov::AnyMap EMPTY_RT_INFO;
    const std::string name;
    const model_version_t version;
    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;
    capi_tensor_shapes_map_t inDimMin;
    capi_tensor_shapes_map_t inDimMax;
    capi_tensor_shapes_map_t outDimMin;
    capi_tensor_shapes_map_t outDimMax;
    // for now this returns ov::Model::get_rt_info("model_info")
    ov::AnyMap info;

public:
    ServableMetadata(const std::string& name,
        model_version_t version,
        const tensor_map_t& inputsInfo,
        const tensor_map_t& outputsInfo,
        const ov::AnyMap& anyMap = EMPTY_RT_INFO);
    const std::string& getName() const { return name; }
    model_version_t getVersion() const { return version; }
    const tensor_map_t& getInputsInfo() const { return inputsInfo; }
    const tensor_map_t& getOutputsInfo() const { return outputsInfo; }
    const capi_tensor_shapes_map_t& getInputDimsMin() const { return this->inDimMin; }
    const capi_tensor_shapes_map_t& getInputDimsMax() const { return this->inDimMax; }
    const capi_tensor_shapes_map_t& getOutputDimsMin() const { return this->outDimMin; }
    const capi_tensor_shapes_map_t& getOutputDimsMax() const { return this->outDimMax; }
    const ov::AnyMap& getInfo() const {
        return this->info;
    }
};
}  // namespace ovms
