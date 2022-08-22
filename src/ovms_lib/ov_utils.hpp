//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <map>
#include <memory>
#include <string>
#include <utility>

#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "modelconfig.hpp"
#include "status.hpp"

namespace ovms {

class TensorInfo;

Status createSharedTensor(ov::Tensor& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape);
/**
 *  Creates new tensor that copies data and owns the copy
 **/
ov::Tensor createSharedTensor(ov::element::Type_t precision, const shape_t& shape, void* data);

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<TensorInfo>>& tensorMap);

Status tensorClone(ov::Tensor& destinationTensor, const ov::Tensor& sourceTensor);

std::optional<ov::Layout> getLayoutFromRTMap(const ov::RTMap& rtMap);

Status validatePluginConfiguration(const plugin_config_t& pluginConfig, const std::string& targetDevice, const ov::Core& ieCore);

}  // namespace ovms
