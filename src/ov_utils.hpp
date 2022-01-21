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

#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "modelconfig.hpp"
#include "status.hpp"

namespace ovms {

class TensorInfo;

Status createSharedTensor(std::shared_ptr<ov::runtime::Tensor>& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape);
/**
 *  Creates new tensor that copies data and owns the copy
 **/
std::shared_ptr<ov::runtime::Tensor> createSharedTensor(ov::element::Type_t precision, const shape_t& shape, void* data);

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<TensorInfo>>& tensorMap);

Status tensorClone(std::shared_ptr<ov::runtime::Tensor>& destinationTensor, const ov::runtime::Tensor& sourceTensor);
}  // namespace ovms
