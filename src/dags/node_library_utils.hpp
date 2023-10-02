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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "../custom_node_interface.h"  // NOLINT
#include "../precision.hpp"
#include "../shape.hpp"
#include "node_library.hpp"
#include "tensormap.hpp"

namespace ovms {

class Status;
class TensorInfo;

CustomNodeTensorPrecision toCustomNodeTensorPrecision(ov::element::Type_t precision);
Precision toInferenceEnginePrecision(CustomNodeTensorPrecision precision);
std::unique_ptr<struct CustomNodeParam[]> createCustomNodeParamArray(const std::unordered_map<std::string, std::string>& paramMap);
std::unique_ptr<struct CustomNodeTensor[]> createCustomNodeTensorArray(const TensorMap& tensorMap, const std::unordered_map<std::string, shape_t>& tensorDims);
Status createTensorInfoMap(struct CustomNodeTensorInfo* info, int infoCount, std::map<std::string, std::shared_ptr<const TensorInfo>>& out, release_fn freeCallback, void* customNodeLibraryInternalManager);

}  // namespace ovms
