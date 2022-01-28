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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <set>
#include <string>

#include <google/protobuf/map.h>

#include "modelversion.hpp"
#include "shape.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {
namespace request_validation_utils {

Status validate(
    const tensorflow::serving::PredictRequest& request,
    const tensor_map_t& inputsInfo,
    const std::string& servableName,
    const model_version_t servableVersion,
    const std::set<std::string>& optionalAllowedInputNames = {},
    const Mode batchingMode = Mode::FIXED,
    const shapes_info_map_t& shapeInfo = shapes_info_map_t());

}  // namespace request_validation_utils
}  // namespace ovms
