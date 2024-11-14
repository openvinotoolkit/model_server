//*****************************************************************************
// Copyright 2024 Intel Corporation
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
//#include "predict_request_validation_utils.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "logging.hpp"
//#include "modelconfig.hpp"
//#include "prediction_service_utils.hpp"
#include "shape.hpp"
#include "anonymous_input_name.hpp"
#include "status.hpp"

namespace ovms {
namespace request_validation_utils {
const size_t MAX_2D_STRING_ARRAY_SIZE = 1024 * 1024 * 1024 * 1;  // 1GB
Status getRawInputContentsBatchSizeAndWidth(const std::string& buffer, int32_t& batchSize, size_t& width); // TODO @atobisze make not static this comes from KFS - may need to move there
Status validateAgainstMax2DStringArraySize(int32_t inputBatchSize, size_t inputWidth);
Mode getShapeMode(const shapes_info_map_t& shapeInfo, const std::string& name);
}  // namespace request_validation_utils
}  // namespace ovms
