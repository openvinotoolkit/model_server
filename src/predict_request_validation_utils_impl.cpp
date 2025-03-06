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
#include "predict_request_validation_utils_impl.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "logging.hpp"
#include "tensorinfo.hpp"
#include "status.hpp"

namespace ovms {
namespace request_validation_utils {
Status validateAgainstMax2DStringArraySize(int32_t inputBatchSize, size_t inputWidth) {
    if (inputBatchSize <= 0) {
        return StatusCode::INVALID_BATCH_SIZE;
    }
    if (inputWidth > std::numeric_limits<size_t>::max() / inputBatchSize) {
        return StatusCode::INVALID_STRING_MAX_SIZE_EXCEEDED;
    }
    size_t expectedTensorSize = inputBatchSize * inputWidth;
    if (expectedTensorSize > MAX_2D_STRING_ARRAY_SIZE) {
        std::stringstream ss;
        ss << "; actual " << expectedTensorSize / (1024 * 1024) << "MB (max 1GB)";
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_STRING_MAX_SIZE_EXCEEDED, details);
    }
    return StatusCode::OK;
}
Mode getShapeMode(const shapes_info_map_t& shapeInfo, const std::string& name) {
    if (shapeInfo.size() == 0) {
        return Mode::FIXED;
    }
    auto it = shapeInfo.find(name);
    if (it == shapeInfo.end()) {
        it = shapeInfo.find(ANONYMOUS_INPUT_NAME);
    }
    if (it == shapeInfo.end()) {
        return Mode::FIXED;
    }
    return it->second.shapeMode;
}
}  // namespace request_validation_utils
}  // namespace ovms
