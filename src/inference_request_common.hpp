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

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "logging.hpp"
#include "shape.hpp"
#include "anonymous_input_name.hpp"
#include "status.hpp"

namespace ovms {
namespace request_validation_utils {
//const size_t MAX_2D_STRING_ARRAY_SIZE = 1024 * 1024 * 1024 * 1;  // 1GB
Status getRawInputContentsBatchSizeAndWidth(const std::string& buffer, int32_t& batchSize, size_t& width) {  // TODO @atobisze make not static this comes from KFS - may need to move there
    size_t offset = 0;
    size_t tmpBatchSize = 0;
    size_t tmpMaxStringLength = 0;
    while (offset + sizeof(uint32_t) <= buffer.size()) {
        size_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer.data() + offset));
        tmpMaxStringLength = std::max(tmpMaxStringLength, inputSize);
        offset += (sizeof(uint32_t) + inputSize);
        tmpBatchSize++;
    }
    if (offset > buffer.size()) {
        SPDLOG_DEBUG("Raw input contents invalid format. Every input need to be preceded by four bytes of its size. Buffer exceeded by {} bytes", offset - buffer.size());
        return StatusCode::INVALID_INPUT_FORMAT;
    } else if (offset < buffer.size()) {
        SPDLOG_DEBUG("Raw input contents invalid format. Every input need to be preceded by four bytes of its size. Unprocessed {} bytes", buffer.size() - offset);
        return StatusCode::INVALID_INPUT_FORMAT;
    }
    batchSize = tmpBatchSize;
    width = tmpMaxStringLength + 1;
    return StatusCode::OK;
}

Status validateAgainstMax2DStringArraySize(int32_t inputBatchSize, size_t inputWidth);
Mode getShapeMode(const shapes_info_map_t& shapeInfo, const std::string& name);
}  // namespace request_validation_utils
}  // namespace ovms
