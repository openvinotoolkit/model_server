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
#include "predict_request_validation_utils.hpp"

#include "precision.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "capi_frontend/buffer.hpp"
#include "capi_frontend/capi_utils.hpp"
#include "capi_frontend/inferencerequest.hpp"
#include "capi_frontend/inferencetensor.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "kfs_frontend/kfs_utils.hpp"
#include "modelconfig.hpp"
//#include "prediction_service_utils.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tfs_frontend/tfs_utils.hpp"

// TODO @atobisze to remove whole file
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

template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, IteratorType, ShapeType>::checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& tensorInfo, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = tensorInfo.getShape();
    bool mismatch = false;
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    if (batchingMode == AUTO) {  // Skip batch dimension
        if (!batchSizeIndex.has_value()) {
            SPDLOG_ERROR("Batching AUTO enabled but batch size is missing");
            return StatusCode::INTERNAL_ERROR;
        }
        for (size_t i = 0; i < batchSizeIndex.value(); i++) {
            if (!shape[i].match(static_cast<dimension_value_t>(rsi.getDim(i)))) {
                mismatch = true;
                break;
            }
        }
        for (size_t i = batchSizeIndex.value() + 1; i < rsi.getShapeSize(); i++) {
            if (!shape[i].match(static_cast<dimension_value_t>(rsi.getDim(i)))) {
                mismatch = true;
                break;
            }
        }
    } else {  // Do not skip batch dimension
        for (size_t i = 0; i < rsi.getShapeSize(); i++) {
            if (!shape[i].match(static_cast<dimension_value_t>(rsi.getDim(i)))) {
                mismatch = true;
                break;
            }
        }
    }
    if (!mismatch) {
        return StatusCode::OK;
    }
    if (shapeMode == AUTO) {
        finalStatus = StatusCode::RESHAPE_REQUIRED;
        return StatusCode::OK;
    } else {
        std::stringstream ss;
        ss << "Expected: " << tensorInfo.getShape().toString()
           << "; Actual: " << tensorShapeToString(rsi.getShape())
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

/*static size_t getStringInputWidth(const KFSTensorInputProto& src) {
    size_t maxStringLength = 0;
    for (const auto& str : src.contents().bytes_contents()) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    return maxStringLength + 1;
}

static size_t getStringInputWidth(const tensorflow::TensorProto& src) {
    size_t maxStringLength = 0;
    for (const auto& str : src.string_val()) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    return maxStringLength + 1;
}

static size_t getStringInputWidth(const InferenceTensor& src) {
    return 0;
}

static int64_t getStringBatchSize(const KFSTensorInputProto& src) {
    return src.contents().bytes_contents_size();
}

static int64_t getStringBatchSize(const tensorflow::TensorProto& src) {
    return src.string_val_size();
}

static int64_t getStringBatchSize(const InferenceTensor& src) {
    return 0;
}*/

}  // namespace request_validation_utils
}  // namespace ovms
