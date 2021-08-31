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

#include <string>
#include <sstream>

#include <spdlog/spdlog.h>

namespace ovms {

Status validateNumberOfInputs_New(const tensorflow::serving::PredictRequest& request, const size_t expectedNumberOfInputs) {
    if (request.inputs_size() > 0 && expectedNumberOfInputs == static_cast<size_t>(request.inputs_size())) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request.inputs_size();
    const std::string details = ss.str();
    SPDLOG_DEBUG("[requested endpoint:{} version:{}] Invalid number of inputs - {}", request.model_spec().name(), request.model_spec().version().value(), details);
    return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
}

Status validateAndGetInput_New(const tensorflow::serving::PredictRequest& request, const std::string& name, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it) {
    it = request.inputs().find(name);
    if (it != request.inputs().end()) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[requested endpoint:{} version:{}] Missing input with specific name - {}", request.model_spec().name(), request.model_spec().version().value(), details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}

Status checkIfShapeValuesNegative_New(const tensorflow::TensorProto& proto) {
    for (size_t i = 0; i < proto.tensor_shape().dim_size(); i++) {
        if (proto.tensor_shape().dim(i).size() <= 0) {
            const std::string details = "Negative or zero dimension size is not acceptable: " + TensorInfo::tensorShapeToString(proto.tensor_shape());
            SPDLOG_DEBUG("Invalid shape - {}", details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

Status validateNumberOfBinaryInputShapeDimensions_New(const tensorflow::TensorProto& proto) {
    if (proto.tensor_shape().dim_size() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << proto.tensor_shape().dim_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid number of shape dimensions - {}", details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

Status checkBatchSizeMismatch_New(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) {
    if (networkBatchSize == 0)
        return StatusCode::OK;
    if (networkBatchSize > 0 && static_cast<size_t>(proto.tensor_shape().dim(0).size()) == networkBatchSize)
        return StatusCode::OK;

    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << networkBatchSize << "; Actual: " << proto.tensor_shape().dim(0).size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid batch size - {}", details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

Status checkBinaryBatchSizeMismatch_New(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) {
    if (proto.string_val_size() <= 0) {
        const std::string details = "Batch size must be positive";
        SPDLOG_DEBUG("Invalid batch size - {}", details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    if (networkBatchSize == 0)
        return StatusCode::OK;
    if (networkBatchSize > 0 && static_cast<size_t>(proto.string_val_size()) == networkBatchSize)
        return StatusCode::OK;

    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << networkBatchSize << "; Actual: " << proto.string_val_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid batch size - {}", details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

Status checkShapeMismatch_New(const tensorflow::TensorProto& proto, const ovms::TensorInfo& inputInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode) {
    const auto& shape = inputInfo.getEffectiveShape();
    int i = (batchingMode == AUTO) ? 1 : 0;  // If batch size is automatic, omit first dimension
    bool mismatch = false;
    for (; i < proto.tensor_shape().dim_size(); i++) {
        if (shape[i] > 0 && proto.tensor_shape().dim(i).size() != shape[i]) {
            mismatch = true;
            break;
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
        ss << "Expected: " << TensorInfo::shapeToString(inputInfo.getEffectiveShape())
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.tensor_shape());
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid shape - {}", details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

Status validateTensorContentSize_New(const tensorflow::TensorProto& proto, InferenceEngine::Precision expectedPrecision) {
    /*
    int8        data in request.tensor_content
    uint8       data in request.tensor_content
    int16       data in request.tensor_content
    uint16      request.tensor_content is empty, data located in request.int_val
    int32       data in request.tensor_content
    uint32      data in request.tensor_content
    int64       data in request.tensor_content
    uint64      data in request.tensor_content
    float16     request.tensor_content is empty, data located in request.half_val
    float32     data in request.tensor_content
    double      data in request.tensor_content

    _TENSOR_CONTENT_TYPES
    https://github.com/tensorflow/tensorflow/blob/903a6399aab19b549fefd0ead836af644f3d00f8/tensorflow/python/framework/tensor_util.py#L237
*/

    size_t expectedValueCount = 1;
    for (int i = 0; i < proto.tensor_shape().dim_size(); i++) {
        expectedValueCount *= proto.tensor_shape().dim(i).size();
    }

    // Network expects tensor content size or value count
    if (proto.dtype() == tensorflow::DataType::DT_UINT16) {
        if (proto.int_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(proto.int_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.int_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid number of values in tensor proto container - {}", details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (proto.dtype() == tensorflow::DataType::DT_HALF) {
        if (proto.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(proto.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.half_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid number of values in tensor proto container - {}", details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * expectedPrecision.size();
        if (expectedContentSize != proto.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << proto.tensor_content().size() << " bytes";
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid content size of tensor proto - {}", details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}

Status validateNumberOfShapeDimensions_New(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getEffectiveShape();
    if (proto.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << TensorInfo::shapeToString(shape)
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.tensor_shape());
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid number of shape dimensions - {}", details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

Status validatePrecision_New(const tensorflow::TensorProto& proto, const ovms::TensorInfo& inputInfo) {
    if (proto.dtype() != inputInfo.getPrecisionAsDataType()) {
        std::stringstream ss;
        ss << "Expected: " << inputInfo.getPrecisionAsString()
           << "; Actual: " << TensorInfo::getDataTypeAsString(proto.dtype());
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid precision - {}", details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

}  // namespace ovms
