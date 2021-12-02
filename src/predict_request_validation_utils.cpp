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

#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "modelconfig.hpp"

namespace ovms {
namespace request_validation_utils {

class RequestValidator {
    const tensorflow::serving::PredictRequest& request;
    const tensor_map_t& inputsInfo;
    const std::string& servableName;
    const model_version_t servableVersion;
    const std::set<const char*>& optionalAllowedInputNames;
    const Mode batchingMode;
    const shapes_map_t& shapeInfo;

    google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator it;

    RequestValidator() = delete;

    const std::string& getCurrentlyValidatedInputName() const {
        return it->first;
    }

public:
    RequestValidator(
        const tensorflow::serving::PredictRequest& request, const tensor_map_t& inputsInfo,
        const std::string& servableName, const model_version_t servableVersion, const std::set<const char*>& optionalAllowedInputNames,
        const Mode batchingMode, const shapes_map_t& shapeInfo) :
        request(request),
        inputsInfo(inputsInfo),
        servableName(servableName),
        servableVersion(servableVersion),
        optionalAllowedInputNames(optionalAllowedInputNames),
        batchingMode(batchingMode),
        shapeInfo(shapeInfo) {}

    Status validateNumberOfInputs(const tensorflow::serving::PredictRequest& request) const;
    Status validateAndGetInput(const tensorflow::serving::PredictRequest& request, const std::string& name, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it);
    Status checkIfShapeValuesNegative(const tensorflow::TensorProto& proto) const;
    Status validateNumberOfBinaryInputShapeDimensions(const tensorflow::TensorProto& proto) const;
    Status checkBatchSizeMismatch(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkBinaryBatchSizeMismatch(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkShapeMismatch(const tensorflow::TensorProto& proto, const ovms::TensorInfo& inputInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validateTensorContentSize(const tensorflow::TensorProto& proto, ovms::Precision expectedPrecision) const;
    Status validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const;
    Status validatePrecision(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const;
    Status validate();
};

Status RequestValidator::validateNumberOfInputs(const tensorflow::serving::PredictRequest& request) const {
    size_t expectedNumberOfInputs = inputsInfo.size();
    for (auto optionalAllowedInputName : optionalAllowedInputNames) {
        if (request.inputs().count(optionalAllowedInputName))
            expectedNumberOfInputs++;
    }
    if (request.inputs_size() > 0 && expectedNumberOfInputs == static_cast<size_t>(request.inputs_size())) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request.inputs_size();
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of inputs - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
}

Status RequestValidator::validateAndGetInput(const tensorflow::serving::PredictRequest& request, const std::string& name, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it) {
    it = request.inputs().find(name);
    if (it != request.inputs().end()) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}

Status RequestValidator::checkIfShapeValuesNegative(const tensorflow::TensorProto& proto) const {
    for (size_t i = 0; i < proto.tensor_shape().dim_size(); i++) {
        if (proto.tensor_shape().dim(i).size() <= 0) {
            std::stringstream ss;
            ss << "Negative or zero dimension size is not acceptable: " << TensorInfo::tensorShapeToString(proto.tensor_shape()) << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

Status RequestValidator::validateNumberOfBinaryInputShapeDimensions(const tensorflow::TensorProto& proto) const {
    if (proto.tensor_shape().dim_size() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << proto.tensor_shape().dim_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

Status RequestValidator::checkBatchSizeMismatch(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    if (networkBatchSize == 0)
        return StatusCode::OK;
    if (networkBatchSize > 0 && static_cast<size_t>(proto.tensor_shape().dim(0).size()) == networkBatchSize)
        return StatusCode::OK;

    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << networkBatchSize << "; Actual: " << proto.tensor_shape().dim(0).size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

Status RequestValidator::checkBinaryBatchSizeMismatch(const tensorflow::TensorProto& proto, const size_t networkBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    if (proto.string_val_size() <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
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
        ss << "Expected: " << networkBatchSize << "; Actual: " << proto.string_val_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

Status RequestValidator::checkShapeMismatch(const tensorflow::TensorProto& proto, const ovms::TensorInfo& inputInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = inputInfo.getShape();
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
        ss << "Expected: " << TensorInfo::shapeToString(inputInfo.getShape())
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.tensor_shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

Status RequestValidator::validateTensorContentSize(const tensorflow::TensorProto& proto, ovms::Precision expectedPrecision) const {
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
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.int_val_size() << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of values in tensor proto container - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (proto.dtype() == tensorflow::DataType::DT_HALF) {
        if (proto.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(proto.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.half_val_size() << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of values in tensor proto container - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
        if (expectedContentSize != proto.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << proto.tensor_content().size() << " bytes; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}

Status RequestValidator::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getShape();
    if (proto.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << TensorInfo::shapeToString(shape)
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.tensor_shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

Status RequestValidator::validatePrecision(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const {
    if (proto.dtype() != inputInfo.getPrecisionAsDataType()) {
        std::stringstream ss;
        ss << "Expected: " << inputInfo.getPrecisionAsString()
           << "; Actual: " << TensorInfo::getDataTypeAsString(proto.dtype())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

Mode getShapeMode(const shapes_map_t& shapeInfo, const std::string& name) {
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

Status RequestValidator::validate() {
    Status finalStatus = StatusCode::OK;

    auto status = validateNumberOfInputs(request);
    if (!status.ok())
        return status;

    for (const auto& [name, inputInfo] : inputsInfo) {
        status = validateAndGetInput(request, name, it);
        if (!status.ok())
            return status;

        const auto& proto = it->second;

        status = checkIfShapeValuesNegative(proto);
        if (!status.ok())
            return status;

        const size_t batchSize = inputInfo->getBatchSize();
        Mode shapeMode = getShapeMode(shapeInfo, name);

        // More detailed binary input validation is performed in next step, during conversion to blob.
        if (proto.dtype() == tensorflow::DataType::DT_STRING) {
            SPDLOG_DEBUG("[servable name: {} version: {}] Received request containing binary input: name: {}; batch size: {}", servableName, servableVersion, name, proto.string_val_size());

            status = validateNumberOfBinaryInputShapeDimensions(proto);
            if (!status.ok())
                return status;

            status = checkBinaryBatchSizeMismatch(proto, batchSize, finalStatus, batchingMode, shapeMode);
            if (!status.ok())
                return status;

            continue;
        }

        status = validatePrecision(*inputInfo, proto);
        if (!status.ok())
            return status;

        status = validateNumberOfShapeDimensions(*inputInfo, proto);
        if (!status.ok())
            return status;

        status = checkBatchSizeMismatch(proto, batchSize, finalStatus, batchingMode, shapeMode);
        if (!status.ok())
            return status;

        status = checkShapeMismatch(proto, *inputInfo, finalStatus, batchingMode, shapeMode);
        if (!status.ok())
            return status;

        status = validateTensorContentSize(proto, inputInfo->getPrecision_2());
        if (!status.ok())
            return status;
    }
    return finalStatus;
}

Status validate(const tensorflow::serving::PredictRequest& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<const char*>& optionalAllowedInputNames, const Mode batchingMode, const shapes_map_t& shapeInfo) {
    return RequestValidator(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

}  // namespace request_validation_utils
}  // namespace ovms
