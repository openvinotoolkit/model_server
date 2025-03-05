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
#include "capi_validation.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "buffer.hpp"
#include "capi_utils.hpp"
#include "inferencerequest.hpp"
#include "inferencetensor.hpp"
#include "capi_request_utils.hpp"
#include "../logging.hpp"
#include "../precision.hpp"
//#include "../modelconfig.hpp"
//#include "prediction_service_utils.hpp"
#include "../profiler.hpp"
#include "../status.hpp"

namespace ovms {
namespace request_validation_utils {
Status validateCapiTensorPrecision(const ovms::TensorInfo& info, const InferenceTensor& tensor, const std::string& tensorName, const std::string& servableName, const model_version_t servableVersion, ValidationChoice choice) {
    if (tensor.getDataType() != getPrecisionAsOVMSDataType(info.getPrecision())) {
        std::stringstream ss;
        ss << "Expected: " << info.getPrecisionAsString()
           << "; Actual: " << toString(getOVMSDataTypeAsPrecision(tensor.getDataType())) << ";";
        if (choice == ValidationChoice::INPUT) {
            ss << " input name: ";
        }
        if (choice == ValidationChoice::OUTPUT) {
            ss << " output name: ";
        }
        ss << tensorName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

Status validateCapiTensorContent(const InferenceTensor& tensor, ovms::Precision expectedPrecision, size_t bufferId, const std::string& tensorName, const std::string& servableName, const model_version_t servableVersion, ValidationChoice choice) {
    const Buffer* buffer = tensor.getBuffer();
    if (nullptr == buffer) {
        std::stringstream ss;
        ss << "Servable: " << servableName
           << "; version: " << servableVersion
           << "; is missing buffer for tensor: " << tensorName;
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::NONEXISTENT_BUFFER, details);
    }
    size_t elementSize = (expectedPrecision == Precision::STRING) ? sizeof(std::string) : ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
    size_t expectedContentSize;
    if (computeExpectedBufferSizeReturnFalseIfOverflow<ovms::dimension_value_t>(tensor.getShape(), elementSize, expectedContentSize) == false) {
        SPDLOG_DEBUG("[servable name: {} version: {}] Expected content size overflow for tensor - {}", servableName, servableVersion, tensorName);
        return StatusCode::INVALID_SHAPE;
    }
    if (expectedContentSize != buffer->getByteSize()) {
        std::stringstream ss;
        ss << "Expected: " << expectedContentSize << " bytes; Actual: " << buffer->getByteSize() << " bytes;";
        if (choice == ValidationChoice::INPUT) {
            ss << " input name: ";
        }
        if (choice == ValidationChoice::OUTPUT) {
            ss << " output name: ";
        }
        ss << tensorName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }
    return StatusCode::OK;
}

Status validateCapiNumberOfShapeDimensions(const InferenceTensor& tensor, const ovms::TensorInfo& tensorInfo, const std::string& tensorName, const std::string& servableName, const model_version_t servableVersion, ValidationChoice choice) {
    // Network and request must have the same number of shape dimensions
    const auto& shape = tensorInfo.getShape();
    if (shape.size() != static_cast<size_t>(tensor.getShape().size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(tensor.getShape()) << ";";
        if (choice == ValidationChoice::INPUT) {
            ss << " input name: ";
        }
        if (choice == ValidationChoice::OUTPUT) {
            ss << " output name: ";
        }
        ss << tensorName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
dimension_value_t RequestShapeInfo<InferenceTensor, signed_shape_t>::getDim(size_t i) {
    return tensor.getShape()[i];
}
template <>
size_t RequestShapeInfo<InferenceTensor, signed_shape_t>::getShapeSize() {
    return tensor.getShape().size();
}
template <>
const signed_shape_t& RequestShapeInfo<InferenceTensor, signed_shape_t>::getShape() {
    return tensor.getShape();
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
const std::string RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::getCurrentlyValidatedTensorName() const {
    return *currentlyValidatedName;
}

template <>
const std::string RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::getCurrentlyValidatedTensorName() const {
    return *currentlyValidatedName;
}

template <>
const InferenceTensor& RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::getTensorFromIt(const InferenceTensor* const& it) const {
    return *it;
}
template <>
const InferenceTensor& RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::getTensorFromIt(const InferenceTensor* const& it) const {
    return *it;
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validateNumberOfTensors() const {
    size_t expectedNumberOfInputs = inputsInfo.size();
    if (request.getInputsSize() > 0 && expectedNumberOfInputs == static_cast<size_t>(request.getInputsSize())) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request.getInputsSize();
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of inputs - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::validateNumberOfTensors() const {
    return Status(StatusCode::OK);
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validateNumberOfBinaryInputShapeDimensions(const InferenceTensor& tensor) const {
    RequestShapeInfo<InferenceTensor, signed_shape_t> rsi(tensor);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::checkBinaryBatchSizeMismatch(const InferenceTensor& tensor, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<InferenceTensor, signed_shape_t> rsi(tensor);
    if (rsi.getDim(0) < 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    if (servableBatchSize.value().match(rsi.getDim(0))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << tensor.getBuffer()->getByteSize() << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

template <>
size_t getStringInputWidth(const InferenceTensor& src) {
    return 0;
}

template <>
int64_t getStringBatchSize(const InferenceTensor& src) {
    return 0;
}

template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, IteratorType, ShapeType>::validateInferenceTensorBufferType(const InputTensorType& it) const {
    const Buffer* buffer = it.getBuffer();
    const OVMS_BufferType bufType = buffer->getBufferType();
    if (bufType < OVMS_BUFFERTYPE_CPU || bufType > OVMS_BUFFERTYPE_HDDL) {
        std::stringstream ss;
        if (choice == ValidationChoice::INPUT) {
            ss << "Required input ";
        }
        if (choice == ValidationChoice::OUTPUT) {
            ss << "Required output ";
        }
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid buffer type for tensor with specific name - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BUFFER_TYPE, details);

    } else {
        // Remove this when other buffer types are supported
        if (bufType != OVMS_BUFFERTYPE_CPU &&
            bufType != OVMS_BUFFERTYPE_OPENCL &&
            bufType != OVMS_BUFFERTYPE_VASURFACE_Y &&
            bufType != OVMS_BUFFERTYPE_VASURFACE_UV) {
            std::stringstream ss;
            ss << "Required input ";
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid buffer type for input with specific name - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_BUFFER_TYPE, details);
        }
    }

    if (buffer->getBufferType() == OVMS_BUFFERTYPE_CPU && buffer->getDeviceId() != std::nullopt && buffer->getDeviceId() != 0) {
        std::stringstream ss;
        ss << "Required input " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid device id for buffer, input with specific name - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_DEVICE_ID, details);
    }

    return StatusCode::OK;
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validateTensorContent(const InferenceTensor& tensor, ovms::Precision expectedPrecision, size_t bufferId) const {
    auto status = validateCapiTensorContent(tensor, expectedPrecision, bufferId, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::INPUT);
    if (!status.ok())
        return status;
    return validateInferenceTensorBufferType(tensor);
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::validateTensorContent(const InferenceTensor& tensor, ovms::Precision expectedPrecision, size_t bufferId) const {
    auto status = validateCapiTensorContent(tensor, expectedPrecision, bufferId, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::OUTPUT);
    if (!status.ok())
        return status;
    return validateInferenceTensorBufferType(tensor);
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validateNumberOfShapeDimensions(const ovms::TensorInfo& tensorInfo, const InferenceTensor& tensor) const {
    return validateCapiNumberOfShapeDimensions(tensor, tensorInfo, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::INPUT);
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::validateNumberOfShapeDimensions(const ovms::TensorInfo& tensorInfo, const InferenceTensor& tensor) const {
    return validateCapiNumberOfShapeDimensions(tensor, tensorInfo, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::OUTPUT);
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>::validatePrecision(const ovms::TensorInfo& tensorInfo, const InferenceTensor& tensor) const {
    return validateCapiTensorPrecision(tensorInfo, tensor, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::INPUT);
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>::validatePrecision(const ovms::TensorInfo& tensorInfo, const InferenceTensor& tensor) const {
    return validateCapiTensorPrecision(tensorInfo, tensor, getCurrentlyValidatedTensorName(), servableName, servableVersion, ValidationChoice::OUTPUT);
}

template <>
bool dataInRawInputContents(const ovms::InferenceRequest& request) {
    return false;
}

template <>
const std::string* getRawInputContents(const ovms::InferenceRequest& request, size_t bufferId) {
    SPDLOG_DEBUG("Raw input contents is not supported for C-API");
    throw std::runtime_error("Raw input contents used in C-API flow.");
    return nullptr;
}

#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }

template <>
Status validate(const InferenceRequest& request, const tensor_map_t& inputsInfo, const tensor_map_t& outputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    auto inputValidationStatus = RequestValidator<InferenceRequest, InferenceTensor, ValidationChoice::INPUT, const InferenceTensor*, signed_shape_t>(request, inputsInfo, outputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
    if (!inputValidationStatus.ok())
        return inputValidationStatus;
    return RequestValidator<InferenceRequest, InferenceTensor, ValidationChoice::OUTPUT, const InferenceTensor*, signed_shape_t>(request, inputsInfo, outputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}
}  // namespace request_validation_utils
}  // namespace ovms
