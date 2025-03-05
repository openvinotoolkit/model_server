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
#include "validation.hpp"

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

#include "tfs_utils.hpp"
#include "../precision.hpp"
#include "../logging.hpp"
//#include "../prediction_service_utils.hpp"
#include "../predict_request_validation_utils.hpp"
#include "../profiler.hpp"
#include "../tensorinfo.hpp"
#include "../status.hpp"

namespace ovms {
namespace request_validation_utils {
using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, TFSInputTensorType>::const_iterator;
using TFSShapeType = tensorflow::TensorShapeProto;

template <>
dimension_value_t RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getDim(size_t i) {
    return tensor.tensor_shape().dim(i).size();
}
template <>
size_t RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getShapeSize() {
    return tensor.tensor_shape().dim_size();
}
template <>
const TFSShapeType& RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getShape() {
    return tensor.tensor_shape();
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
const std::string RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::getCurrentlyValidatedTensorName() const {
    return *currentlyValidatedName;
}
template <>
const TFSInputTensorType& RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::getTensorFromIt(const TFSInputTensorIteratorType& it) const {
    return it->second;
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfTensors() const {
    size_t expectedNumberOfInputs = inputsInfo.size();
    for (auto& optionalAllowedInputName : optionalAllowedInputNames) {
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
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfBinaryInputShapeDimensions(const TFSInputTensorType& proto) const {
    RequestShapeInfo<TFSInputTensorType, TFSShapeType> rsi(proto);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::checkBinaryBatchSizeMismatch(const TFSInputTensorType& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<TFSInputTensorType, TFSShapeType> rsi(proto);
    if (inputBatchSize < 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    if (rsi.getDim(0) != inputBatchSize) {
        SPDLOG_DEBUG("[servable name: {} version: {}] Batch size in request {} does not match actual {}", servableName, servableVersion, rsi.getDim(0), inputBatchSize);
        return StatusCode::INVALID_BATCH_SIZE;
    }
    if (servableBatchSize.value().match(rsi.getDim(0))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << proto.string_val_size() << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <>
size_t getStringInputWidth(const tensorflow::TensorProto& src) {
    size_t maxStringLength = 0;
    for (const auto& str : src.string_val()) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    return maxStringLength + 1;
}
template <>
int64_t getStringBatchSize(const tensorflow::TensorProto& src) {
    return src.string_val_size();
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validateTensorContent(const TFSInputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
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

    // For POD types
    size_t expectedValueCount = 1;
    for (int i = 0; i < proto.tensor_shape().dim_size(); i++) {
        expectedValueCount *= proto.tensor_shape().dim(i).size();
    }

    // Network expects tensor content size or value count
    if (proto.dtype() == tensorflow::DataType::DT_STRING) {
        if (proto.string_val_size() < 0 || proto.string_val_size() != proto.tensor_shape().dim(0).size()) {
            std::stringstream ss;
            ss << "Expected: " << proto.tensor_shape().dim(0).size() << "; Actual: " << proto.string_val_size() << "; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of values in tensor proto string container - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (proto.dtype() == tensorflow::DataType::DT_UINT16) {
        if (proto.int_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(proto.int_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.int_val_size() << "; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of values in tensor proto container - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (proto.dtype() == tensorflow::DataType::DT_HALF) {
        if (proto.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(proto.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << proto.half_val_size() << "; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of values in tensor proto container - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
        if (expectedContentSize != proto.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << proto.tensor_content().size() << " bytes; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& tensorInfo, const TFSInputTensorType& proto) const {
    // Network and request must have the same number of shape dimensions
    const auto& shape = tensorInfo.getShape();
    if (proto.tensor_shape().dim_size() < 0 ||
        shape.size() != static_cast<size_t>(proto.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(proto.tensor_shape())
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>::validatePrecision(const ovms::TensorInfo& tensorInfo, const TFSInputTensorType& proto) const {
    if (proto.dtype() != getPrecisionAsDataType(tensorInfo.getPrecision())) {
        std::stringstream ss;
        ss << "Expected: " << tensorInfo.getPrecisionAsString()
           << "; Actual: " << getDataTypeAsString(proto.dtype())
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}
template <>
bool dataInRawInputContents(const TFSRequestType& request) {
    return false;
}

template <>
const std::string* getRawInputContents(const TFSRequestType& request, size_t bufferId) {
    SPDLOG_DEBUG("Raw input contents is not supported for TFS API");
    throw std::runtime_error("Raw input contents used in TFSflow.");
    return nullptr;
}

template <>
Status validate(const TFSRequestType& request, const tensor_map_t& inputsInfo, const tensor_map_t& outputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<TFSRequestType, TFSInputTensorType, ValidationChoice::INPUT, TFSInputTensorIteratorType, TFSShapeType>(request, inputsInfo, outputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}
}  // namespace request_validation_utils
}  // namespace ovms
