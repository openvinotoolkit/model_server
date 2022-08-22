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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "kfs_grpc_inference_service.hpp"
#include "modelconfig.hpp"
#include "profiler.hpp"
#include "tfs_frontend/tfs_utils.hpp"

namespace ovms {
namespace request_validation_utils {
template <typename RequestTensorType, typename RequestTensorShapeType>
struct RequestShapeInfo {
    const RequestTensorType& tensor;
    RequestShapeInfo(const RequestTensorType& tensor) :
        tensor(tensor) {}
    dimension_value_t getDim(size_t i);
    size_t getShapeSize();
    const RequestTensorShapeType& getShape();
};

using KFSRequestType = ::inference::ModelInferRequest;
using KFSInputTensorType = ::inference::ModelInferRequest_InferInputTensor;
using KFSInputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferRequest_InferInputTensor>;
using KFSShapeType = google::protobuf::RepeatedField<int64_t>;
using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, TFSInputTensorType>::const_iterator;
using TFSShapeType = tensorflow::TensorShapeProto;

template <>
dimension_value_t RequestShapeInfo<KFSInputTensorType, KFSShapeType>::getDim(size_t i) {
    return tensor.shape()[i];
}
template <>
dimension_value_t RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getDim(size_t i) {
    return tensor.tensor_shape().dim(i).size();
}
template <>
size_t RequestShapeInfo<KFSInputTensorType, KFSShapeType>::getShapeSize() {
    return tensor.shape().size();
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
const KFSShapeType& RequestShapeInfo<KFSInputTensorType, KFSShapeType>::getShape() {
    return tensor.shape();
}

template <typename RequestType, typename InputTensorType, typename InputIterator, typename ShapeType>
class RequestValidator {
    const RequestType& request;
    const tensor_map_t& inputsInfo;
    const std::string& servableName;
    const model_version_t servableVersion;
    const std::set<std::string>& optionalAllowedInputNames;
    const Mode batchingMode;
    const shapes_info_map_t& shapeInfo;

    InputIterator it;

    RequestValidator() = delete;

    const std::string& getCurrentlyValidatedInputName() const;
    const InputTensorType& getInputFromIt(const InputIterator& it) const;

public:
    RequestValidator(
        const RequestType& request, const tensor_map_t& inputsInfo,
        const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames,
        const Mode batchingMode, const shapes_info_map_t& shapeInfo) :
        request(request),
        inputsInfo(inputsInfo),
        servableName(servableName),
        servableVersion(servableVersion),
        optionalAllowedInputNames(optionalAllowedInputNames),
        batchingMode(batchingMode),
        shapeInfo(shapeInfo) {}

    Status validateNumberOfInputs() const;
    Status validateAndGetInput(const RequestType& request, const std::string& name, InputIterator& it, size_t& bufferId);
    Status checkIfShapeValuesNegative(const InputTensorType& proto) const;
    Status validateNumberOfBinaryInputShapeDimensions(const InputTensorType& proto) const;
    Status checkBatchSizeMismatch(const InputTensorType& proto, const Dimension& servableBatchSize, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkBinaryBatchSizeMismatch(const InputTensorType& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validateTensorContentSize(const InputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const;
    Status validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const InputTensorType& proto) const;
    Status validatePrecision(const ovms::TensorInfo& inputInfo, const InputTensorType& proto) const;
    bool checkIfBinaryInputUsed(const InputTensorType& proto, const std::string inputName) const;
    Status validateRequestCoherency() const;
    Status validate();
};

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateRequestCoherency() const {
    if (!request.raw_input_contents().empty()) {
        for (auto& input : request.inputs()) {
            if (input.has_contents()) {
                std::stringstream ss;
                ss << "Passing buffers both in InferInputTensor contents and in raw_input_contents is not allowed. Detected buffer in InferInputTensor contents for input: " << input.name();
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid request message - {}", servableName, servableVersion, details);
                return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
            }
        }
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfInputs() const {
    size_t expectedNumberOfInputs = inputsInfo.size();

    if (optionalAllowedInputNames.size() > 0) {
        auto it = request.inputs().begin();
        while (it != request.inputs().end()) {
            if (optionalAllowedInputNames.find(it->name()) != optionalAllowedInputNames.end()) {
                ++expectedNumberOfInputs;
            }
            ++it;
        }
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
const std::string& RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::getCurrentlyValidatedInputName() const {
    return it->name();
}

template <>
const std::string& RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::getCurrentlyValidatedInputName() const {
    return it->first;
}
template <>
const KFSInputTensorType& RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::getInputFromIt(const KFSInputTensorIteratorType& it) const {
    return *it;
}

template <>
const TFSInputTensorType& RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::getInputFromIt(const TFSInputTensorIteratorType& it) const {
    return it->second;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfInputs() const {
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

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateAndGetInput(const TFSRequestType& request, const std::string& name, TFSInputTensorIteratorType& it, size_t& bufferId) {
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

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateAndGetInput(const KFSRequestType& request, const std::string& name, KFSInputTensorIteratorType& it, size_t& bufferId) {
    it = request.inputs().begin();
    bufferId = 0;
    while (it != request.inputs().end()) {
        if (it->name() == name) {
            break;
        }
        ++it;
        ++bufferId;
    }
    if (it != request.inputs().end()) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}

template <>
template <typename RequestType, typename InputTensorType, typename InputTensorIteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, InputTensorIteratorType, ShapeType>::checkIfShapeValuesNegative(const InputTensorType& proto) const {
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    for (size_t i = 0; i < rsi.getShapeSize(); i++) {
        if (rsi.getDim(i) <= 0) {
            std::stringstream ss;
            ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(rsi.getShape()) << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfBinaryInputShapeDimensions(const TFSInputTensorType& proto) const {
    RequestShapeInfo<TFSInputTensorType, TFSShapeType> rsi(proto);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfBinaryInputShapeDimensions(const KFSInputTensorType& proto) const {
    RequestShapeInfo<KFSInputTensorType, KFSShapeType> rsi(proto);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, typename InputIteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, InputIteratorType, ShapeType>::checkBatchSizeMismatch(const InputTensorType& proto, const Dimension& servableBatchSize, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    if (servableBatchSize.match(rsi.getDim(batchSizeIndex))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.toString() << "; Actual: " << rsi.getDim(batchSizeIndex) << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::checkBinaryBatchSizeMismatch(const TFSInputTensorType& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<TFSInputTensorType, TFSShapeType> rsi(proto);
    if (proto.string_val_size() <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    if (servableBatchSize.match(rsi.getDim(0))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.toString() << "; Actual: " << proto.string_val_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::checkBinaryBatchSizeMismatch(const KFSInputTensorType& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<KFSInputTensorType, KFSShapeType> rsi(proto);
    if (proto.contents().bytes_contents_size() <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    if (servableBatchSize.match(rsi.getDim(0))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.toString() << "; Actual: " << proto.contents().bytes_contents_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = inputInfo.getShape();
    bool mismatch = false;
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    if (batchingMode == AUTO) {  // Skip batch dimension
        for (int i = 0; i < batchSizeIndex; i++) {
            if (!shape[i].match(static_cast<dimension_value_t>(rsi.getDim(i)))) {
                mismatch = true;
                break;
            }
        }
        for (int i = batchSizeIndex + 1; i < rsi.getShapeSize(); i++) {
            if (!shape[i].match(static_cast<dimension_value_t>(rsi.getDim(i)))) {
                mismatch = true;
                break;
            }
        }
    } else {  // Do not skip batch dimension
        for (int i = 0; i < rsi.getShapeSize(); i++) {
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
        ss << "Expected: " << inputInfo.getShape().toString()
           << "; Actual: " << tensorShapeToString(rsi.getShape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateTensorContentSize(const TFSInputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
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

size_t getElementsCount(const KFSInputTensorType& proto, ovms::Precision expectedPrecision) {
    switch (expectedPrecision) {
    case ovms::Precision::BOOL: {
        return proto.contents().bool_contents().size();
    }
        /// int_contents
    case ovms::Precision::I8:
    case ovms::Precision::I16:
    case ovms::Precision::I32: {
        return proto.contents().int_contents().size();
    }
        /// int64_contents
    case ovms::Precision::I64: {
        return proto.contents().int64_contents().size();
    }
        // uint_contents
    case ovms::Precision::U8:
    case ovms::Precision::U16:
    case ovms::Precision::U32: {
        return proto.contents().uint_contents().size();
    }
        // uint64_contents
    case ovms::Precision::U64: {
        return proto.contents().uint64_contents().size();
    }
        // fp32_contents
    case ovms::Precision::FP32: {
        return proto.contents().fp32_contents().size();
    }
        // fp64_contentes
    case ovms::Precision::FP64: {
        return proto.contents().fp64_contents().size();
    }
    case ovms::Precision::FP16:
    case ovms::Precision::U1:
    case ovms::Precision::CUSTOM:
    case ovms::Precision::UNDEFINED:
    case ovms::Precision::DYNAMIC:
    case ovms::Precision::MIXED:
    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
    default:
        return 0;
    }
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateTensorContentSize(const KFSInputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
    size_t expectedValueCount = 1;
    for (int i = 0; i < proto.shape().size(); i++) {
        expectedValueCount *= proto.shape()[i];
    }
    if (request.raw_input_contents().size()) {
        size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
        if (expectedContentSize != request.raw_input_contents()[bufferId].size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << request.raw_input_contents()[bufferId].size() << " bytes; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    } else {  // buffers placed in InputTensor content
        // here we should check that the elements count is equal since for some precisions there is padding
        // we need to decide first which exact datatype_contents we extract that information from
        size_t elementsCount = getElementsCount(proto, expectedPrecision);
        if (expectedValueCount != elementsCount) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << " values; Actual: " << elementsCount << " values; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid value count of tensor proto - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const TFSInputTensorType& proto) const {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getShape();
    if (proto.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(proto.tensor_shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const KFSInputTensorType& proto) const {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getShape();
    if (proto.shape().size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.shape().size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(proto.shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validatePrecision(const ovms::TensorInfo& inputInfo, const TFSInputTensorType& proto) const {
    if (proto.dtype() != getPrecisionAsDataType(inputInfo.getPrecision())) {
        std::stringstream ss;
        ss << "Expected: " << inputInfo.getPrecisionAsString()
           << "; Actual: " << getDataTypeAsString(proto.dtype())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::validatePrecision(const ovms::TensorInfo& inputInfo, const KFSInputTensorType& proto) const {
    if (proto.datatype() != inputInfo.getPrecisionAsKFSPrecision()) {
        std::stringstream ss;
        ss << "Expected: " << inputInfo.getPrecisionAsString()
           << "; Actual: " << proto.datatype()
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
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

template <>
bool RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::checkIfBinaryInputUsed(const TFSInputTensorType& proto, const std::string inputName) const {
    if (proto.dtype() == tensorflow::DataType::DT_STRING) {
        SPDLOG_DEBUG("[servable name: {} version: {}] Received request containing binary input: name: {}; batch size: {}", servableName, servableVersion, inputName, proto.string_val_size());
        return true;
    }
    return false;
}

template <>
bool RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>::checkIfBinaryInputUsed(const KFSInputTensorType& proto, const std::string inputName) const {
    if (proto.datatype() == "BYTES") {
        SPDLOG_DEBUG("[servable name: {} version: {}] Received request containing binary input: name: {}; batch size: {}", servableName, servableVersion, inputName, proto.contents().bytes_contents_size());
        return true;
    }
    return false;
}

template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::validate() {
    Status finalStatus = StatusCode::OK;

    auto status = validateNumberOfInputs();
    if (!status.ok())
        return status;
    status = validateRequestCoherency();
    if (!status.ok())
        return status;

    size_t bufferId = 0;
    for (const auto& [name, inputInfo] : inputsInfo) {
        status = validateAndGetInput(request, name, it, bufferId);
        if (!status.ok())
            return status;

        const auto& proto = getInputFromIt(it);

        status = checkIfShapeValuesNegative(proto);
        if (!status.ok())
            return status;
        auto batchIndex = inputInfo->getLayout().getBatchIndex();
        if (!batchIndex.has_value()) {
            SPDLOG_DEBUG("[servable name: {} version: {}] Missing batch index in input: {} layout: {}",
                servableName, servableVersion, name, inputInfo->getLayout());
            return StatusCode::INTERNAL_ERROR;
        }
        if (inputInfo->getShape().size() < batchIndex.value() + 1) {
            SPDLOG_DEBUG("[servable name: {} version: {}] Batch index out of shape range for input: {} layout: {} shape: {}",
                servableName, servableVersion, name, inputInfo->getLayout(), inputInfo->getShape().toString());
            return StatusCode::INTERNAL_ERROR;
        }
        const Dimension& batchSize = inputInfo->getShape()[batchIndex.value()];
        Mode shapeMode = getShapeMode(shapeInfo, name);
        if (checkIfBinaryInputUsed(proto, name)) {
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
        status = checkBatchSizeMismatch(proto, batchSize, batchIndex.value(), finalStatus, batchingMode, shapeMode);
        if (!status.ok())
            return status;
        status = checkShapeMismatch(proto, *inputInfo, batchIndex.value(), finalStatus, batchingMode, shapeMode);
        if (!status.ok())
            return status;
        status = validateTensorContentSize(proto, inputInfo->getPrecision(), bufferId);
        if (!status.ok())
            return status;
    }
    return finalStatus;
}

template <>
Status validate(const TFSRequestType& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

template <>
Status validate(const KFSRequestType& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType, KFSShapeType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}
}  // namespace request_validation_utils
}  // namespace ovms
