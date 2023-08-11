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
#include "profiler.hpp"
#include "status.hpp"
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

using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, TFSInputTensorType>::const_iterator;
using TFSShapeType = tensorflow::TensorShapeProto;

template <>
dimension_value_t RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getDim(size_t i) {
    return tensor.shape()[i];
}
template <>
dimension_value_t RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getDim(size_t i) {
    return tensor.tensor_shape().dim(i).size();
}

template <>
dimension_value_t RequestShapeInfo<InferenceTensor, signed_shape_t>::getDim(size_t i) {
    return tensor.getShape()[i];
}
template <>
size_t RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getShapeSize() {
    return tensor.shape().size();
}
template <>
size_t RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getShapeSize() {
    return tensor.tensor_shape().dim_size();
}
template <>
size_t RequestShapeInfo<InferenceTensor, signed_shape_t>::getShapeSize() {
    return tensor.getShape().size();
}
template <>
const TFSShapeType& RequestShapeInfo<TFSInputTensorType, TFSShapeType>::getShape() {
    return tensor.tensor_shape();
}
template <>
const KFSShapeType& RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getShape() {
    return tensor.shape();
}
template <>
const signed_shape_t& RequestShapeInfo<InferenceTensor, signed_shape_t>::getShape() {
    return tensor.getShape();
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

    const std::string* currentlyValidatedName;

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

    Status validateInferenceTensorBufferType(const InferenceTensor& it) const;
    Status validateNumberOfInputs() const;
    Status validateAndGetInput(const RequestType& request, const std::string& name, InputIterator& it, size_t& bufferId);
    Status checkIfShapeValuesNegative(const InputTensorType& proto) const;
    Status validateNumberOfBinaryInputShapeDimensions(const InputTensorType& proto) const;
    Status checkBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkBinaryBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const;
    Status checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validateTensorContent(const InputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const;
    Status validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const InputTensorType& proto) const;
    Status validateRawInputContentsFormatAndShape(const ovms::TensorInfo& inputInfo, const RequestType& request, const size_t& bufferId, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validatePrecision(const ovms::TensorInfo& inputInfo, const InputTensorType& proto) const;
    Status checkStringShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize, size_t inputWidth) const;
    Status validateRequestCoherency() const;
    Status validate();
};

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateRequestCoherency() const {
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
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateRequestCoherency() const {
    return StatusCode::OK;
}

template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfInputs() const {
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
const std::string& RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::getCurrentlyValidatedInputName() const {
    return *currentlyValidatedName;
}
template <>
const std::string& RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::getCurrentlyValidatedInputName() const {
    return *currentlyValidatedName;
}
template <>
const std::string& RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::getCurrentlyValidatedInputName() const {
    return *currentlyValidatedName;
}

template <>
const KFSTensorInputProto& RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::getInputFromIt(const KFSInputTensorIteratorType& it) const {
    return *it;
}
template <>
const TFSInputTensorType& RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::getInputFromIt(const TFSInputTensorIteratorType& it) const {
    return it->second;
}
template <>
const InferenceTensor& RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::getInputFromIt(const InferenceTensor* const& it) const {
    return *it;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfInputs() const {
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
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateNumberOfInputs() const {
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
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateAndGetInput(const TFSRequestType& request, const std::string& name, TFSInputTensorIteratorType& it, size_t& bufferId) {
    it = request.inputs().find(name);
    if (it != request.inputs().end()) {
        currentlyValidatedName = &name;
        return StatusCode::OK;
    }
    currentlyValidatedName = nullptr;
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateAndGetInput(const KFSRequest& request, const std::string& name, KFSInputTensorIteratorType& it, size_t& bufferId) {
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
        currentlyValidatedName = &name;
        return StatusCode::OK;
    }
    currentlyValidatedName = nullptr;
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", servableName, servableVersion, details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}

template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateAndGetInput(const InferenceRequest& request, const std::string& name, const InferenceTensor*& it, size_t& bufferId) {
    if (request.getInput(name.c_str(), &it) != StatusCode::NONEXISTENT_TENSOR) {
        currentlyValidatedName = &name;
        return StatusCode::OK;
    }

    currentlyValidatedName = nullptr;
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
        if (rsi.getDim(i) < 0) {
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

const size_t MAX_2D_STRING_ARRAY_SIZE = 1024 * 1024 * 1024 * 1;  // 1GB

static Status validateAgainstMax2DStringArraySize(int32_t inputBatchSize, size_t inputWidth) {
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

template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfBinaryInputShapeDimensions(const KFSTensorInputProto& proto) const {
    RequestShapeInfo<KFSTensorInputProto, KFSShapeType> rsi(proto);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateNumberOfBinaryInputShapeDimensions(const InferenceTensor& tensor) const {
    RequestShapeInfo<InferenceTensor, signed_shape_t> rsi(tensor);
    if (rsi.getShapeSize() != 1) {
        std::stringstream ss;
        ss << "Expected number of input shape dimensions: 1; Actual: " << rsi.getShapeSize() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, typename InputIteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, InputIteratorType, ShapeType>::checkBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    if (!servableBatchSize.has_value() || !batchSizeIndex.has_value()) {
        // Do not validate batch size in this case.
        // Let entire shape be validated instead.
        return StatusCode::OK;
    }
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    if (servableBatchSize.value().match(rsi.getDim(batchSizeIndex.value()))) {
        return StatusCode::OK;
    }
    if (batchingMode == AUTO) {
        finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
        return StatusCode::OK;
    } else if (shapeMode != AUTO) {
        std::stringstream ss;
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << rsi.getDim(batchSizeIndex.value()) << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::checkBinaryBatchSizeMismatch(const TFSInputTensorType& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<TFSInputTensorType, TFSShapeType> rsi(proto);
    if (inputBatchSize <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
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
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << proto.string_val_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::checkBinaryBatchSizeMismatch(const KFSTensorInputProto& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<KFSTensorInputProto, KFSShapeType> rsi(proto);
    if (inputBatchSize <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
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
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << proto.contents().bytes_contents_size() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::checkBinaryBatchSizeMismatch(const InferenceTensor& tensor, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<InferenceTensor, signed_shape_t> rsi(tensor);
    if (rsi.getDim(0) <= 0) {
        std::stringstream ss;
        ss << "Batch size must be positive; input name: " << getCurrentlyValidatedInputName();
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
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << tensor.getBuffer()->getByteSize() << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = inputInfo.getShape();
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
        ss << "Expected: " << inputInfo.getShape().toString()
           << "; Actual: " << tensorShapeToString(rsi.getShape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

static size_t getStringInputWidth(const KFSTensorInputProto& src) {
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
}

// To be called only for already validated proto against models with two dimensions.
template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::checkStringShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& inputInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize, size_t inputWidth) const {
    const auto& shape = inputInfo.getShape();
    bool mismatch = false;
    if (batchingMode == AUTO) {  // Skip batch dimension
        if (!shape[1].match(static_cast<dimension_value_t>(inputWidth))) {
            mismatch = true;
        }
    } else {  // Do not skip batch dimension
        if (!shape.match(ov::Shape{static_cast<uint64_t>(inputBatchSize), inputWidth})) {
            mismatch = true;
        }
    }
    if (!mismatch) {
        return StatusCode::OK;
    }
    if (shapeMode == AUTO) {
        finalStatus = StatusCode::RESHAPE_REQUIRED;
        return StatusCode::OK;
    } else {
        auto stringInputShape = Shape({static_cast<int64_t>(inputBatchSize), static_cast<int64_t>(inputWidth)});
        std::stringstream ss;
        ss << "Expected batch size: " << shape[0].toString()
           << "; got: " << inputBatchSize
           << "; Expected max null terminated string length: " << shape[1].toString()
           << "; got: " << inputWidth
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::validateInferenceTensorBufferType(const InferenceTensor& it) const {
    const Buffer* buffer = it.getBuffer();
    const OVMS_BufferType bufType = buffer->getBufferType();
    if (bufType < OVMS_BUFFERTYPE_CPU || bufType > OVMS_BUFFERTYPE_HDDL) {
        std::stringstream ss;
        ss << "Required input ";
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid buffer type for input with specific name - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BUFFER_TYPE, details);

    } else {
        // Remove this when other buffer types are supported
        if (bufType != OVMS_BUFFERTYPE_CPU) {
            std::stringstream ss;
            ss << "Required input ";
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid buffer type for input with specific name - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_BUFFER_TYPE, details);
        }
    }

    if (buffer->getBufferType() == OVMS_BUFFERTYPE_CPU && buffer->getDeviceId() != std::nullopt && buffer->getDeviceId() != 0) {
        std::stringstream ss;
        ss << "Required input ";
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Has invalid device id for buffer, input with specific name - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_DEVICE_ID, details);
    }

    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateTensorContent(const TFSInputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
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

static size_t getElementsCount(const KFSTensorInputProto& proto, ovms::Precision expectedPrecision) {
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
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateTensorContent(const KFSTensorInputProto& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
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
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateTensorContent(const InferenceTensor& tensor, ovms::Precision expectedPrecision, size_t bufferId) const {
    const Buffer* buffer = tensor.getBuffer();
    if (nullptr == buffer) {
        std::stringstream ss;
        ss << "Servable: " << servableName
           << "; version: " << servableVersion
           << "; is missing buffer for tensor: " << bufferId;
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }
    size_t expectedValueCount = 1;
    for (size_t i = 0; i < tensor.getShape().size(); i++) {
        expectedValueCount *= tensor.getShape()[i];
    }
    size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
    if (expectedContentSize != buffer->getByteSize()) {
        std::stringstream ss;
        ss << "Expected: " << expectedContentSize << " bytes; Actual: " << buffer->getByteSize() << " bytes; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }

    return validateInferenceTensorBufferType(tensor);
}

template <>
Status RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const TFSInputTensorType& proto) const {
    // Network and request must have the same number of shape dimensions
    const auto& shape = inputInfo.getShape();
    if (proto.tensor_shape().dim_size() < 0 ||
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
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const KFSTensorInputProto& proto) const {
    // Network and request must have the same number of shape dimensions
    const auto& shape = inputInfo.getShape();
    if (proto.shape().size() < 0 ||
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
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const InferenceTensor& tensor) const {
    // Network and request must have the same number of shape dimensions
    const auto& shape = inputInfo.getShape();
    if (tensor.getShape().size() < 0 ||
        shape.size() != static_cast<size_t>(tensor.getShape().size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(tensor.getShape())
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
Status RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>::validatePrecision(const ovms::TensorInfo& inputInfo, const KFSTensorInputProto& proto) const {
    if (proto.datatype() != ovmsPrecisionToKFSPrecision(inputInfo.getPrecision())) {
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
template <>
Status RequestValidator<ovms::InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>::validatePrecision(const ovms::TensorInfo& inputInfo, const InferenceTensor& tensor) const {
    if (tensor.getDataType() != getPrecisionAsOVMSDataType(inputInfo.getPrecision())) {
        std::stringstream ss;
        ss << "Expected: " << inputInfo.getPrecisionAsString()
           << "; Actual: " << tensor.getDataType()
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

static Mode getShapeMode(const shapes_info_map_t& shapeInfo, const std::string& name) {
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

static bool dataInRawInputContents(const ovms::InferenceRequest& request) {
    return false;
}

static bool dataInRawInputContents(const TFSRequestType& request) {
    return false;
}

static bool dataInRawInputContents(const KFSRequest& request) {
    return request.raw_input_contents().size() > 0;
}

static const std::string* getRawInputContents(const ovms::InferenceRequest& request, size_t bufferId) {
    SPDLOG_DEBUG("Raw input contents is not supported for C-API");
    throw std::runtime_error("Raw input contents used in C-API flow.");
    return nullptr;
}

static const std::string* getRawInputContents(const TFSRequestType& request, size_t bufferId) {
    SPDLOG_DEBUG("Raw input contents is not supported for TFS API");
    throw std::runtime_error("Raw input contents used in TFSflow.");
    return nullptr;
}

static const std::string* getRawInputContents(const KFSRequest& request, size_t bufferId) {
    return &(request.raw_input_contents().at(bufferId));
}

#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }

template <typename RequestType, typename InputTensorType, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, IteratorType, ShapeType>::validate() {
    Status finalStatus = StatusCode::OK;

    RETURN_IF_ERR(validateNumberOfInputs());
    RETURN_IF_ERR(validateRequestCoherency());

    size_t bufferId = 0;
    for (const auto& [name, inputInfo] : inputsInfo) {
        RETURN_IF_ERR(validateAndGetInput(request, name, it, bufferId));

        const auto& proto = getInputFromIt(it);

        RETURN_IF_ERR(checkIfShapeValuesNegative(proto));

        // Batch and mode retrieval for given input
        auto batchIndex = inputInfo->getLayout().getBatchIndex();
        if (batchIndex.has_value() && batchIndex.value() >= inputInfo->getShape().size()) {
            SPDLOG_DEBUG("[servable name: {} version: {}] Batch index out of shape range for input: {} layout: {} shape: {}",
                servableName, servableVersion, name, inputInfo->getLayout(), inputInfo->getShape().toString());
            return StatusCode::INTERNAL_ERROR;
        }

        Mode shapeMode = getShapeMode(shapeInfo, name);

        if (requiresPreProcessing(proto)) {
            const auto processingHint = inputInfo->getPreProcessingHint();
            int32_t inputBatchSize = 0;
            size_t inputWidth = 0;
            if (dataInRawInputContents(request)) {
                const std::string* buffer = getRawInputContents(request, bufferId);
                RETURN_IF_ERR(getRawInputContentsBatchSizeAndWidth(*buffer, inputBatchSize, inputWidth));
            } else {
                inputBatchSize = getStringBatchSize(proto);
                inputWidth = getStringInputWidth(proto);
            }
            if (processingHint == TensorInfo::ProcessingHint::STRING_1D_U8) {
                SPDLOG_DEBUG("[servable name: {} version: {}] Validating request containing 1D string input: name: {}",
                    servableName, servableVersion, name);
                RETURN_IF_ERR(validateNumberOfBinaryInputShapeDimensions(proto));
                continue;
            } else if (processingHint == TensorInfo::ProcessingHint::STRING_2D_U8) {
                SPDLOG_DEBUG("[servable name: {} version: {}] Validating request containing 2D string input: name: {}",
                    servableName, servableVersion, name);
                RETURN_IF_ERR(validateNumberOfBinaryInputShapeDimensions(proto));
                RETURN_IF_ERR(validateAgainstMax2DStringArraySize(inputBatchSize, inputWidth));
                RETURN_IF_ERR(checkBinaryBatchSizeMismatch(proto, inputInfo->getBatchSize(), finalStatus, batchingMode, shapeMode, inputBatchSize));  // 2 dimensions assumed
                RETURN_IF_ERR(checkStringShapeMismatch(proto, *inputInfo, finalStatus, batchingMode, shapeMode, inputBatchSize, inputWidth));
                continue;
            } else if (processingHint == TensorInfo::ProcessingHint::IMAGE) {
                SPDLOG_DEBUG("[servable name: {} version: {}] Validating request containing binary image input: name: {}",
                    servableName, servableVersion, name);
                RETURN_IF_ERR(validateNumberOfBinaryInputShapeDimensions(proto));
                RETURN_IF_ERR(checkBinaryBatchSizeMismatch(proto, inputInfo->getBatchSize(), finalStatus, batchingMode, shapeMode, inputBatchSize));  // 4/5 dimensions assumed
                continue;
            } else {
                SPDLOG_DEBUG("Request input: {} requires conversion but endpoint specifies no processing hint. Number of dimensions: {}; precision: {}; demultiplexer: {}",
                    name, inputInfo->getShape().size(), toString(inputInfo->getPrecision()), inputInfo->isInfluencedByDemultiplexer());
                return StatusCode::NOT_IMPLEMENTED;
            }
        }

        // Data Array Proto
        RETURN_IF_ERR(validatePrecision(*inputInfo, proto));
        RETURN_IF_ERR(validateNumberOfShapeDimensions(*inputInfo, proto));
        RETURN_IF_ERR(checkBatchSizeMismatch(proto, inputInfo->getBatchSize(), batchIndex, finalStatus, batchingMode, shapeMode));
        RETURN_IF_ERR(checkShapeMismatch(proto, *inputInfo, batchIndex, finalStatus, batchingMode, shapeMode));
        RETURN_IF_ERR(validateTensorContent(proto, inputInfo->getPrecision(), bufferId));
    }
    return finalStatus;
}

template <>
Status validate(const TFSRequestType& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType, TFSShapeType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

template <>
Status validate(const KFSRequest& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<KFSRequest, KFSTensorInputProto, KFSInputTensorIteratorType, KFSShapeType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

template <>
Status validate(const InferenceRequest& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<InferenceRequest, InferenceTensor, const InferenceTensor*, signed_shape_t>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}
}  // namespace request_validation_utils
}  // namespace ovms
