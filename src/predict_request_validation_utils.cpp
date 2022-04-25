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

namespace ovms {
namespace request_validation_utils {
template <typename RequestTensorType, typename RequestTensorShapeType>
struct RequestShapeInfo {
    const RequestTensorType& r;
    RequestShapeInfo(const RequestTensorType& r) :
        r(r) {}
    dimension_value_t getDim(size_t i);
    size_t getShapeSize();
    const RequestTensorShapeType& getShape();
};

using KFSRequestType = ::inference::ModelInferRequest;
using KFSInputTensorType = inference::ModelInferRequest_InferInputTensor;
using KFSInputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferRequest_InferInputTensor>;
using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator;

template <>
dimension_value_t RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>>::getDim(size_t i) {
    return r.shape()[i];
}
template <>
dimension_value_t RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto>::getDim(size_t i) {
    return r.tensor_shape().dim(i).size();
}
template <>
size_t RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>>::getShapeSize() {
    return r.shape().size();
}
template <>
size_t RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto>::getShapeSize() {
    return r.tensor_shape().dim_size();
}
template <>
const tensorflow::TensorShapeProto& RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto>::getShape() {
    return r.tensor_shape();
}
template <>
const google::protobuf::RepeatedField<int64_t>& RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>>::getShape() {
    return r.shape();
}

template <typename RequestType, typename InputType, typename InputIterator>
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
    const InputType& getInputFromIt(const InputIterator& it) const;

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
    Status checkIfShapeValuesNegative(const InputType& proto) const;
    Status validateNumberOfBinaryInputShapeDimensions(const InputType& proto) const;
    Status checkBatchSizeMismatch(const InputType& proto, const Dimension& servableBatchSize, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkBinaryBatchSizeMismatch(const InputType& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkShapeMismatch(const InputType& proto, const ovms::TensorInfo& inputInfo, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validateTensorContentSize(const InputType& proto, ovms::Precision expectedPrecision, size_t bufferId) const;
    Status validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const InputType& proto) const;
    Status validatePrecision(const ovms::TensorInfo& inputInfo, const InputType& proto) const;
    bool checkIfBinaryInputUsed(const InputType& proto, const std::string inputName) const;
    Status validate();
};

template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validateNumberOfInputs() const {
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
const std::string& RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::getCurrentlyValidatedInputName() const {
    return it->name();
}

template <>
const std::string& RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::getCurrentlyValidatedInputName() const {
    return it->first;
}
template <>
const KFSInputTensorType& RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::getInputFromIt(const KFSInputTensorIteratorType& it) const {
    return *it;
}

template <>
const tensorflow::TensorProto& RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::getInputFromIt(const google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it) const {
    return it->second;
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validateNumberOfInputs() const {
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
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validateAndGetInput(const TFSRequestType& request, const std::string& name, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it, size_t& bufferId) {
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
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validateAndGetInput(const ::inference::ModelInferRequest& request, const std::string& name, KFSInputTensorIteratorType& it, size_t& bufferId) {
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
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::checkIfShapeValuesNegative(const tensorflow::TensorProto& proto) const {
    RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto> rsi(proto);
    for (size_t i = 0; i < rsi.getShapeSize(); i++) {
        if (rsi.getDim(i) <= 0) {
            std::stringstream ss;
            ss << "Negative or zero dimension size is not acceptable: " << TensorInfo::tensorShapeToString(rsi.getShape()) << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::checkIfShapeValuesNegative(const KFSInputTensorType& proto) const {
    RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>> rsi(proto);
    const auto& shape = proto.shape();
    for (size_t i = 0; i < rsi.getShapeSize(); ++i) {
        if (rsi.getDim(i) <= 0) {
            std::stringstream ss;
            ss << "Negative or zero dimension size is not acceptable: " << TensorInfo::tensorShapeToString(shape) << "; input name: " << getCurrentlyValidatedInputName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validateNumberOfBinaryInputShapeDimensions(const tensorflow::TensorProto& proto) const {
    RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto> rsi(proto);
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
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validateNumberOfBinaryInputShapeDimensions(const KFSInputTensorType& proto) const {
    return StatusCode::OK;  //  TODO implement with KFS binary inputs
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::checkBatchSizeMismatch(const tensorflow::TensorProto& proto, const Dimension& servableBatchSize, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto> rsi(proto);
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
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::checkBatchSizeMismatch(const KFSInputTensorType& proto, const Dimension& servableBatchSize, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>> rsi(proto);
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
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::checkBinaryBatchSizeMismatch(const tensorflow::TensorProto& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto> rsi(proto);
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
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::checkBinaryBatchSizeMismatch(const KFSInputTensorType& proto, const Dimension& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    return StatusCode::OK;  //  TODO implement with KFS binary inputs
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::checkShapeMismatch(const tensorflow::TensorProto& proto, const ovms::TensorInfo& inputInfo, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = inputInfo.getShape();
    bool mismatch = false;
    RequestShapeInfo<tensorflow::TensorProto, tensorflow::TensorShapeProto> rsi(proto);
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
           << "; Actual: " << TensorInfo::tensorShapeToString(rsi.getShape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::checkShapeMismatch(const KFSInputTensorType& proto, const ovms::TensorInfo& inputInfo, const size_t batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
    const auto& shape = inputInfo.getShape();
    bool mismatch = false;
    RequestShapeInfo<KFSInputTensorType, google::protobuf::RepeatedField<int64_t>> rsi(proto);
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
           << "; Actual: " << TensorInfo::tensorShapeToString(rsi.getShape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validateTensorContentSize(const tensorflow::TensorProto& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
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

template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validateTensorContentSize(const KFSInputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
    size_t expectedValueCount = 1;
    for (int i = 0; i < proto.shape().size(); i++) {
        expectedValueCount *= proto.shape()[i];
    }

    // TODO KFS can store buffer inside ModelInferRequest_InferInputTensor as well
    size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
    if (expectedContentSize != request.raw_input_contents()[bufferId].size()) {
        std::stringstream ss;
        ss << "Expected: " << expectedContentSize << " bytes; Actual: " << request.raw_input_contents()[bufferId].size() << " bytes; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getShape();
    if (proto.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.tensor_shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& inputInfo, const KFSInputTensorType& proto) const {
    // Network and request must have the same number of shape dimensions, higher than 0
    const auto& shape = inputInfo.getShape();
    if (proto.shape().size() <= 0 ||
        shape.size() != static_cast<size_t>(proto.shape().size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << TensorInfo::tensorShapeToString(proto.shape())
           << "; input name: " << getCurrentlyValidatedInputName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

template <>
Status RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::validatePrecision(const ovms::TensorInfo& inputInfo, const tensorflow::TensorProto& proto) const {
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

template <>
Status RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::validatePrecision(const ovms::TensorInfo& inputInfo, const KFSInputTensorType& proto) const {
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
bool RequestValidator<TFSRequestType, tensorflow::TensorProto, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator>::checkIfBinaryInputUsed(const tensorflow::TensorProto& proto, const std::string inputName) const {
    if (proto.dtype() == tensorflow::DataType::DT_STRING) {
        SPDLOG_DEBUG("[servable name: {} version: {}] Received request containing binary input: name: {}; batch size: {}", servableName, servableVersion, inputName, proto.string_val_size());
        return true;
    }
    return false;
}

// TODO verify/implement for KFS binary inputs
template <>
bool RequestValidator<::inference::ModelInferRequest, KFSInputTensorType, KFSInputTensorIteratorType>::checkIfBinaryInputUsed(const KFSInputTensorType& proto, const std::string inputName) const {
    if (proto.datatype() == "BYTES") {
        SPDLOG_DEBUG("[servable name: {} version: {}] Received request containing binary input: name: {}; batch size: {}", servableName, servableVersion, inputName, 0);
        return true;
    }
    return false;
}

template <typename RequestType, typename InputType, typename IteratorType>
Status RequestValidator<RequestType, InputType, IteratorType>::validate() {
    Status finalStatus = StatusCode::OK;

    auto status = validateNumberOfInputs();
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
    return RequestValidator<TFSRequestType, TFSInputTensorType, TFSInputTensorIteratorType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

template <>
Status validate(const ::inference::ModelInferRequest& request, const tensor_map_t& inputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<KFSRequestType, KFSInputTensorType, KFSInputTensorIteratorType>(request, inputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}

}  // namespace request_validation_utils
}  // namespace ovms
