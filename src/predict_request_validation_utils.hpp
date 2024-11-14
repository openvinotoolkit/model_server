//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include <limits>
#include <set>
#include <string>
#include <vector>

//#include <google/protobuf/map.h>

#include "logging.hpp"
#include "modelversion.hpp"
#include "shape.hpp"
#include "requesttensorextractor.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"
#include "predict_request_validation_utils_impl.hpp"

namespace inference{
class KFSRequest;
}
class TFSRequestType;
namespace ovms {

template<typename RequestShapeType>
std::string tensorShapeToString(const RequestShapeType& tensorShape) { throw 42;}

namespace request_validation_utils {
enum class ValidationChoice {
    INPUT,
    OUTPUT,
};

template<typename RequestType>
bool dataInRawInputContents(RequestType& request);

template<typename RequestType>
const std::string* getRawInputContents(const RequestType& request, size_t bufferId);

template<typename RequestTensorType>
int64_t getStringBatchSize(const RequestTensorType& tensor);
template<typename RequestTensorType>
size_t getStringInputWidth(const RequestTensorType& tensor);

template <typename RequestTensorType, typename RequestTensorShapeType>
struct RequestShapeInfo {
    const RequestTensorType& tensor;
    RequestShapeInfo(const RequestTensorType& tensor) :
        tensor(tensor) {}
    dimension_value_t getDim(size_t i);
    size_t getShapeSize();
    const RequestTensorShapeType& getShape();
};

template <typename RequestType>
Status validate(
    const RequestType& request,
    const tensor_map_t& inputsInfo,
    const tensor_map_t& outputsInfo,
    const std::string& servableName,
    const model_version_t servableVersion,
    const std::set<std::string>& optionalAllowedInputNames = {},
    const Mode batchingMode = Mode::FIXED,
    const shapes_info_map_t& shapeInfo = shapes_info_map_t());

template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename InputIterator, typename ShapeType>
class RequestValidator {
    const RequestType& request;
    const tensor_map_t& inputsInfo;
    const tensor_map_t& outputsInfo;
    const std::string& servableName;
    const model_version_t servableVersion;
    const std::set<std::string>& optionalAllowedInputNames;
    const Mode batchingMode;
    const shapes_info_map_t& shapeInfo;

    const InputTensorType* proto{nullptr};

    RequestValidator() = delete;

    const std::string* currentlyValidatedName;

    const std::string getCurrentlyValidatedTensorName() const;
    const InputTensorType& getTensorFromIt(const InputIterator& it) const;

public:
    RequestValidator(
        const RequestType& request, const tensor_map_t& inputsInfo, const tensor_map_t& outputsInfo,
        const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames,
        const Mode batchingMode, const shapes_info_map_t& shapeInfo) :
        request(request),
        inputsInfo(inputsInfo),
        outputsInfo(outputsInfo),
        servableName(servableName),
        servableVersion(servableVersion),
        optionalAllowedInputNames(optionalAllowedInputNames),
        batchingMode(batchingMode),
        shapeInfo(shapeInfo) {}

    Status validateInferenceTensorBufferType(const InputTensorType& it) const;
    Status validateNumberOfTensors() const;
    Status validateAndGetTensor(const RequestType& request, const std::string& name, size_t& bufferId);
    Status checkIfShapeValuesNegative(const InputTensorType& proto) const;
    Status validateNumberOfBinaryInputShapeDimensions(const InputTensorType& proto) const;
    Status checkBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status checkBinaryBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const;
    Status checkShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& tensorInfo, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validateTensorContent(const InputTensorType& proto, ovms::Precision expectedPrecision, size_t bufferId) const;
    Status validateNumberOfShapeDimensions(const ovms::TensorInfo& tensorInfo, const InputTensorType& proto) const;
    Status validateRawInputContentsFormatAndShape(const ovms::TensorInfo& tensorInfo, const RequestType& request, const size_t& bufferId, Status& finalStatus, Mode batchingMode, Mode shapeMode) const;
    Status validatePrecision(const ovms::TensorInfo& tensorInfo, const InputTensorType& proto) const;
    Status checkStringShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& tensorInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize, size_t inputWidth) const;
    Status validateRequestCoherency() const;
    Status validate();
};

template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename InputTensorIteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, InputTensorIteratorType, ShapeType>::checkIfShapeValuesNegative(const InputTensorType& proto) const {
    RequestShapeInfo<InputTensorType, ShapeType> rsi(proto);
    for (size_t i = 0; i < rsi.getShapeSize(); i++) {
        if (rsi.getDim(i) < 0) {
            std::stringstream ss;
            ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(rsi.getShape()) << "; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}
// To be called only for already validated proto against models with two dimensions.
template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, IteratorType, ShapeType>::checkStringShapeMismatch(const InputTensorType& proto, const ovms::TensorInfo& tensorInfo, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize, size_t inputWidth) const {
    const auto& shape = tensorInfo.getShape();
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
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    return StatusCode::OK;
}

template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename InputIteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, InputIteratorType, ShapeType>::checkBatchSizeMismatch(const InputTensorType& proto, const std::optional<Dimension>& servableBatchSize, const std::optional<size_t>& batchSizeIndex, Status& finalStatus, Mode batchingMode, Mode shapeMode) const {
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
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << rsi.getDim(batchSizeIndex.value()) << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename InputIterator, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, InputIterator, ShapeType>::validateAndGetTensor(const RequestType& request, const std::string& name, size_t& bufferId) {
    Status status;
    if (choice == ValidationChoice::INPUT) {
        status = RequestTensorExtractor<RequestType, InputTensorType, ExtractChoice::EXTRACT_INPUT>::extract(request, name, &proto, &bufferId);
    }
    if (choice == ValidationChoice::OUTPUT) {
        status = RequestTensorExtractor<RequestType, InputTensorType, ExtractChoice::EXTRACT_OUTPUT>::extract(request, name, &proto, &bufferId);
    }
    if (status.ok()) {
        currentlyValidatedName = &name;
        return StatusCode::OK;
    }

    currentlyValidatedName = nullptr;
    StatusCode code = StatusCode::INTERNAL_ERROR;
    std::stringstream ss;
    if (choice == ValidationChoice::INPUT) {
        ss << "Required input: ";
        code = StatusCode::INVALID_MISSING_INPUT;
    }
    if (choice == ValidationChoice::OUTPUT) {
        ss << "Optional output: ";
        code = StatusCode::INVALID_MISSING_OUTPUT;
    }
    ss << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[servable name: {} version: {}] Missing tensor with specific name - {}", servableName, servableVersion, details);
    return Status(code, details);
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


#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }
template <typename RequestType, typename InputTensorType, ValidationChoice choice, typename IteratorType, typename ShapeType>
Status RequestValidator<RequestType, InputTensorType, choice, IteratorType, ShapeType>::validate() {
    if ((std::is_same<RequestType, ::inference::KFSRequest>::value || std::is_same<RequestType, TFSRequestType>::value) && choice == ValidationChoice::OUTPUT) {
        return StatusCode::NOT_IMPLEMENTED;
    }
    Status finalStatus = StatusCode::OK;
    RETURN_IF_ERR(validateNumberOfTensors());
    RETURN_IF_ERR(validateRequestCoherency());
    size_t bufferId = 0;
    for (const auto& [name, tensorInfo] : ((choice == ValidationChoice::INPUT) ? inputsInfo : outputsInfo)) {
        auto getTensorStatus = validateAndGetTensor(request, name, bufferId);
        if (!getTensorStatus.ok() && choice == ValidationChoice::OUTPUT) {
            continue;
        }
        RETURN_IF_ERR(getTensorStatus);
        RETURN_IF_ERR(checkIfShapeValuesNegative(*proto));

        // Batch and mode retrieval for given input
        auto batchIndex = tensorInfo->getLayout().getBatchIndex();
        if (batchIndex.has_value() && batchIndex.value() >= tensorInfo->getShape().size()) {
            SPDLOG_DEBUG("[servable name: {} version: {}] Batch index out of shape range for input: {} layout: {} shape: {}",
                servableName, servableVersion, name, tensorInfo->getLayout(), tensorInfo->getShape().toString());
            return StatusCode::INTERNAL_ERROR;
        }

        Mode shapeMode = getShapeMode(shapeInfo, name);
        if (choice == ValidationChoice::INPUT) {
            if (requiresPreProcessing(*proto)) {
                const auto processingHint = tensorInfo->getPreProcessingHint();
                int32_t inputBatchSize = 0;
                size_t inputWidth = 0;
                if (dataInRawInputContents(request)) {
                    const std::string* buffer = getRawInputContents(request, bufferId);
                    RETURN_IF_ERR(getRawInputContentsBatchSizeAndWidth(*buffer, inputBatchSize, inputWidth));
                } else {
                    inputBatchSize = getStringBatchSize(*proto);
                    inputWidth = getStringInputWidth(*proto);
                }
                if (processingHint == TensorInfo::ProcessingHint::STRING_NATIVE) {
                    // Pass through to normal validation
                } else if (processingHint == TensorInfo::ProcessingHint::STRING_2D_U8) {
                    SPDLOG_DEBUG("[servable name: {} version: {}] Validating request containing 2D string input: name: {}",
                        servableName, servableVersion, name);
                    RETURN_IF_ERR(validateNumberOfBinaryInputShapeDimensions(*proto));
                    RETURN_IF_ERR(validateAgainstMax2DStringArraySize(inputBatchSize, inputWidth));
                    RETURN_IF_ERR(checkBinaryBatchSizeMismatch(*proto, tensorInfo->getBatchSize(), finalStatus, batchingMode, shapeMode, inputBatchSize));  // 2 dimensions assumed
                    RETURN_IF_ERR(checkStringShapeMismatch(*proto, *tensorInfo, finalStatus, batchingMode, shapeMode, inputBatchSize, inputWidth));
                    continue;
                } else if (processingHint == TensorInfo::ProcessingHint::IMAGE) {
                    SPDLOG_DEBUG("[servable name: {} version: {}] Validating request containing binary image input: name: {}",
                        servableName, servableVersion, name);
                    RETURN_IF_ERR(validateNumberOfBinaryInputShapeDimensions(*proto));
                    RETURN_IF_ERR(checkBinaryBatchSizeMismatch(*proto, tensorInfo->getBatchSize(), finalStatus, batchingMode, shapeMode, inputBatchSize));  // 4/5 dimensions assumed
                    continue;
                } else {
                    SPDLOG_DEBUG("Request input: {} requires conversion but endpoint specifies no processing hint. Number of dimensions: {}; precision: {}; demultiplexer: {}",
                        name, tensorInfo->getShape().size(), toString(tensorInfo->getPrecision()), tensorInfo->isInfluencedByDemultiplexer());
                    return StatusCode::NOT_IMPLEMENTED;
                }
            }
        }

        // Data Array Proto
        RETURN_IF_ERR(validatePrecision(*tensorInfo, *proto));
        RETURN_IF_ERR(validateNumberOfShapeDimensions(*tensorInfo, *proto));
        RETURN_IF_ERR(checkBatchSizeMismatch(*proto, tensorInfo->getBatchSize(), batchIndex, finalStatus, batchingMode, shapeMode));
        RETURN_IF_ERR(checkShapeMismatch(*proto, *tensorInfo, batchIndex, finalStatus, batchingMode, shapeMode));
        RETURN_IF_ERR(validateTensorContent(*proto, tensorInfo->getPrecision(), bufferId));
    }
    return finalStatus;
}
// This function is expected to be called with already validated shape that does not contain negative dimensions
template <typename T>
static bool computeExpectedBufferSizeReturnFalseIfOverflow(const std::vector<T>& shape, const size_t& itemsize, size_t& expectedBufferSize) {
    expectedBufferSize = 1;
    if (itemsize == 0) {
        expectedBufferSize = 0;
        return true;
    }
    for (const T& dim : shape) {
        if (dim == 0) {
            expectedBufferSize = 0;
            return true;
        }
        if (expectedBufferSize > std::numeric_limits<size_t>::max() / dim)
            return false;
        expectedBufferSize *= dim;
    }
    if (expectedBufferSize > std::numeric_limits<size_t>::max() / itemsize)
        return false;
    expectedBufferSize *= itemsize;
    return true;
}
// TODO FIXME remove Status validateAgainstMax2DStringArraySize(int32_t inputBatchSize, size_t inputWidth);
}  // namespace request_validation_utils
}  // namespace ovms
