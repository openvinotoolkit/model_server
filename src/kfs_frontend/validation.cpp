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

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "../tensor_conversion_common.hpp"
#include "kfs_utils.hpp"
#include "../precision.hpp"
#include "../predict_request_validation_utils.hpp"
#include "../profiler.hpp"
#include "../logging.hpp"
#include "../status.hpp"

namespace ovms {
namespace request_validation_utils {
template <>
dimension_value_t RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getDim(size_t i) {
    return tensor.shape()[i];
}
template <>
size_t RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getShapeSize() {
    return tensor.shape().size();
}
template <>
const KFSShapeType& RequestShapeInfo<KFSTensorInputProto, KFSShapeType>::getShape() {
    return tensor.shape();
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validateRequestCoherency() const {
    return validateRequestCoherencyKFS(this->request, this->servableName, this->servableVersion);
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfTensors() const {
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
const std::string RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::getCurrentlyValidatedTensorName() const {
    return "input name: " + *currentlyValidatedName;
}
template <>
const KFSTensorInputProto& RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::getTensorFromIt(const KFSInputTensorIteratorType& it) const {
    return *it;
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfBinaryInputShapeDimensions(const KFSTensorInputProto& proto) const {
    RequestShapeInfo<KFSTensorInputProto, KFSShapeType> rsi(proto);
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
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::checkBinaryBatchSizeMismatch(const KFSTensorInputProto& proto, const std::optional<Dimension>& servableBatchSize, Status& finalStatus, Mode batchingMode, Mode shapeMode, int32_t inputBatchSize) const {
    if (!servableBatchSize.has_value()) {
        std::stringstream ss;
        ss << "Batch not present in input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    RequestShapeInfo<KFSTensorInputProto, KFSShapeType> rsi(proto);
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
        ss << "Expected: " << servableBatchSize.value().toString() << "; Actual: " << proto.contents().bytes_contents_size() << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid batch size - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_BATCH_SIZE, details);
    }
    return StatusCode::OK;
}
template <>
size_t getStringInputWidth(const KFSTensorInputProto& src) {
    size_t maxStringLength = 0;
    for (const auto& str : src.contents().bytes_contents()) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    return maxStringLength + 1;
}
template <>
int64_t getStringBatchSize(const KFSTensorInputProto& src) {
    return src.contents().bytes_contents_size();
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validateTensorContent(const KFSTensorInputProto& proto, ovms::Precision expectedPrecision, size_t bufferId) const {
    size_t expectedValueCount = 1;
    for (int i = 0; i < proto.shape().size(); i++) {
        expectedValueCount *= proto.shape()[i];
    }
    if (request.raw_input_contents().size()) {
        if (proto.datatype() == "BYTES") {
            // Special content validation - 4 byte length metadata
            size_t processedBytes = 0;
            size_t batchSize = 0;
            while (request.raw_input_contents(bufferId).size() >= processedBytes + sizeof(uint32_t)) {
                uint32_t size = *reinterpret_cast<const uint32_t*>(request.raw_input_contents(bufferId).data() + processedBytes);
                if (processedBytes + size + sizeof(uint32_t) > request.raw_input_contents(bufferId).size()) {
                    std::stringstream ss;
                    ss << "Batch length metadata exceeded buffer size, buffer size: " << request.raw_input_contents(bufferId).size() << ", batch length: " << size << "; input name: " << getCurrentlyValidatedTensorName();
                    const std::string details = ss.str();
                    SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
                    return Status(StatusCode::INVALID_CONTENT_SIZE, details);
                }
                processedBytes += size + sizeof(uint32_t);
                batchSize++;
            }
            if (request.raw_input_contents(bufferId).size() != processedBytes) {
                std::stringstream ss;
                ss << "Processed bytes: " << processedBytes << " do not equal to buffer size: " << request.raw_input_contents(bufferId).size() << "; input name: " << getCurrentlyValidatedTensorName();
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
                return Status(StatusCode::INVALID_CONTENT_SIZE, details);
            }
            if (batchSize != expectedValueCount) {
                std::stringstream ss;
                ss << "Expected: " << expectedValueCount << " values; Actual: " << batchSize << " values; input name: " << getCurrentlyValidatedTensorName();
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid value count of tensor proto - {}", servableName, servableVersion, details);
                return Status(StatusCode::INVALID_VALUE_COUNT, details);
            }
        } else {
            // Plain old data
            size_t expectedContentSize = expectedValueCount * ov::element::Type(ovmsPrecisionToIE2Precision(expectedPrecision)).size();
            if (expectedContentSize != request.raw_input_contents()[bufferId].size()) {
                std::stringstream ss;
                ss << "Expected: " << expectedContentSize << " bytes; Actual: " << request.raw_input_contents()[bufferId].size() << " bytes; input name: " << getCurrentlyValidatedTensorName();
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", servableName, servableVersion, details);
                return Status(StatusCode::INVALID_CONTENT_SIZE, details);
            }
        }
    } else {  // buffers placed in InputTensor content
        // here we should check that the elements count is equal since for some precisions there is padding
        // we need to decide first which exact datatype_contents we extract that information from
        size_t elementsCount = getElementsCount(proto, expectedPrecision);
        if (expectedValueCount != elementsCount) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << " values; Actual: " << elementsCount << " values; input name: " << getCurrentlyValidatedTensorName();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid value count of tensor proto - {}", servableName, servableVersion, details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validateNumberOfShapeDimensions(const ovms::TensorInfo& tensorInfo, const KFSTensorInputProto& proto) const {
    // Network and request must have the same number of shape dimensions
    const auto& shape = tensorInfo.getShape();
    if (proto.shape().size() < 0 ||
        shape.size() != static_cast<size_t>(proto.shape().size())) {
        std::stringstream ss;
        ss << "Expected: " << shape.toString()
           << "; Actual: " << tensorShapeToString(proto.shape())
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of shape dimensions - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}
template <>
Status RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>::validatePrecision(const ovms::TensorInfo& tensorInfo, const KFSTensorInputProto& proto) const {
    if (proto.datatype() != ovmsPrecisionToKFSPrecision(tensorInfo.getPrecision())) {
        std::stringstream ss;
        ss << "Expected: " << tensorInfo.getPrecisionAsString()
           << "; Actual: " << proto.datatype()
           << "; input name: " << getCurrentlyValidatedTensorName();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid precision - {}", servableName, servableVersion, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}
template <>
bool dataInRawInputContents(const KFSRequest& request) {
    return request.raw_input_contents().size() > 0;
}
template <>
const std::string* getRawInputContents(const KFSRequest& request, size_t bufferId) {
    return &(request.raw_input_contents().at(bufferId));
}

#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }

template <>
Status validate(const KFSRequest& request, const tensor_map_t& inputsInfo, const tensor_map_t& outputsInfo, const std::string& servableName, const model_version_t servableVersion, const std::set<std::string>& optionalAllowedInputNames, const Mode batchingMode, const shapes_info_map_t& shapeInfo) {
    OVMS_PROFILE_FUNCTION();
    return RequestValidator<KFSRequest, KFSTensorInputProto, ValidationChoice::INPUT, KFSInputTensorIteratorType, KFSShapeType>(request, inputsInfo, outputsInfo, servableName, servableVersion, optionalAllowedInputNames, batchingMode, shapeInfo).validate();
}
}  // namespace request_validation_utils
/*Status validateTensor(const TensorInfo& tensorInfo,
    const ::KFSRequest::InferInputTensor& src,
    const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    bool rawInputsContentsUsed = (buffer != nullptr);
    auto status = tensor_conversion::validateLayout(tensorInfo);
    if (!status.ok()) {
        return status;
    }
    // 4 for default pipelines, 5 for pipelines with demultiplication at entry
    bool isShapeLengthValid = tensorInfo.getShape().size() == 4 ||
                              (tensorInfo.isInfluencedByDemultiplexer() && tensorInfo.getShape().size() == 5);
    if (!isShapeLengthValid) {
        return StatusCode::INVALID_SHAPE;
    }

    size_t batchSize = !rawInputsContentsUsed ? src.contents().bytes_contents_size() : getNumberOfInputs(buffer);
    if (tensor_conversion::checkBatchSizeMismatch(tensorInfo, batchSize)) {
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}",
            tensorInfo.getMappedName(),
            tensorInfo.getBatchSize().has_value() ? tensorInfo.getBatchSize().value().toString() : std::string{"none"},
            src.contents().bytes_contents_size());
        return StatusCode::INVALID_BATCH_SIZE;
    }

    if (!rawInputsContentsUsed) {
        for (size_t i = 0; i < batchSize; i++) {
            if (src.contents().bytes_contents(i).size() <= 0) {
                SPDLOG_DEBUG("Tensor: {} {}th image of the batch is empty.", src.name(), i);
                return StatusCode::BYTES_CONTENTS_EMPTY;
            }
        }
    } else {
        if (buffer->size() <= 0) {
            SPDLOG_DEBUG("Tensor: {} raw_inputs_contents is empty", src.name());
            return StatusCode::BYTES_CONTENTS_EMPTY;
        }
    }

    return StatusCode::OK;
}*/
}  // namespace ovms
