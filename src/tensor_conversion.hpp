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
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

#include "image_utils/kserve_image_conversion.hpp"
#include "logging.hpp"
#include "precision.hpp"
#include "predict_request_validation_utils_impl.hpp"
#include "profiler.hpp"
#include "tensorinfo.hpp"
#include "status.hpp"
#include "tensor_conversion_common.hpp"

namespace inference {
class ModelInferRequest_InferInputTensor;
}
namespace ovms {
class Status;
template <typename TensorType>
Status convertStringRequestToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);

template <typename TensorType>
Status convertStringRequestToOVTensor2D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst);

template <typename TensorType>
static Status convertTensorToImagesMatchingTensorInfo(const TensorType& src, std::vector<ov::Tensor>& images, const TensorInfo& tensorInfo, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    Dimension targetHeight = tensor_conversion::getTensorInfoHeightDim(tensorInfo);
    Dimension targetWidth = tensor_conversion::getTensorInfoWidthDim(tensorInfo);

    // Enforce resolution alignment against first image in the batch if resize is not supported.
    bool resizeSupported = tensor_conversion::isResizeSupported(tensorInfo);
    bool enforceResolutionAlignment = !resizeSupported;

    bool rawInputsContentsUsed = (buffer != nullptr);
    std::vector<std::string> inputs;
    auto status = tensor_conversion::getInputs(buffer, inputs);
    if (status != StatusCode::OK) {
        return status;
    }
    int numberOfInputs = (!rawInputsContentsUsed ? getBinaryInputsSize(src) : inputs.size());
    size_t totalAllocatedPixels = 0;
    size_t maxAllowedImagePixels = request_validation_utils::getMaxImageDecodePixels();
    for (int i = 0; i < numberOfInputs; i++) {
        const std::string& encodedImage = !rawInputsContentsUsed ? getBinaryInput(src, i) : inputs[i];
        status = tensor_conversion::checkEstimatedImageSize(encodedImage, tensorInfo.getMappedName(), totalAllocatedPixels, maxAllowedImagePixels);
        if (status != StatusCode::OK) {
            return status;
        }
        ov::Tensor image;
        status = image_utils::decodeImage(encodedImage, image);
        if (!status.ok())
            return status;

        status = tensor_conversion::accumulateAndCheckDecodedImageSize(image, tensorInfo, totalAllocatedPixels, maxAllowedImagePixels);
        if (status != StatusCode::OK) {
            return status;
        }
        const ov::Tensor* firstImage = images.empty() ? nullptr : &images.at(0);
        status = tensor_conversion::validateInput(tensorInfo, image, firstImage, enforceResolutionAlignment);
        if (status != StatusCode::OK) {
            return status;
        }
        if (i == 0) {
            tensor_conversion::updateTargetResolution(targetHeight, targetWidth, image);
        }

        if (!tensor_conversion::isSupportedImagePrecision(tensorInfo.getPrecision()))
            return StatusCode::INVALID_PRECISION;
        if (!targetHeight.isStatic() || !targetWidth.isStatic()) {
            return StatusCode::INTERNAL_ERROR;
        }
        const size_t targetHeightValue = static_cast<size_t>(targetHeight.getStaticValue());
        const size_t targetWidthValue = static_cast<size_t>(targetWidth.getStaticValue());
        if (!resizeSupported && (image.get_shape()[1] != targetHeightValue || image.get_shape()[2] != targetWidthValue)) {
            return StatusCode::INVALID_SHAPE;
        }
        ov::Tensor processed;
        status = image_utils::preprocessImage(image, processed, tensorInfo.getOvPrecision(),
            targetHeightValue, targetWidthValue);
        if (!status.ok())
            return status;
        image = std::move(processed);

        if (firstImage != nullptr && image.get_shape()[3] != firstImage->get_shape()[3]) {
            SPDLOG_DEBUG("Binary data sent to input: {} has mixed channel count within one batch. First image channels: {} current image channels: {}",
                tensorInfo.getMappedName(),
                firstImage->get_shape()[3],
                image.get_shape()[3]);
            return StatusCode::INVALID_NO_OF_CHANNELS;
        }

        images.push_back(std::move(image));
    }
    return StatusCode::OK;
}

template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    auto status = validateTensor(tensorInfo, src, buffer);
    if (status != StatusCode::OK) {
        SPDLOG_DEBUG("Input native file format validation failed");
        return status;
    }
    std::vector<ov::Tensor> images;
    status = convertTensorToImagesMatchingTensorInfo(src, images, tensorInfo, buffer);
    if (!status.ok()) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return status;
    }
    tensor = tensor_conversion::createTensorFromImages(images, tensorInfo);
    if (!tensor) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}

class InferenceTensor;

template <>
Status convertNativeFileFormatRequestTensorToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);

template <>
Status convertStringRequestToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const std::string* buffer);

template <>
Status convertStringRequestToOVTensor(const ::inference::ModelInferRequest_InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst) {
    if (tensor.get_shape().size() != 2) {
        return StatusCode::INTERNAL_ERROR;
    }
    if (tensor.get_element_type() != ov::element::Type_t::u8) {
        return StatusCode::INTERNAL_ERROR;
    }
    size_t batchSize = tensor.get_shape()[0];
    size_t maxStringLen = tensor.get_shape()[1];
    setBatchSize(dst, batchSize);
    setStringPrecision(dst);
    for (size_t i = 0; i < batchSize; i++) {
        const char* strStart = reinterpret_cast<const char*>(tensor.data<unsigned char>() + i * maxStringLen);
        size_t strLen = strnlen(strStart, maxStringLen);
        createOrGetString(dst, i).assign(strStart, strLen);
    }
    return StatusCode::OK;
}
template <>
Status convertStringRequestToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const std::string* buffer);
template <>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, InferenceTensor& dst);
}  // namespace ovms
