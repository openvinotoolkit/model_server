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
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "precision.hpp"
#include "predict_request_validation_utils_impl.hpp"
#include "profiler.hpp"
#include "tensorinfo.hpp"
#include "status.hpp"

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


namespace tensor_conversion {
int getMatTypeFromTensorPrecision(ovms::Precision tensorPrecision);
bool isPrecisionEqual(int matPrecision, ovms::Precision tensorPrecision);
Status validateInput(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment);
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize);
Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage);
Status validateNumberOfChannels(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage);
Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width);
bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width);
Status validateLayout(const TensorInfo& tensorInfo);
Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision);
cv::Mat convertStringToMat(const std::string& image);
int getNumberOfInputs(const std::string* buffer);
Dimension getTensorInfoHeightDim(const TensorInfo& tensorInfo);
Dimension getTensorInfoWidthDim(const TensorInfo& tensorInfo);
void updateTargetResolution(Dimension& height, Dimension& width, const cv::Mat& image);
bool isResizeSupported(const TensorInfo& tensorInfo);
Status getInputs(const std::string* buffer, std::vector<std::string>& inputs);
ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
}  // namespace tensor_conversion
template <typename TensorType>
static Status convertTensorToMatsMatchingTensorInfo(const TensorType& src, std::vector<cv::Mat>& images, const TensorInfo& tensorInfo, const std::string* buffer) {
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
    for (int i = 0; i < numberOfInputs; i++) {
        cv::Mat image = tensor_conversion::convertStringToMat(!rawInputsContentsUsed ? getBinaryInput(src, i) : inputs[i]);
        if (image.data == nullptr)
            return StatusCode::IMAGE_PARSING_FAILED;
        cv::Mat* firstImage = images.size() == 0 ? nullptr : &images.at(0);
        auto status = tensor_conversion::validateInput(tensorInfo, image, firstImage, enforceResolutionAlignment);
        if (status != StatusCode::OK) {
            return status;
        }
        if (i == 0) {
            tensor_conversion::updateTargetResolution(targetHeight, targetWidth, image);
        }

        if (!tensor_conversion::isPrecisionEqual(image.depth(), tensorInfo.getPrecision())) {
            cv::Mat imageCorrectPrecision;
            status = tensor_conversion::convertPrecision(image, imageCorrectPrecision, tensorInfo.getPrecision());

            if (status != StatusCode::OK) {
                return status;
            }
            image = std::move(imageCorrectPrecision);
        }
        if (!targetHeight.isStatic() || !targetWidth.isStatic()) {
            return StatusCode::INTERNAL_ERROR;
        }
        if (tensor_conversion::resizeNeeded(image, targetHeight.getStaticValue(), targetWidth.getStaticValue())) {
            if (!resizeSupported) {
                return StatusCode::INVALID_SHAPE;
            }
            cv::Mat imageResized;
            status = tensor_conversion::resizeMat(image, imageResized, targetHeight.getStaticValue(), targetWidth.getStaticValue());
            if (!status.ok()) {
                return status;
            }
            image = std::move(imageResized);
        }

        // if (i == 0 && src.contents().bytes_contents_size() > 1) {
        //     // Multiply src.string_val_size() * image resolution * precision size
        // }
        images.push_back(image);
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
    std::vector<cv::Mat> images;
    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo, buffer);
    if (!status.ok()) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return status;
    }
    tensor = tensor_conversion::convertMatsToTensor(images, tensorInfo);
    if (!tensor) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}

template <typename TensorType>
Status convertStringRequestToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    if (buffer != nullptr) {
        return convertBinaryExtensionStringFromBufferToNativeOVTensor(src, tensor, buffer);
    }
    int batchSize = getBinaryInputsSize(src);
    tensor = ov::Tensor(ov::element::Type_t::string, ov::Shape{static_cast<size_t>(batchSize)});
    std::string* data = tensor.data<std::string>();
    for (int i = 0; i < batchSize; i++) {
        data[i].assign(getBinaryInput(src, i));
    }
    return StatusCode::OK;
}

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
}  // namespace ovms
