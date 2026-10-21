//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "tensor_conversion_common.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "image_utils/decoded_image_size.hpp"
#include "image_utils/kserve_image_conversion.hpp"
#include "logging.hpp"
#include "precision.hpp"
#include "predict_request_validation_utils_impl.hpp"
#include "profiler.hpp"
#include "tensorinfo.hpp"
#include "status.hpp"
#include "inference_request_common.hpp"

namespace ovms {
class Status;

namespace tensor_conversion {
Status validateNumberOfChannels(const TensorInfo& tensorInfo,
    const ov::Tensor& input,
    const ov::Tensor* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();

    // At this point we can either have nhwc format or pretendant to be nhwc but with ANY layout in pipeline info
    Dimension numberOfChannels;
    if (tensorInfo.getShape().size() == 4) {
        numberOfChannels = tensorInfo.getShape()[3];
    } else if (tensorInfo.isInfluencedByDemultiplexer() && tensorInfo.getShape().size() == 5) {
        numberOfChannels = tensorInfo.getShape()[4];
    } else {
        return StatusCode::INVALID_NO_OF_CHANNELS;
    }
    if (numberOfChannels.isAny() && firstBatchImage) {
        numberOfChannels = firstBatchImage->get_shape()[3];
    }
    if (numberOfChannels.isAny()) {
        return StatusCode::OK;
    }
    if (!numberOfChannels.match(input.get_shape()[3])) {
        SPDLOG_DEBUG("Binary data sent to input: {} has invalid number of channels. Expected: {} Actual: {}",
            tensorInfo.getMappedName(),
            numberOfChannels.toString(),
            input.get_shape()[3]);
        return StatusCode::INVALID_NO_OF_CHANNELS;
    }
    return StatusCode::OK;
}
Status validateLayout(const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    static const std::string binarySupportedLayout = "N...HWC";
    if (!tensorInfo.getLayout().createIntersection(Layout(binarySupportedLayout), tensorInfo.getShape().size()).has_value()) {
        SPDLOG_DEBUG("Endpoint needs to be compatible with {} to support binary image inputs, actual: {}",
            binarySupportedLayout,
            tensorInfo.getLayout());
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    return StatusCode::OK;
}
int getNumberOfInputs(const std::string* buffer) {
    int32_t batchSize;
    size_t width;
    auto status = request_validation_utils::getRawInputContentsBatchSizeAndWidth(*buffer, batchSize, width);
    if (!status.ok())
        return 0;
    return batchSize;
}
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize) {
    OVMS_PROFILE_FUNCTION();
    if (!tensorInfo.getBatchSize().has_value() || batchSize == 0) {
        return true;
    }
    return !tensorInfo.getBatchSize().value().match(batchSize);
}
bool isSupportedImagePrecision(ovms::Precision tensorPrecision) {
    switch (tensorPrecision) {
    case ovms::Precision::FP32:
    case ovms::Precision::FP64:
    case ovms::Precision::FP16:
    case ovms::Precision::I16:
    case ovms::Precision::U8:
    case ovms::Precision::I8:
    case ovms::Precision::U16:
    case ovms::Precision::I32:
        return true;
    default:
        return false;
    }
}
Status validateInput(const TensorInfo& tensorInfo, const ov::Tensor& input, const ov::Tensor* firstBatchImage, bool enforceResolutionAlignment) {
    // Binary inputs are supported for any endpoint that is compatible with N...HWC layout.
    // With unknown layout, there is no way to deduce expected endpoint input resolution.
    // This forces binary utility to create tensors with resolution inherited from first batch of binary input image (request).
    // In case of any dimension in endpoint shape is dynamic, we need to validate images against first image resolution.
    // Otherwise we can omit that, and proceed to image resize.
    OVMS_PROFILE_FUNCTION();
    if (firstBatchImage && enforceResolutionAlignment) {
        auto status = validateResolutionAgainstFirstBatchImage(input, firstBatchImage);
        if (!status.ok()) {
            return status;
        }
    }
    return validateNumberOfChannels(tensorInfo, input, firstBatchImage);
}
Dimension getTensorInfoHeightDim(const TensorInfo& tensorInfo) {
    size_t numberOfShapeDimensions = tensorInfo.getShape().size();
    if (numberOfShapeDimensions < 4 || numberOfShapeDimensions > 5) {
        throw std::logic_error("wrong number of shape dimensions");
    }
    size_t position = numberOfShapeDimensions == 4 ? /*NHWC*/ 1 : /*N?HWC*/ 2;
    return tensorInfo.getShape()[position];
}
Dimension getTensorInfoWidthDim(const TensorInfo& tensorInfo) {
    size_t numberOfShapeDimensions = tensorInfo.getShape().size();
    if (numberOfShapeDimensions < 4 || numberOfShapeDimensions > 5) {
        throw std::logic_error("wrong number of shape dimensions");
    }
    size_t position = numberOfShapeDimensions == 4 ? /*NHWC*/ 2 : /*N?HWC*/ 3;
    return tensorInfo.getShape()[position];
}
void updateTargetResolution(Dimension& height, Dimension& width, const ov::Tensor& image) {
    const auto& shape = image.get_shape();
    if (height.isAny()) {
        height = shape[1];
    } else if (height.isDynamic()) {
        if (height.match(shape[1])) {
            height = shape[1];
        } else {
            if (shape[1] > height.getMaxValue()) {
                height = height.getMaxValue();
            } else {
                height = height.getMinValue();
            }
        }
    }
    if (width.isAny()) {
        width = shape[2];
    } else if (width.isDynamic()) {
        if (width.match(shape[2])) {
            width = shape[2];
        } else {
            if (shape[2] > width.getMaxValue()) {
                width = width.getMaxValue();
            } else {
                width = width.getMinValue();
            }
        }
    }
}
bool isResizeSupported(const TensorInfo& tensorInfo) {
    for (const auto& dim : tensorInfo.getShape()) {
        if (dim.isAny()) {
            return false;
        }
    }
    if (tensorInfo.getLayout() != "NHWC" &&
        tensorInfo.getLayout() != "N?HWC" &&
        tensorInfo.getLayout() != Layout::getUnspecifiedLayout()) {
        return false;
    }
    return true;
}
Status checkEstimatedImageSize(std::string_view encodedImage, const std::string& inputName,
    size_t alreadyAllocatedPixels, size_t maxAllowedImagePixels) {
    uint64_t estimatedDecodedPixels = 0;
    auto estimate = image_utils::estimateDecodedImageSize(encodedImage, estimatedDecodedPixels);
    if (estimate == image_utils::DecodedSizeEstimate::InputTooLarge) {
        SPDLOG_DEBUG("Image binary payload too large to inspect for input: {}. Size: {}", inputName, encodedImage.size());
        return StatusCode::INVALID_IMAGE_MAX_SIZE_EXCEEDED;
    }
    if (estimate == image_utils::DecodedSizeEstimate::UnsupportedFormat) {
        if (!request_validation_utils::allowUnestimatableImageFormats()) {
            SPDLOG_DEBUG("Decoded size could not be estimated for input: {} and unestimatable formats are not allowed", inputName);
            std::string detail = "Decoded image size could not be estimated; unestimatable formats are disabled";
            const auto& dependencyError = image_utils::openCvImageDecoderUnavailableReason();
            if (!dependencyError.empty())
                detail += ". " + dependencyError;
            return Status(StatusCode::IMAGE_PARSING_FAILED, detail);
        }
        return StatusCode::OK;
    }
    size_t remainingBudget = alreadyAllocatedPixels >= maxAllowedImagePixels ? 0 : maxAllowedImagePixels - alreadyAllocatedPixels;
    if (estimatedDecodedPixels > remainingBudget) {
        SPDLOG_DEBUG("Estimated decoded image pixels exceed budget for input: {}. Estimated pixels: {}, max allowed: {}",
            inputName, estimatedDecodedPixels, maxAllowedImagePixels);
        return StatusCode::INVALID_IMAGE_MAX_SIZE_EXCEEDED;
    }
    return StatusCode::OK;
}
Status accumulateAndCheckDecodedImageSize(const ov::Tensor& image, const TensorInfo& tensorInfo,
    size_t& totalAllocatedPixels, size_t maxAllowedImagePixels) {
    size_t imagePixels = image.get_shape()[1] * image.get_shape()[2];
    size_t remainingBudget = totalAllocatedPixels >= maxAllowedImagePixels ? 0 : maxAllowedImagePixels - totalAllocatedPixels;
    if (imagePixels > remainingBudget) {
        SPDLOG_DEBUG("Decoded image pixel budget exceeded for input: {}. Image pixels: {}, max allowed: {}",
            tensorInfo.getMappedName(), imagePixels, maxAllowedImagePixels);
        return StatusCode::INVALID_IMAGE_MAX_SIZE_EXCEEDED;
    }
    totalAllocatedPixels += imagePixels;
    return StatusCode::OK;
}
}  // namespace tensor_conversion

}  // namespace ovms
