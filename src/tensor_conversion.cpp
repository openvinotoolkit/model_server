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
#include "tensor_conversion.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include "image_utils/kserve_image_conversion.hpp"
#include "tensor_conversion_common.hpp"
#include "predict_request_validation_utils_impl.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"

namespace ovms {
namespace tensor_conversion {
Status convertEncodedImagesToOVTensor(const std::vector<std::string_view>& encodedImages, ov::Tensor& tensor,
    const TensorInfo& tensorInfo, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    Dimension targetHeight = getTensorInfoHeightDim(tensorInfo);
    Dimension targetWidth = getTensorInfoWidthDim(tensorInfo);
    const bool resizeSupported = isResizeSupported(tensorInfo);
    const bool enforceResolutionAlignment = !resizeSupported;
    std::vector<std::string_view> inputs;
    auto status = getInputs(buffer, inputs);
    if (status != StatusCode::OK) {
        return status;
    }
    if (buffer == nullptr) {
        inputs = encodedImages;
    }
    std::vector<ov::Tensor> images;
    size_t totalAllocatedPixels = 0;
    const size_t maxAllowedImagePixels = request_validation_utils::getMaxImageDecodePixels();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::string_view encodedImage = inputs[i];
        status = checkEstimatedImageSize(encodedImage, tensorInfo.getMappedName(), totalAllocatedPixels, maxAllowedImagePixels);
        if (status != StatusCode::OK) {
            return status;
        }
        ov::Tensor image;
        status = image_utils::decodeImage(encodedImage, image);
        if (!status.ok()) {
            return status;
        }
        status = accumulateAndCheckDecodedImageSize(image, tensorInfo, totalAllocatedPixels, maxAllowedImagePixels);
        if (status != StatusCode::OK) {
            return status;
        }
        const ov::Tensor* firstImage = images.empty() ? nullptr : &images.front();
        status = validateInput(tensorInfo, image, firstImage, enforceResolutionAlignment);
        if (status != StatusCode::OK) {
            return status;
        }
        if (i == 0) {
            updateTargetResolution(targetHeight, targetWidth, image);
        }
        if (!isSupportedImagePrecision(tensorInfo.getPrecision())) {
            return StatusCode::INVALID_PRECISION;
        }
        if (!targetHeight.isStatic() || !targetWidth.isStatic()) {
            return StatusCode::INTERNAL_ERROR;
        }
        const size_t targetHeightValue = static_cast<size_t>(targetHeight.getStaticValue());
        const size_t targetWidthValue = static_cast<size_t>(targetWidth.getStaticValue());
        if (!resizeSupported && (image.get_shape()[1] != targetHeightValue || image.get_shape()[2] != targetWidthValue)) {
            return StatusCode::INVALID_SHAPE;
        }
        ov::Tensor processed;
        status = image_utils::preprocessImage(image, processed, tensorInfo.getOvPrecision(), targetHeightValue, targetWidthValue);
        if (!status.ok()) {
            return status;
        }
        image = std::move(processed);
        if (firstImage != nullptr && image.get_shape()[3] != firstImage->get_shape()[3]) {
            SPDLOG_DEBUG("Binary data sent to input: {} has mixed channel count within one batch. First image channels: {} current image channels: {}",
                tensorInfo.getMappedName(), firstImage->get_shape()[3], image.get_shape()[3]);
            return StatusCode::INVALID_NO_OF_CHANNELS;
        }
        images.push_back(std::move(image));
    }
    tensor = createTensorFromImages(images, tensorInfo);
    return tensor ? StatusCode::OK : StatusCode::IMAGE_PARSING_FAILED;
}
}  // namespace tensor_conversion
}  // namespace ovms
