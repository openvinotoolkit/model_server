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

#include <memory>
#include <vector>
#include <string>

#include "deps/opencv_core_inc.hpp"
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
    const cv::Mat input,
    cv::Mat* firstBatchImage) {
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
        numberOfChannels = firstBatchImage->channels();
    }
    if (numberOfChannels.isAny()) {
        return StatusCode::OK;
    }
    if (!numberOfChannels.match(input.channels())) {
        SPDLOG_DEBUG("Binary data sent to input: {} has invalid number of channels. Expected: {} Actual: {}",
            tensorInfo.getMappedName(),
            numberOfChannels.toString(),
            input.channels());
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
int getMatTypeFromTensorPrecision(ovms::Precision tensorPrecision) {
    switch (tensorPrecision) {
    case ovms::Precision::FP32:
        return CV_32F;
    case ovms::Precision::FP64:
        return CV_64F;
    case ovms::Precision::FP16:
        return CV_16F;
    case ovms::Precision::I16:
        return CV_16S;
    case ovms::Precision::U8:
        return CV_8U;
    case ovms::Precision::I8:
        return CV_8S;
    case ovms::Precision::U16:
        return CV_16U;
    case ovms::Precision::I32:
        return CV_32S;
    default:
        return -1;
    }
}
bool isPrecisionEqual(int matPrecision, ovms::Precision tensorPrecision) {
    int convertedTensorPrecision = getMatTypeFromTensorPrecision(tensorPrecision);
    if (convertedTensorPrecision == matPrecision) {
        return true;
    }
    return false;
}
cv::Mat convertStringToMat(const std::string& image) {
    OVMS_PROFILE_FUNCTION();
    std::vector<unsigned char> data(image.begin(), image.end());
    cv::Mat dataMat(data, true);

    try {
        return cv::imdecode(dataMat, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        SPDLOG_DEBUG("Error during string_val to mat conversion: {}", e.what());
        return cv::Mat{};
    }
}

Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision) {
    OVMS_PROFILE_FUNCTION();
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        SPDLOG_DEBUG("Error during binary input conversion: not supported precision: {}", toString(requestedPrecision));
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}
bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width) {
    if (height != image.rows || width != image.cols) {
        return true;
    }
    return false;
}
Status validateInput(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment) {
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
void updateTargetResolution(Dimension& height, Dimension& width, const cv::Mat& image) {
    if (height.isAny()) {
        height = image.rows;
    } else if (height.isDynamic()) {
        if (height.match(image.rows)) {
            height = image.rows;
        } else {
            if (image.rows > height.getMaxValue()) {
                height = height.getMaxValue();
            } else {
                height = height.getMinValue();
            }
        }
    }
    if (width.isAny()) {
        width = image.cols;
    } else if (width.isDynamic()) {
        if (width.match(image.cols)) {
            width = image.cols;
        } else {
            if (image.cols > width.getMaxValue()) {
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
/////////////////////////////////
}  // namespace tensor_conversion
/////////////////////////////////

shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    shape_t dims;
    dims.push_back(images.size());
    if (tensorInfo.isInfluencedByDemultiplexer()) {
        dims.push_back(1);
    }
    dims.push_back(images[0].rows);
    dims.push_back(images[0].cols);
    dims.push_back(images[0].channels());
    return dims;
}
ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    ov::Shape shape = getShapeFromImages(images, tensorInfo);
    ov::element::Type precision = tensorInfo.getOvPrecision();
    ov::Tensor tensor(precision, shape);
    char* ptr = (char*)tensor.data();
    for (cv::Mat image : images) {
        memcpy(ptr, (char*)image.data, image.total() * image.elemSize());
        ptr += (image.total() * image.elemSize());
    }
    return tensor;
}
ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    switch (tensorInfo.getPrecision()) {
    case ovms::Precision::FP32:
    case ovms::Precision::I32:
    case ovms::Precision::FP64:
    case ovms::Precision::I8:
    case ovms::Precision::U8:
    case ovms::Precision::FP16:
    case ovms::Precision::U16:
    case ovms::Precision::I16:
        return createTensorFromMats(images, tensorInfo);
    case ovms::Precision::MIXED:
    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
    case ovms::Precision::BOOL:
    case ovms::Precision::CUSTOM:
    default:
        return ov::Tensor();
    }
}

}  // namespace ovms
