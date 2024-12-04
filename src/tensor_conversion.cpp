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
#include "predict_request_validation_utils_impl.hpp"
#include "logging.hpp"
#include "opencv2/opencv.hpp"
#include "profiler.hpp"
#include "status.hpp"

namespace ovms {
namespace tensor_conversion {
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
bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width) {
    if (height != image.rows || width != image.cols) {
        return true;
    }
    return false;
}
Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width) {
    OVMS_PROFILE_FUNCTION();
    cv::resize(src, dst, cv::Size(width, height));
    return StatusCode::OK;
}
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
Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();
    if (input.cols == firstBatchImage->cols && input.rows == firstBatchImage->rows) {
        return StatusCode::OK;
    }
    SPDLOG_DEBUG("Each binary image in request needs to have resolution matched. First cols: {}, rows: {}, current cols: {}, rows: {}",
        firstBatchImage->cols, firstBatchImage->rows, input.cols, input.rows);
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize) {
    OVMS_PROFILE_FUNCTION();
    if (!tensorInfo.getBatchSize().has_value() || batchSize == 0) {
        return true;
    }
    return !tensorInfo.getBatchSize().value().match(batchSize);
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
int getNumberOfInputs(const std::string* buffer) {
    int32_t batchSize;
    size_t width;
    // TODO @atobisze common request utils?
    auto status = request_validation_utils::getRawInputContentsBatchSizeAndWidth(*buffer, batchSize, width);
    if (!status.ok())
        return 0;
    return batchSize;
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
Status getInputs(const std::string* buffer, std::vector<std::string>& inputs) {
    if (buffer == nullptr) {
        return StatusCode::OK;
    }
    size_t offset = 0;
    while (offset + sizeof(uint32_t) <= buffer->size()) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += sizeof(uint32_t);
        if (offset + inputSize > buffer->size())
            break;
        inputs.push_back(buffer->substr(offset, inputSize));
        offset += inputSize;
    }
    if (offset != buffer->size()) {
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}
}  // namespace tensor_conversion
using namespace tensor_conversion;
template <typename TensorType>
static Status convertTensorToMatsMatchingTensorInfo(const TensorType& src, std::vector<cv::Mat>& images, const TensorInfo& tensorInfo, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    Dimension targetHeight = getTensorInfoHeightDim(tensorInfo);
    Dimension targetWidth = getTensorInfoWidthDim(tensorInfo);

    // Enforce resolution alignment against first image in the batch if resize is not supported.
    bool resizeSupported = isResizeSupported(tensorInfo);
    bool enforceResolutionAlignment = !resizeSupported;

    bool rawInputsContentsUsed = (buffer != nullptr);
    std::vector<std::string> inputs;
    auto status = getInputs(buffer, inputs);
    if (status != StatusCode::OK) {
        return status;
    }
    int numberOfInputs = (!rawInputsContentsUsed ? getBinaryInputsSize(src) : inputs.size());
    for (int i = 0; i < numberOfInputs; i++) {
        cv::Mat image = convertStringToMat(!rawInputsContentsUsed ? getBinaryInput(src, i) : inputs[i]);
        if (image.data == nullptr)
            return StatusCode::IMAGE_PARSING_FAILED;
        cv::Mat* firstImage = images.size() == 0 ? nullptr : &images.at(0);
        auto status = validateInput(tensorInfo, image, firstImage, enforceResolutionAlignment);
        if (status != StatusCode::OK) {
            return status;
        }
        if (i == 0) {
            updateTargetResolution(targetHeight, targetWidth, image);
        }

        if (!isPrecisionEqual(image.depth(), tensorInfo.getPrecision())) {
            cv::Mat imageCorrectPrecision;
            status = convertPrecision(image, imageCorrectPrecision, tensorInfo.getPrecision());

            if (status != StatusCode::OK) {
                return status;
            }
            image = std::move(imageCorrectPrecision);
        }
        if (!targetHeight.isStatic() || !targetWidth.isStatic()) {
            return StatusCode::INTERNAL_ERROR;
        }
        if (resizeNeeded(image, targetHeight.getStaticValue(), targetWidth.getStaticValue())) {
            if (!resizeSupported) {
                return StatusCode::INVALID_SHAPE;
            }
            cv::Mat imageResized;
            status = resizeMat(image, imageResized, targetHeight.getStaticValue(), targetWidth.getStaticValue());
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

static shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
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

static ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
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

static ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
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
    tensor = convertMatsToTensor(images, tensorInfo);
    if (!tensor) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}
template <typename TensorType>
Status convertStringRequestToOVTensor2D(
    const TensorType& src,
    ov::Tensor& tensor,
    const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    if (buffer != nullptr) {
        return convertStringRequestFromBufferToOVTensor2D(src, tensor, buffer);
    }
    int batchSize = getBinaryInputsSize(src);
    size_t maxStringLength = 0;
    for (int i = 0; i < batchSize; i++) {
        maxStringLength = std::max(maxStringLength, getBinaryInput(src, i).size());
    }
    size_t width = maxStringLength + 1;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{static_cast<size_t>(batchSize), width});
    for (int i = 0; i < batchSize; i++) {
        std::memcpy(
            tensor.data<unsigned char>() + i * width,
            reinterpret_cast<const unsigned char*>(getBinaryInput(src, i).c_str()),
            getBinaryInput(src, i).size());
        for (size_t j = getBinaryInput(src, i).size(); j < width; j++) {
            tensor.data<unsigned char>()[i * width + j] = 0;
        }
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
