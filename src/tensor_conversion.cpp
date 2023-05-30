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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "kfs_frontend/kfs_utils.hpp"
#include "logging.hpp"
#include "opencv2/opencv.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tfs_frontend/tfs_utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#pragma GCC diagnostic pop

namespace ovms {

static int getMatTypeFromTensorPrecision(ovms::Precision tensorPrecision) {
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

static bool isPrecisionEqual(int matPrecision, ovms::Precision tensorPrecision) {
    int convertedTensorPrecision = getMatTypeFromTensorPrecision(tensorPrecision);
    if (convertedTensorPrecision == matPrecision) {
        return true;
    }
    return false;
}

static cv::Mat convertStringToMat(const std::string& image) {
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

static Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision) {
    OVMS_PROFILE_FUNCTION();
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        SPDLOG_DEBUG("Error during binary input conversion: not supported precision: {}", toString(requestedPrecision));
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}

static Status validateLayout(const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    static const std::string binarySupportedLayout = "N...HWC";
    if (!tensorInfo->getLayout().createIntersection(Layout(binarySupportedLayout), tensorInfo->getShape().size()).has_value()) {
        SPDLOG_DEBUG("Endpoint needs to be compatible with {} to support binary image inputs, actual: {}",
            binarySupportedLayout,
            tensorInfo->getLayout());
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    return StatusCode::OK;
}

static bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width) {
    if (height != image.rows || width != image.cols) {
        return true;
    }
    return false;
}

static Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width) {
    OVMS_PROFILE_FUNCTION();
    cv::resize(src, dst, cv::Size(width, height));
    return StatusCode::OK;
}

static Status validateNumberOfChannels(const std::shared_ptr<const TensorInfo>& tensorInfo,
    const cv::Mat input,
    cv::Mat* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();

    // At this point we can either have nhwc format or pretendant to be nhwc but with ANY layout in pipeline info
    Dimension numberOfChannels;
    if (tensorInfo->getShape().size() == 4) {
        numberOfChannels = tensorInfo->getShape()[3];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getShape().size() == 5) {
        numberOfChannels = tensorInfo->getShape()[4];
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
            tensorInfo->getMappedName(),
            numberOfChannels.toString(),
            input.channels());
        return StatusCode::INVALID_NO_OF_CHANNELS;
    }

    return StatusCode::OK;
}

static Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();
    if (input.cols == firstBatchImage->cols && input.rows == firstBatchImage->rows) {
        return StatusCode::OK;
    }
    SPDLOG_DEBUG("Each binary image in request needs to have resolution matched. First cols: {}, rows: {}, current cols: {}, rows: {}",
        firstBatchImage->cols, firstBatchImage->rows, input.cols, input.rows);
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}

static bool checkBatchSizeMismatch(const std::shared_ptr<const TensorInfo>& tensorInfo,
    const int batchSize) {
    OVMS_PROFILE_FUNCTION();
    if (!tensorInfo->getBatchSize().has_value() || batchSize == 0) {
        return true;
    }
    return !tensorInfo->getBatchSize().value().match(batchSize);
}

static Status validateInput(const std::shared_ptr<const TensorInfo>& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment) {
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

static Status validateTensor(const std::shared_ptr<const TensorInfo>& tensorInfo,
    const tensorflow::TensorProto& src,
    const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    auto status = validateLayout(tensorInfo);
    if (!status.ok()) {
        return status;
    }
    // 4 for default pipelines, 5 for pipelines with demultiplication at entry
    bool isShapeLengthValid = tensorInfo->getShape().size() == 4 ||
                              (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getShape().size() == 5);
    if (!isShapeLengthValid) {
        return StatusCode::INVALID_SHAPE;
    }

    if (checkBatchSizeMismatch(tensorInfo, src.string_val_size())) {
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}",
            tensorInfo->getMappedName(),
            tensorInfo->getBatchSize().has_value() ? tensorInfo->getBatchSize().value().toString() : std::string{"none"},
            src.string_val_size());
        return StatusCode::INVALID_BATCH_SIZE;
    }

    for (int i = 0; i < src.string_val_size(); i++) {
        if (src.string_val(i).size() <= 0) {
            return StatusCode::STRING_VAL_EMPTY;
        }
    }
    return StatusCode::OK;
}

inline static int getNumberOfInputs(const std::string* buffer) {
    int32_t batchSize;
    size_t width;
    auto status = getRawInputContentsBatchSizeAndWidth(*buffer, batchSize, width);
    if (!status.ok())
        return 0;
    return batchSize;
}

static Status validateTensor(const std::shared_ptr<const TensorInfo>& tensorInfo,
    const ::KFSRequest::InferInputTensor& src,
    const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    bool rawInputsContentsUsed = (buffer != nullptr);
    auto status = validateLayout(tensorInfo);
    if (!status.ok()) {
        return status;
    }
    // 4 for default pipelines, 5 for pipelines with demultiplication at entry
    bool isShapeLengthValid = tensorInfo->getShape().size() == 4 ||
                              (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getShape().size() == 5);
    if (!isShapeLengthValid) {
        return StatusCode::INVALID_SHAPE;
    }

    size_t batchSize = !rawInputsContentsUsed ? src.contents().bytes_contents_size() : getNumberOfInputs(buffer);
    if (checkBatchSizeMismatch(tensorInfo, batchSize)) {
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}",
            tensorInfo->getMappedName(),
            tensorInfo->getBatchSize().has_value() ? tensorInfo->getBatchSize().value().toString() : std::string{"none"},
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
}

static Dimension getTensorInfoHeightDim(const std::shared_ptr<const TensorInfo>& tensorInfo) {
    size_t numberOfShapeDimensions = tensorInfo->getShape().size();
    if (numberOfShapeDimensions < 4 || numberOfShapeDimensions > 5) {
        throw std::logic_error("wrong number of shape dimensions");
    }
    size_t position = numberOfShapeDimensions == 4 ? /*NHWC*/ 1 : /*N?HWC*/ 2;
    return tensorInfo->getShape()[position];
}

static Dimension getTensorInfoWidthDim(const std::shared_ptr<const TensorInfo>& tensorInfo) {
    size_t numberOfShapeDimensions = tensorInfo->getShape().size();
    if (numberOfShapeDimensions < 4 || numberOfShapeDimensions > 5) {
        throw std::logic_error("wrong number of shape dimensions");
    }
    size_t position = numberOfShapeDimensions == 4 ? /*NHWC*/ 2 : /*N?HWC*/ 3;
    return tensorInfo->getShape()[position];
}

static void updateTargetResolution(Dimension& height, Dimension& width, const cv::Mat& image) {
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

static bool isResizeSupported(const std::shared_ptr<const TensorInfo>& tensorInfo) {
    for (const auto& dim : tensorInfo->getShape()) {
        if (dim.isAny()) {
            return false;
        }
    }
    if (tensorInfo->getLayout() != "NHWC" &&
        tensorInfo->getLayout() != "N?HWC" &&
        tensorInfo->getLayout() != Layout::getUnspecifiedLayout()) {
        return false;
    }
    return true;
}

inline static const std::string& getBinaryInput(const tensorflow::TensorProto& tensor, size_t i) {
    return tensor.string_val(i);
}

inline static const std::string& getBinaryInput(const ::KFSRequest::InferInputTensor& tensor, size_t i) {
    return tensor.contents().bytes_contents(i);
}

inline static int getBinaryInputsSize(const tensorflow::TensorProto& tensor) {
    return tensor.string_val_size();
}

inline static int getBinaryInputsSize(const ::KFSRequest::InferInputTensor& tensor) {
    return tensor.contents().bytes_contents_size();
}

inline static Status getInputs(const std::string* buffer, std::vector<std::string>& inputs) {
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

template <typename TensorType>
static Status convertTensorToMatsMatchingTensorInfo(const TensorType& src, std::vector<cv::Mat>& images, const std::shared_ptr<const TensorInfo>& tensorInfo, const std::string* buffer) {
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

        if (!isPrecisionEqual(image.depth(), tensorInfo->getPrecision())) {
            cv::Mat imageCorrectPrecision;
            status = convertPrecision(image, imageCorrectPrecision, tensorInfo->getPrecision());

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

static shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    shape_t dims;
    dims.push_back(images.size());
    if (tensorInfo->isInfluencedByDemultiplexer()) {
        dims.push_back(1);
    }
    dims.push_back(images[0].rows);
    dims.push_back(images[0].cols);
    dims.push_back(images[0].channels());
    return dims;
}

static ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    ov::Shape shape = getShapeFromImages(images, tensorInfo);
    ov::element::Type precision = tensorInfo->getOvPrecision();
    ov::Tensor tensor(precision, shape);
    char* ptr = (char*)tensor.data();
    for (cv::Mat image : images) {
        memcpy(ptr, (char*)image.data, image.total() * image.elemSize());
        ptr += (image.total() * image.elemSize());
    }
    return tensor;
}

static ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    switch (tensorInfo->getPrecision()) {
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
static Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::shared_ptr<const TensorInfo>& tensorInfo, const std::string* buffer) {
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

static Status convertStringRequestFromBufferToOVTensor2D(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer) {
    return StatusCode::NOT_IMPLEMENTED;
}

static Status convertStringRequestFromBufferToOVTensor2D(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer) {
    size_t batchSize = 0;
    size_t offset = 0;
    size_t maxStringLength = 0;
    while (offset + sizeof(uint32_t) <= buffer->size()) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += (sizeof(uint32_t) + inputSize);
        maxStringLength = std::max(maxStringLength, inputSize);
        batchSize++;
    }
    if (offset != buffer->size()) {
        SPDLOG_DEBUG("Input string format conversion failed");
        return StatusCode::INVALID_STRING_INPUT;
    }
    size_t width = maxStringLength + 1;
    offset = 0;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{batchSize, width});
    for (size_t i = 0; i < batchSize; i++) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += sizeof(uint32_t);
        auto data = tensor.data<unsigned char>() + i * width;
        std::memcpy(data, reinterpret_cast<const unsigned char*>(buffer->data() + offset), inputSize);
        for (size_t j = inputSize; j < width; j++) {
            data[j] = 0;
        }
        offset += inputSize;
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

static Status convertStringRequestFromBufferToOVTensor1D(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer) {
    return StatusCode::NOT_IMPLEMENTED;
}

static Status convertStringRequestFromBufferToOVTensor1D(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer) {
    std::vector<uint32_t> stringSizes;
    uint32_t totalStringsLength = 0;
    while (totalStringsLength + stringSizes.size() * sizeof(uint32_t) + sizeof(uint32_t) <= buffer->size()) {
        uint32_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + totalStringsLength + stringSizes.size() * sizeof(uint32_t)));
        stringSizes.push_back(inputSize);
        totalStringsLength += inputSize;
    }
    size_t batchSize = stringSizes.size();
    if ((totalStringsLength + batchSize * sizeof(uint32_t)) != buffer->size()) {
        SPDLOG_DEBUG("Input string format conversion failed");
        return StatusCode::INVALID_STRING_INPUT;
    }
    size_t metadataLength = sizeof(uint32_t) * (batchSize + 2);
    size_t width = totalStringsLength + metadataLength;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{width});
    uint32_t* data = reinterpret_cast<uint32_t*>(tensor.data<uint8_t>());
    data[0] = static_cast<uint32_t>(batchSize);
    data[1] = 0;  // first string start offset
    unsigned char* condensedStringsStart = tensor.data<unsigned char>() + metadataLength;
    size_t tensorStringsOffset = 0;
    for (size_t i = 0; i < stringSizes.size(); i++) {
        data[i + 2] = data[i + 1] + stringSizes[i];
        std::memcpy(condensedStringsStart + tensorStringsOffset, reinterpret_cast<const unsigned char*>(buffer->data() + (i + 1) * sizeof(uint32_t) + tensorStringsOffset), stringSizes[i]);
        tensorStringsOffset += stringSizes[i];
    }
    return StatusCode::OK;
}

template <typename TensorType>
Status convertStringRequestToOVTensor1D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer) {
    if (buffer != nullptr) {
        return convertStringRequestFromBufferToOVTensor1D(src, tensor, buffer);
    }
    int batchSize = getBinaryInputsSize(src);
    size_t totalStringsLength = 0;
    for (int i = 0; i < batchSize; i++) {
        totalStringsLength += getBinaryInput(src, i).size();
    }
    // space for metadata:
    // - batch size (uint32_t) x 1
    // - first string start offset (uint32_t) x 1
    // - end offsets for each batch of string (uint32_t) x batchSize
    size_t metadataLength = sizeof(uint32_t) * (batchSize + 2);
    size_t width = totalStringsLength + metadataLength;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{static_cast<size_t>(width)});
    uint32_t* data = reinterpret_cast<uint32_t*>(tensor.data<uint8_t>());
    data[0] = static_cast<uint32_t>(batchSize);
    data[1] = 0;  // first string start offset
    unsigned char* condensedStringsStart = tensor.data<unsigned char>() + metadataLength;
    for (int i = 0; i < batchSize; i++) {
        // write end offset
        data[i + 2] = data[i + 1] + getBinaryInput(src, i).size();
        // write the bytes
        if (getBinaryInput(src, i).size()) {
            std::memcpy(
                condensedStringsStart + data[i + 1],
                reinterpret_cast<const unsigned char*>(getBinaryInput(src, i).c_str()),
                getBinaryInput(src, i).size());
        }
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

template Status convertNativeFileFormatRequestTensorToOVTensor<tensorflow::TensorProto>(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::shared_ptr<const TensorInfo>& tensorInfo, const std::string* buffer);
template Status convertNativeFileFormatRequestTensorToOVTensor<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::shared_ptr<const TensorInfo>& tensorInfo, const std::string* buffer);

template Status convertStringRequestToOVTensor2D<tensorflow::TensorProto>(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer);
template Status convertStringRequestToOVTensor2D<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);

template Status convertStringRequestToOVTensor1D<tensorflow::TensorProto>(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer);
template Status convertStringRequestToOVTensor1D<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);

template Status convertOVTensor2DToStringResponse<tensorflow::TensorProto>(const ov::Tensor& tensor, tensorflow::TensorProto& dst);
template Status convertOVTensor2DToStringResponse<::KFSResponse::InferOutputTensor>(const ov::Tensor& tensor, ::KFSResponse::InferOutputTensor& dst);

}  // namespace ovms
