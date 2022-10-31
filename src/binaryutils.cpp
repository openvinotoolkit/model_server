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
#include "binaryutils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "logging.hpp"
#include "opencv2/opencv.hpp"
#include "profiler.hpp"
#include "status.hpp"

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
    std::vector<uint8_t> data(image.begin(), image.end());
    cv::Mat dataMat(data);

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

static Status validateLayout(const std::shared_ptr<TensorInfo>& tensorInfo) {
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

static Status validateNumberOfChannels(const std::shared_ptr<TensorInfo>& tensorInfo,
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

static bool checkBatchSizeMismatch(const std::shared_ptr<TensorInfo>& tensorInfo,
    const int batchSize) {
    OVMS_PROFILE_FUNCTION();
    if (!tensorInfo->getBatchSize().has_value()) {
        return true;
    }
    return !tensorInfo->getBatchSize().value().match(batchSize);
}

static Status validateInput(const std::shared_ptr<TensorInfo>& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment) {
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

static Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const tensorflow::TensorProto& src) {
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


static Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const ::KFSRequest::InferInputTensor& src) {
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

    if (checkBatchSizeMismatch(tensorInfo, src.contents().bytes_contents_size())) {
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}",
            tensorInfo->getMappedName(),
            tensorInfo->getBatchSize().has_value() ? tensorInfo->getBatchSize().value().toString() : std::string{"none"},
            src.contents().bytes_contents_size());
        return StatusCode::INVALID_BATCH_SIZE;
    }

    for (int i = 0; i < src.contents().bytes_contents_size(); i++) {
        if (src.contents().bytes_contents(i).size() <= 0) {
            return StatusCode::BYTES_CONTENTS_EMPTY;
        }
    }

    if (src.contents().bytes_contents_size() <= 0) {
        return StatusCode::BYTES_CONTENTS_EMPTY;
    }

    return StatusCode::OK;
}

static Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const ::inference::ModelInferRequest& src) {
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

    if (src.raw_input_contents().size() <= 0) {
        return StatusCode::BYTES_CONTENTS_EMPTY;
    }

    return StatusCode::OK;
}

static Dimension getTensorInfoHeightDim(const std::shared_ptr<TensorInfo>& tensorInfo) {
    size_t numberOfShapeDimensions = tensorInfo->getShape().size();
    if (numberOfShapeDimensions < 4 || numberOfShapeDimensions > 5) {
        throw std::logic_error("wrong number of shape dimensions");
    }
    size_t position = numberOfShapeDimensions == 4 ? /*NHWC*/ 1 : /*N?HWC*/ 2;
    return tensorInfo->getShape()[position];
}

static Dimension getTensorInfoWidthDim(const std::shared_ptr<TensorInfo>& tensorInfo) {
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

static bool isResizeSupported(const std::shared_ptr<TensorInfo>& tensorInfo) {
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

const std::string getBinaryInput(const ::inference::ModelInferRequest& tensor, size_t i) {
    return tensor.raw_input_contents()[0];
}

inline static const std::string& getBinaryInput(const ::inference::ModelInferRequest::InferInputTensor& tensor, size_t i) {
    return tensor.contents().bytes_contents(i);
}

inline static int getBinaryInputsSize(const tensorflow::TensorProto& tensor) {
    return tensor.string_val_size();
}

inline static int getBinaryInputsSize(const ::KFSRequest::InferInputTensor& tensor) {
    return tensor.contents().bytes_contents_size();
}

int getBinaryInputsSize(const ::inference::ModelInferRequest& tensor) {
    return 1;
}

inline static int getBinaryInputsSize(const ::inference::ModelInferRequest::InferInputTensor& tensor) {
    return tensor.contents().bytes_contents_size();
}

template <typename TensorType>
static Status convertTensorToMatsMatchingTensorInfo(const TensorType& src, std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    Dimension targetHeight = getTensorInfoHeightDim(tensorInfo);
    Dimension targetWidth = getTensorInfoWidthDim(tensorInfo);

    // Enforce resolution alignment against first image in the batch if resize is not supported.
    bool resizeSupported = isResizeSupported(tensorInfo);
    bool enforceResolutionAlignment = !resizeSupported;

    for (int i = 0; i < getBinaryInputsSize(src); i++) {
        cv::Mat image = convertStringToMat(getBinaryInput(src, i));
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

static shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
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

static ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
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

static ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
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
static Status convertBinaryRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    auto status = validateTensor(tensorInfo, src);
    if (status != StatusCode::OK) {
        return status;
    }

    std::vector<cv::Mat> images;
    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo);
    if (!status.ok()) {
        return status;
    }
    tensor = convertMatsToTensor(images, tensorInfo);
    if (!tensor) {
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}

template Status convertBinaryRequestTensorToOVTensor<tensorflow::TensorProto>(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);
template Status convertBinaryRequestTensorToOVTensor<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);
template Status convertBinaryRequestTensorToOVTensor<::inference::ModelInferRequest>(const ::inference::ModelInferRequest& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);
}  // namespace ovms
