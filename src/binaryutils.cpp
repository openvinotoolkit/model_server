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
#include "status.hpp"

namespace ovms {

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

cv::Mat convertStringValToMat(const std::string& stringVal) {
    std::vector<unsigned char> data(stringVal.begin(), stringVal.end());
    cv::Mat dataMat(data, true);

    try {
        return cv::imdecode(dataMat, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        SPDLOG_ERROR("Error during string_val to mat conversion: {}", e.what());
        return cv::Mat{};
    }
}

Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision) {
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        SPDLOG_ERROR("Error during binary input conversion: not supported precision: {}", toString(requestedPrecision));
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}

Status validateLayout(const std::shared_ptr<TensorInfo>& tensorInfo) {
    if ((tensorInfo->getLayout() != "NHWC") &&
        (tensorInfo->getLayout() != Layout::getUnspecifiedLayout()) &&  // handle DAG
        (tensorInfo->getLayout() != Layout::getDefaultLayout())) {      // handle model without Layout set
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    return StatusCode::OK;
}

bool resizeNeeded(const cv::Mat& image, const std::shared_ptr<TensorInfo>& tensorInfo) {
    Dimension cols = Dimension::any();
    Dimension rows = Dimension::any();
    if (tensorInfo->getShape().size() == 4) {
        cols = tensorInfo->getShape()[2];
        rows = tensorInfo->getShape()[1];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getShape().size() == 5) {
        cols = tensorInfo->getShape()[3];
        rows = tensorInfo->getShape()[2];
    } else {
        return false;
    }
    if (cols.isAny()) {
        cols = image.cols;
    }
    if (rows.isAny()) {
        rows = image.rows;
    }
    if ((!cols.match(image.cols)) || (!rows.match(image.rows))) {
        return true;
    }
    return false;
}

Status resizeMat(const cv::Mat& src, cv::Mat& dst, const std::shared_ptr<TensorInfo>& tensorInfo) {
    Dimension cols = Dimension::any();
    Dimension rows = Dimension::any();
    if (tensorInfo->getShape().size() == 4) {
        cols = tensorInfo->getShape()[2];
        rows = tensorInfo->getShape()[1];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getShape().size() == 5) {
        cols = tensorInfo->getShape()[3];
        rows = tensorInfo->getShape()[2];
    } else {
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    if (cols.isAny()) {
        cols = src.cols;
    }
    if (rows.isAny()) {
        rows = src.rows;
    }
    if (cols.isDynamic()) {
        dimension_value_t value = src.cols;
        if (src.cols < cols.getMinValue())
            value = cols.getMinValue();

        if (src.cols > cols.getMaxValue())
            value = cols.getMaxValue();

        if (value != src.cols)
            cols = Dimension(value);
    }
    if (rows.isDynamic()) {
        dimension_value_t value = src.rows;
        if (src.rows < rows.getMinValue())
            value = rows.getMinValue();

        if (src.rows > rows.getMaxValue())
            value = rows.getMaxValue();

        if (value != src.rows)
            rows = Dimension(value);
    }
    cv::resize(src, dst, cv::Size(cols.getStaticValue(), rows.getStaticValue()));
    return StatusCode::OK;
}

Status validateNumberOfChannels(const std::shared_ptr<TensorInfo>& tensorInfo,
    const cv::Mat input,
    cv::Mat* firstBatchImage) {

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

Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage) {
    if (input.cols == firstBatchImage->cols && input.rows == firstBatchImage->rows) {
        return StatusCode::OK;
    }
    SPDLOG_ERROR("Each binary image in request needs to have resolution matched. First cols: {}, rows: {}, current cols: {}, rows: {}",
        firstBatchImage->cols, firstBatchImage->rows, input.cols, input.rows);
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}

bool checkBatchSizeMismatch(const std::shared_ptr<TensorInfo>& tensorInfo,
    const int batchSize) {
    if (!tensorInfo->getBatchSize().has_value()) {
        return true;
    }
    return !tensorInfo->getBatchSize().value().match(batchSize);
}

Status validateInput(const std::shared_ptr<TensorInfo>& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage) {
    // For pipelines with only custom nodes entry, or models with default layout there is no way to deduce layout.
    // With unknown layout, there is no way to deduce pipeline input resolution.
    // This forces binary utility to create tensors with resolution inherited from input binary image from request.
    // To achieve it, in this specific case we require all binary images to have the same resolution.
    // TODO check if H/W is undefined and only then check this CVS-77193
    if (firstBatchImage &&
        (tensorInfo->getLayout() == Layout::getUnspecifiedLayout())) {
        auto status = validateResolutionAgainstFirstBatchImage(input, firstBatchImage);
        if (!status.ok()) {
            return status;
        }
    }
    return validateNumberOfChannels(tensorInfo, input, firstBatchImage);
}

Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const tensorflow::TensorProto& src) {
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

    for (size_t i = 0; i < src.string_val_size(); i++) {
        if (src.string_val(i).size() <= 0) {
            return StatusCode::STRING_VAL_EMPTY;
        }
    }

    return StatusCode::OK;
}

Status convertTensorToMatsMatchingTensorInfo(const tensorflow::TensorProto& src, std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
    for (int i = 0; i < src.string_val_size(); i++) {
        cv::Mat image = convertStringValToMat(src.string_val(i));
        if (image.data == nullptr)
            return StatusCode::IMAGE_PARSING_FAILED;

        cv::Mat* firstImage = images.size() == 0 ? nullptr : &images.at(0);
        auto status = validateInput(tensorInfo, image, firstImage);
        if (status != StatusCode::OK) {
            return status;
        }
        if (!isPrecisionEqual(image.depth(), tensorInfo->getPrecision())) {
            cv::Mat imageCorrectPrecision;
            status = convertPrecision(image, imageCorrectPrecision, tensorInfo->getPrecision());

            if (status != StatusCode::OK) {
                return status;
            }
            image = std::move(imageCorrectPrecision);
        }
        if (resizeNeeded(image, tensorInfo)) {
            cv::Mat imageResized;
            status = resizeMat(image, imageResized, tensorInfo);
            if (!status.ok()) {
                return status;
            }
            image = std::move(imageResized);
        }
        images.push_back(image);
    }

    return StatusCode::OK;
}

shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
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

ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
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

ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    switch (tensorInfo->getPrecision()) {
    case ovms::Precision::FP32:
    case ovms::Precision::I32:
    case ovms::Precision::FP64:
    case ovms::Precision::I8:
    case ovms::Precision::U8:
    case ovms::Precision::FP16:
    case ovms::Precision::U16:
    case ovms::Precision::I16:
        return createTensorFromMats(images, tensorInfo, isPipeline);
    case ovms::Precision::MIXED:
    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
    case ovms::Precision::BOOL:
    case ovms::Precision::CUSTOM:
    default:
        return ov::Tensor();
    }
}

Status convertStringValToTensor(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    auto status = validateTensor(tensorInfo, src);
    if (status != StatusCode::OK) {
        return status;
    }

    std::vector<cv::Mat> images;

    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo);
    if (!status.ok()) {
        return status;
    }

    tensor = convertMatsToTensor(images, tensorInfo, isPipeline);
    if (!tensor) {
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}
}  // namespace ovms
