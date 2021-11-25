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

#include "status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>

#include "binaryutils.hpp"
#include "logging.hpp"
#include "opencv2/opencv.hpp"

namespace ovms {

int getMatTypeFromTensorPrecision(InferenceEngine::Precision tensorPrecision) {
    switch (tensorPrecision) {
    case InferenceEngine::Precision::FP32:
        return CV_32F;
    case InferenceEngine::Precision::FP16:
        return CV_16F;
    case InferenceEngine::Precision::I16:
        return CV_16S;
    case InferenceEngine::Precision::U8:
        return CV_8U;
    case InferenceEngine::Precision::I8:
        return CV_8S;
    case InferenceEngine::Precision::U16:
        return CV_16U;
    case InferenceEngine::Precision::I32:
        return CV_32S;
    default:
        return -1;
    }
}

bool isPrecisionEqual(int matPrecision, InferenceEngine::Precision tensorPrecision) {
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

Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const InferenceEngine::Precision requestedPrecision) {
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}

bool resizeNeeded(const cv::Mat& image, const std::shared_ptr<TensorInfo>& tensorInfo) {
    if (tensorInfo->getLayout() != InferenceEngine::Layout::NHWC && tensorInfo->getLayout() != InferenceEngine::Layout::ANY) {
        return false;
    }
    int cols = 0;
    int rows = 0;
    if (tensorInfo->getEffectiveShape().size() == 4) {
        cols = tensorInfo->getEffectiveShape()[2];
        rows = tensorInfo->getEffectiveShape()[1];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getEffectiveShape().size() == 5) {
        cols = tensorInfo->getEffectiveShape()[3];
        rows = tensorInfo->getEffectiveShape()[2];
    } else {
        return false;
    }
    if (tensorInfo->getLayout() == InferenceEngine::Layout::ANY) {
        if (cols == 0) {
            cols = image.cols;
        }
        if (rows == 0) {
            rows = image.rows;
        }
    }
    if (cols != image.cols || rows != image.rows) {
        return true;
    }
    return false;
}

Status resizeMat(const cv::Mat& src, cv::Mat& dst, const std::shared_ptr<TensorInfo>& tensorInfo) {
    if (tensorInfo->getLayout() != InferenceEngine::Layout::NHWC && tensorInfo->getLayout() != InferenceEngine::Layout::ANY) {
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    int cols = 0;
    int rows = 0;
    if (tensorInfo->getEffectiveShape().size() == 4) {
        cols = tensorInfo->getEffectiveShape()[2];
        rows = tensorInfo->getEffectiveShape()[1];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getEffectiveShape().size() == 5) {
        cols = tensorInfo->getEffectiveShape()[3];
        rows = tensorInfo->getEffectiveShape()[2];
    } else {
        return StatusCode::UNSUPPORTED_LAYOUT;
    }
    if (tensorInfo->getLayout() == InferenceEngine::Layout::ANY) {
        if (cols == 0) {
            cols = src.cols;
        }
        if (rows == 0) {
            rows = src.rows;
        }
    }
    cv::resize(src, dst, cv::Size(cols, rows));
    return StatusCode::OK;
}

Status validateNumberOfChannels(const std::shared_ptr<TensorInfo>& tensorInfo,
    const cv::Mat input,
    cv::Mat* firstBatchImage) {
    if (tensorInfo->getLayout() != InferenceEngine::Layout::NHWC && tensorInfo->getLayout() != InferenceEngine::ANY) {
        return StatusCode::UNSUPPORTED_LAYOUT;
    }

    // At this point we can either have nhwc format or pretendant to be nhwc but with ANY layout in pipeline info
    size_t numberOfChannels = 0;
    if (tensorInfo->getEffectiveShape().size() == 4) {
        numberOfChannels = tensorInfo->getEffectiveShape()[3];
    } else if (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getEffectiveShape().size() == 5) {
        numberOfChannels = tensorInfo->getEffectiveShape()[4];
    } else {
        return StatusCode::INVALID_NO_OF_CHANNELS;
    }
    if (numberOfChannels == 0 && firstBatchImage) {
        numberOfChannels = firstBatchImage->channels();
    }
    if (numberOfChannels == 0) {
        return StatusCode::OK;
    }
    if ((unsigned int)(input.channels()) != numberOfChannels) {
        SPDLOG_DEBUG("Binary data sent to input: {} has invalid number of channels. Expected: {} Actual: {}",
            tensorInfo->getMappedName(),
            numberOfChannels,
            input.channels());
        return StatusCode::INVALID_NO_OF_CHANNELS;
    }

    return StatusCode::OK;
}

Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage) {
    if (input.cols == firstBatchImage->cols && input.rows == firstBatchImage->rows) {
        return StatusCode::OK;
    }
    SPDLOG_ERROR("Each binary image in request need to have resolution matched");
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}

bool checkBatchSizeMismatch(const std::shared_ptr<TensorInfo>& tensorInfo,
    const int batchSize) {
    if (batchSize <= 0)
        return true;
    if (tensorInfo->getEffectiveShape()[0] == 0) {
        return false;
    }
    if (static_cast<size_t>(batchSize) != tensorInfo->getEffectiveShape()[0])
        return true;
    return false;
}

Status validateInput(const std::shared_ptr<TensorInfo>& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage) {
    // For pipelines with only custom nodes entry, there is no way to deduce layout.
    // With unknown layout, there is no way to deduce pipeline input resolution.
    // This forces binary utility to create blobs with resolution inherited from input binary image from request.
    // To achieve it, in this specific case we require all binary images to have the same resolution.
    if (firstBatchImage && tensorInfo->getLayout() == InferenceEngine::Layout::ANY) {
        auto status = validateResolutionAgainstFirstBatchImage(input, firstBatchImage);
        if (!status.ok()) {
            return status;
        }
    }
    return validateNumberOfChannels(tensorInfo, input, firstBatchImage);
}

Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const tensorflow::TensorProto& src) {
    if (tensorInfo->getLayout() != InferenceEngine::Layout::NHWC && tensorInfo->getLayout() != InferenceEngine::Layout::ANY) {
        return StatusCode::UNSUPPORTED_LAYOUT;
    }

    // 4 for default pipelines, 5 for pipelines with demultiplication at entry
    bool isShapeDimensionValid = tensorInfo->getEffectiveShape().size() == 4 ||
                                 (tensorInfo->isInfluencedByDemultiplexer() && tensorInfo->getEffectiveShape().size() == 5);
    if (!isShapeDimensionValid) {
        return StatusCode::INVALID_SHAPE;
    }

    if (checkBatchSizeMismatch(tensorInfo, src.string_val_size())) {
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}", tensorInfo->getMappedName(), tensorInfo->getEffectiveShape()[0], src.string_val_size());
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

InferenceEngine::SizeVector getShapeFromImages(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
    InferenceEngine::SizeVector dims;
    dims.push_back(images.size());
    if (tensorInfo->isInfluencedByDemultiplexer()) {
        dims.push_back(1);
    }
    dims.push_back(images[0].rows);
    dims.push_back(images[0].cols);
    dims.push_back(images[0].channels());
    return dims;
}

template <typename T>
InferenceEngine::Blob::Ptr createBlobFromMats(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    auto dims = !isPipeline ? tensorInfo->getShape() : getShapeFromImages(images, tensorInfo);
    InferenceEngine::TensorDesc desc{tensorInfo->getPrecision(), dims, InferenceEngine::Layout::ANY};
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<T>(desc);
    blob->allocate();
    char* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<char*>();
    for (cv::Mat image : images) {
        memcpy(ptr, (char*)image.data, image.total() * image.elemSize());
        ptr += (image.total() * image.elemSize());
    }
    return blob;
}

ov::runtime::Tensor createBlobFromMats_2(const std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    ov::Shape shape = tensorInfo->getShape();
    ov::element::Type precision = tensorInfo->getOvPrecision();
    ov::runtime::Tensor tensor(precision, shape);
    char* ptr = (char*)tensor.data();
    for (cv::Mat image : images) {
        memcpy(ptr, (char*)image.data, image.total() * image.elemSize());
        ptr += (image.total() * image.elemSize());
    }
    return tensor;
}

InferenceEngine::Blob::Ptr convertMatsToBlob(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    switch (tensorInfo->getPrecision()) {
    case InferenceEngine::Precision::FP32:
        return createBlobFromMats<float>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::I32:
        return createBlobFromMats<int32_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::I8:
        return createBlobFromMats<int8_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::U8:
        return createBlobFromMats<uint8_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::FP16:
        return createBlobFromMats<uint16_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::U16:
        return createBlobFromMats<uint16_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::I16:
        return createBlobFromMats<int16_t>(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::I64:
    case InferenceEngine::Precision::MIXED:
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::BIN:
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::CUSTOM:
    default:
        return nullptr;
    }
}

ov::runtime::Tensor convertMatsToBlob_2(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    switch (tensorInfo->getPrecision()) {
    case InferenceEngine::Precision::FP32:
    case InferenceEngine::Precision::I32:
    case InferenceEngine::Precision::I8:
    case InferenceEngine::Precision::U8:
    case InferenceEngine::Precision::FP16:
    case InferenceEngine::Precision::U16:
    case InferenceEngine::Precision::I16:
        return createBlobFromMats_2(images, tensorInfo, isPipeline);
    case InferenceEngine::Precision::I64:
    case InferenceEngine::Precision::MIXED:
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::BIN:
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::CUSTOM:
    default:
        return ov::runtime::Tensor();
    }
}

Status convertStringValToBlob(const tensorflow::TensorProto& src, InferenceEngine::Blob::Ptr& blob, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    auto status = validateTensor(tensorInfo, src);
    if (status != StatusCode::OK) {
        return status;
    }

    std::vector<cv::Mat> images;

    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo);
    if (!status.ok()) {
        return status;
    }

    blob = convertMatsToBlob(images, tensorInfo, isPipeline);
    return StatusCode::OK;
}

Status convertStringValToBlob_2(const tensorflow::TensorProto& src, ov::runtime::Tensor& blob, const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    auto status = validateTensor(tensorInfo, src);
    if (status != StatusCode::OK) {
        return status;
    }

    std::vector<cv::Mat> images;

    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo);
    if (!status.ok()) {
        return status;
    }

    blob = convertMatsToBlob_2(images, tensorInfo, isPipeline);
    return StatusCode::OK;
}
}  // namespace ovms
