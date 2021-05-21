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

#include "binaryutils.hpp"
#include "logging.hpp"
#include "opencv2/opencv.hpp"

#include <inference_engine.hpp>

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

template <typename T>
std::vector<T> reorder_to_nchw(const T* nhwcVector, int rows, int cols, int channels) {
    std::vector<T> nchwVector(rows * cols * channels);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                nchwVector[c * (rows * cols) + y * cols + x] = reinterpret_cast<const T*>(nhwcVector)[y * channels * cols + x * channels + c];
            }
        }
    }
    return std::move(nchwVector);
}

cv::Mat convertStringValToMat(const std::string& stringVal) {
    std::vector<unsigned char> vectordata(stringVal.begin(), stringVal.end());
    cv::Mat data_mat(vectordata, true);

    return cv::imdecode(data_mat, cv::IMREAD_UNCHANGED);
}

Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const InferenceEngine::Precision requestedPrecision) {
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}

Status resizeMat(const cv::Mat& src, cv::Mat& dst, const std::shared_ptr<TensorInfo>& tensorInfo) {
    if(tensorInfo->getLayout() == InferenceEngine::Layout::NCHW)
    {
        int cols = tensorInfo->getShape()[3];
        int rows = tensorInfo->getShape()[2];
        cv::resize(src, dst, cv::Size(cols,rows));
        return StatusCode::OK;
    }
    else if (tensorInfo->getLayout() == InferenceEngine::Layout::NHWC)
    {
        int cols = tensorInfo->getShape()[2];
        int rows = tensorInfo->getShape()[1];
        cv::resize(src, dst, cv::Size(cols,rows));
        return StatusCode::OK;
    }

    return StatusCode::UNSUPPORTED_LAYOUT;
}

Status validateNumberOfChannels(const std::shared_ptr<TensorInfo>& tensorInfo,
    const cv::Mat input) {
    // Network and input must have the same number of shape dimensions. 
    if ((unsigned int)(input.channels()) != tensorInfo->getShape()[1]) {
        SPDLOG_DEBUG("Binary sent to input: {} has invalid number of channels. Expected: {} Actual: {}", tensorInfo->getMappedName(),tensorInfo->getShape()[1], input.dims);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
    }
    return StatusCode::OK;
}

const bool checkBatchSizeMismatch(const std::shared_ptr<TensorInfo>& tensorInfo,
    const int batchSize) {
    if (static_cast<size_t>(batchSize) != tensorInfo->getShape()[0])
        return true;
    return false;
}

const Status validateInput(const std::shared_ptr<TensorInfo>& tensorInfo,
    const cv::Mat input) {
    auto status = validateNumberOfChannels(tensorInfo, input);
    if (!status.ok())
        return status;

    return StatusCode::OK;
}

const Status validateTensor(const std::shared_ptr<TensorInfo>& tensorInfo,
    const tensorflow::TensorProto& src) {
    if(tensorInfo->getLayout() == InferenceEngine::Layout::NCHW ||
       tensorInfo->getLayout() == InferenceEngine::Layout::NHWC)
    {
        return StatusCode::OK;
    }

    if (checkBatchSizeMismatch(tensorInfo, src.string_val_size())){
        SPDLOG_DEBUG("Input: {} request batch size is incorrect. Expected: {} Actual: {}", tensorInfo->getMappedName(), tensorInfo->getShape()[0],src.string_val_size());
        return Status(StatusCode::UNSUPPORTED_LAYOUT);
    }

    return StatusCode::OK;
}

Status convertTensorToMatsMatchingTensorInfo(const tensorflow::TensorProto& src, std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {

    for (int i = 0; i < src.string_val_size(); i++) {
        cv::Mat image = convertStringValToMat(src.string_val(i));
        if (image.data == NULL)
            return StatusCode::IMAGE_PARSING_FAILED;
                    
        auto status = validateInput(tensorInfo, image);
        if (status != StatusCode::OK) {
            return status;
        }

        if (!isPrecisionEqual(image.depth(), tensorInfo->getPrecision())) {
            cv::Mat imagePrecisionConverted;
            status = convertPrecision(image, imagePrecisionConverted, tensorInfo->getPrecision());

            if (status != StatusCode::OK) {
                return status;
            }

            cv::Mat imageResized;
            resizeMat(imagePrecisionConverted, imageResized, tensorInfo);
            if (status != StatusCode::OK) {
                return status;
            }

            images.push_back(imageResized);
        } else {
            cv::Mat imageResized;
            resizeMat(image, imageResized, tensorInfo);
            if (status != StatusCode::OK) {
                return status;
            }
            images.push_back(imageResized);
        }
    }

    return StatusCode::OK;
}

template <typename T>
InferenceEngine::Blob::Ptr createBlobFromMats(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo){
    int offset = 0;
    auto blob = InferenceEngine::make_shared_blob<T>(tensorInfo->getTensorDesc());
    blob->allocate();
    T* ptr = blob->buffer();
    for(cv::Mat image : images){
        if(tensorInfo->getLayout() == InferenceEngine::Layout::NCHW)
        {
            auto imgBuffer = reorder_to_nchw((T*)image.data, image.rows, image.cols, image.channels());
            memcpy(ptr + offset, (char*)imgBuffer.data(), image.total() * image.elemSize());
            offset += image.total() * image.elemSize();
            break;
        }
        else
        {            
            memcpy(ptr + offset, image.data, image.total() * image.elemSize());
            offset += image.total() * image.elemSize();
        }
    }
    return blob;
}

InferenceEngine::Blob::Ptr convertMatsToBlob(std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
    switch (tensorInfo->getPrecision()) {
        case InferenceEngine::Precision::FP32:
            return createBlobFromMats<float>(images, tensorInfo);
        case InferenceEngine::Precision::I32:
            return createBlobFromMats<int32_t>(images, tensorInfo);
        case InferenceEngine::Precision::I8:
            return createBlobFromMats<int8_t>(images, tensorInfo);
        case InferenceEngine::Precision::U8:
            return createBlobFromMats<uint8_t>(images, tensorInfo);
        case InferenceEngine::Precision::FP16:
            return createBlobFromMats<uint16_t>(images, tensorInfo);
        case InferenceEngine::Precision::U16:
            return createBlobFromMats<uint16_t>(images, tensorInfo);
        case InferenceEngine::Precision::I16:
            return createBlobFromMats<int16_t>(images, tensorInfo);
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

Status convertStringValToBlob(const tensorflow::TensorProto& src, InferenceEngine::Blob::Ptr* blob, const std::shared_ptr<TensorInfo>& tensorInfo) {
    auto status = validateTensor(tensorInfo, src);
    if (status != StatusCode::OK) {
        return status;
    }
    
    std::vector<cv::Mat> images;

    status = convertTensorToMatsMatchingTensorInfo(src, images, tensorInfo);
    if (status != StatusCode::OK) {
        return status;
    }

    *blob = convertMatsToBlob(images, tensorInfo);
    return StatusCode::OK;
}
}  // namespace ovms