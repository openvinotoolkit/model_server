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

StatusCode convertPrecision(const cv::Mat& src, cv::Mat& dst, const InferenceEngine::Precision requestedPrecision) {
    int type = getMatTypeFromTensorPrecision(requestedPrecision);
    if (type == -1) {
        return StatusCode::INVALID_PRECISION;
    }

    src.convertTo(dst, type);
    return StatusCode::OK;
}

StatusCode convertTensorToCorrectPrecisionMats(const tensorflow::TensorProto& src, std::vector<cv::Mat>& images, const std::shared_ptr<TensorInfo>& tensorInfo) {
    for (int i = 0; i < src.string_val_size(); i++) {
        cv::Mat image = convertStringValToMat(src.string_val(i));
        if (image.data == NULL)
            return StatusCode::IMAGE_PARSING_FAILED;

        if (!isPrecisionEqual(image.depth(), tensorInfo->getPrecision())) {
            cv::Mat imagePrecisionConverted;
            auto status = convertPrecision(image, imagePrecisionConverted, tensorInfo->getPrecision());

            if (status != StatusCode::OK) {
                return status;
            }

            images.push_back(imagePrecisionConverted);
        } else {
            images.push_back(image);
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
            int cols = tensorInfo->getShape()[3];
            int rows = tensorInfo->getShape()[2];
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(cols,rows));

            auto imgBuffer = reorder_to_nchw((T*)resized.data, resized.rows, resized.cols, resized.channels());
            memcpy(ptr + offset, (char*)imgBuffer.data(), resized.total() * resized.elemSize());
            offset += resized.total() * resized.elemSize();
            break;
        }
        else
        {
            int cols = tensorInfo->getShape()[2];
            int rows = tensorInfo->getShape()[1];
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(cols,rows));
            memcpy(ptr + offset, resized.data, resized.total() * resized.elemSize());
            offset += resized.total() * resized.elemSize();
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
        // case InferenceEngine::Precision::FP16: {
        //     auto blob = InferenceEngine::make_shared_blob<uint16_t>(tensorInfo->getTensorDesc());
        //     blob->allocate();
        //     // Needs conversion due to zero padding for each value:
        //     // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
        //     uint16_t* ptr = blob->buffer().as<uint16_t*>();
        //     auto size = static_cast<size_t>(requestInput.half_val_size());
        //     for (size_t i = 0; i < size; i++) {
        //         ptr[i] = requestInput.half_val(i);
        //     }
        //     return blob;
        // }
        // case InferenceEngine::Precision::U16: {
        //     auto blob = InferenceEngine::make_shared_blob<uint16_t>(tensorInfo->getTensorDesc());
        //     blob->allocate();
        //     // Needs conversion due to zero padding for each value:
        //     // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
        //     uint16_t* ptr = blob->buffer().as<uint16_t*>();
        //     auto size = static_cast<size_t>(requestInput.int_val_size());
        //     for (size_t i = 0; i < size; i++) {
        //         ptr[i] = requestInput.int_val(i);
        //     }
        //     return blob;
        // }
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

StatusCode convertStringValToBlob(const tensorflow::TensorProto& src, InferenceEngine::Blob::Ptr* blob, const std::shared_ptr<TensorInfo>& tensorInfo) {
    std::vector<cv::Mat> images;

    auto status = convertTensorToCorrectPrecisionMats(src, images, tensorInfo);

    if (status != StatusCode::OK) {
        return status;
    }

    *blob = convertMatsToBlob(images, tensorInfo);
    return StatusCode::OK;
}
}  // namespace ovms