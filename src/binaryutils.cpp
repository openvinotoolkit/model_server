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

#include "opencv2/opencv.hpp"
#include "binaryutils.hpp"
#include "logging.hpp"

namespace ovms {

int getMatTypeFromTensorPrecision(InferenceEngine::Precision tensorPrecision)
{
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

bool isPrecisionEqual(int matPrecision, InferenceEngine::Precision tensorPrecision)
{
    int convertedTensorPrecision = getMatTypeFromTensorPrecision(tensorPrecision);
    if(convertedTensorPrecision == matPrecision){
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

StatusCode convertStringValToTensorContent(const tensorflow::TensorProto& stringVal, tensorflow::TensorProto& tensorContent, const std::shared_ptr<TensorInfo>& tensorInfo){
    int contentSize = 0;
    std::vector<cv::Mat> images;

    for(int i = 0; i < stringVal.string_val_size(); i++){
        std::vector<unsigned char> vectordata(stringVal.string_val(i).begin(),stringVal.string_val(i).end());
        cv::Mat data_mat(vectordata,true);
        cv::Mat image(cv::imdecode(data_mat,cv::IMREAD_UNCHANGED)); 

        if(!isPrecisionEqual(image.depth(), tensorInfo->getPrecision())){
            int type = getMatTypeFromTensorPrecision(tensorInfo->getPrecision());
            if(type == -1)
            {
                return StatusCode::INVALID_PRECISION;
            }
            
            cv::Mat imagePrecisionConverted(image.rows,image.cols,type);
            image.convertTo(imagePrecisionConverted, type);

            contentSize += imagePrecisionConverted.total() * imagePrecisionConverted.elemSize();
            images.push_back(imagePrecisionConverted);
        }
        else{
            contentSize += image.total() * image.elemSize();
            images.push_back(image);
        }
    }
    char* content = new char[contentSize];

    int offset = 0;
    for(cv::Mat image : images){
        if(tensorInfo->getLayout() == InferenceEngine::Layout::NCHW)
        {
            int cols = tensorInfo->getShape()[3];
            int rows = tensorInfo->getShape()[2];
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(cols,rows));
            switch (image.elemSize1()) {
            case 1:
                {
                    auto imgBuffer = reorder_to_nchw((uint8_t*)resized.data, resized.rows, resized.cols, resized.channels());
                    memcpy(content, imgBuffer.data(), resized.total() * resized.elemSize());
                    offset += resized.total() * resized.elemSize();
                    break;
                }
            case 2:
                {
                    auto imgBuffer = reorder_to_nchw((uint16_t*)resized.data, resized.rows, resized.cols, resized.channels());
                    memcpy(content, imgBuffer.data(), resized.total() * resized.elemSize());
                    offset += resized.total() * resized.elemSize();
                    break;
                }
            case 4:
                {
                    auto imgBuffer = reorder_to_nchw((uint32_t*)resized.data, resized.rows, resized.cols, resized.channels());
                    memcpy(content, imgBuffer.data(), resized.total() * resized.elemSize());
                    offset += resized.total() * resized.elemSize();
                    break;
                }
            default:
                return StatusCode::INVALID_PRECISION;
            }

        }
        else
        {
            int cols = tensorInfo->getShape()[2];
            int rows = tensorInfo->getShape()[1];
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(cols,rows));

            memcpy(content, resized.data, resized.total() * resized.elemSize());
            offset += resized.total() * resized.elemSize();
        }
    }
    std::string sContent(content, contentSize);
    tensorContent.set_allocated_tensor_content(&sContent);
    return StatusCode::OK;
}
}