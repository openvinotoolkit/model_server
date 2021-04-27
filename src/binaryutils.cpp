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

namespace ovms {

int precisionToMatType(InferenceEngine::Precision precision)
{
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return CV_32FC3;
    case InferenceEngine::Precision::FP16:
        return CV_16FC3;
    case InferenceEngine::Precision::U8:
        return CV_8UC3;
    case InferenceEngine::Precision::I8:
        return CV_8SC3;
    case InferenceEngine::Precision::U16:
        return CV_16UC3;
    case InferenceEngine::Precision::I16:
        return CV_16SC3;
    case InferenceEngine::Precision::I32:
        return CV_32SC3;
    case InferenceEngine::Precision::I64:
    case InferenceEngine::Precision::MIXED:
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::BIN:
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::CUSTOM:
    default:
        return CV_32SC3;
    }
}

StatusCode convertBinaryToTensor(void *byte, const std::shared_ptr<TensorInfo>& tensorInfo, tensorflow::TensorProto& tensor){
    auto layout = tensorInfo->getLayout();
    int height, width;

    if(layout == InferenceEngine::Layout::NCHW)
    {
        height = tensorInfo->getShape()[2];
        width = tensorInfo->getShape()[3];
    }
    else
    {
        height = tensorInfo->getShape()[1];
        width = tensorInfo->getShape()[2];
    }
    auto image = cv::Mat(height, width, precisionToMatType(tensorInfo->getPrecision()), byte);
    tensor.set_tensor_content(image.data, image.datastart - image.dataend);
    return StatusCode::OK;
}
}