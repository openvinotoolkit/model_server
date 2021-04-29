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

StatusCode convertBinaryStringValToTensorContent(tensorflow::TensorProto& tensor){
    int sizeOfTensorContent = 0;
    std::vector<cv::Mat> images;
    for(int i = 0; i < tensor.string_val_size(); i++){
        sizeOfTensorContent += tensor.string_val(i).length();
        std::vector<char> vectordata(tensor.string_val(i).begin(),tensor.string_val(i).end());
        cv::Mat data_mat(vectordata,true);
        cv::Mat image(cv::imdecode(data_mat,1));
        sizeOfTensorContent += image.total() * image.elemSize();
        images.push_back(image);
    }
    tensor.clear_string_val();
    char* tensorContent = new char[sizeOfTensorContent];

    int offset = 0;
    for(cv::Mat image : images){
        memcpy(&(tensorContent[offset]), image.data, image.total() * image.elemSize());
        offset += image.total() * image.elemSize();
    }
    tensor.set_tensor_content(tensorContent, sizeOfTensorContent);
    return StatusCode::OK;
}
}