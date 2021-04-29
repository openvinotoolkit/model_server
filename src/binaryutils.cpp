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

StatusCode convertStringValToTensorContent(const tensorflow::TensorProto& stringVal, tensorflow::TensorProto& tensorContent){
    int contentSize = 0;
    std::vector<cv::Mat> images;
    for(int i = 0; i < stringVal.string_val_size(); i++){
        contentSize += stringVal.string_val(i).length();
        std::vector<char> vectordata(stringVal.string_val(i).begin(),stringVal.string_val(i).end());
        cv::Mat data_mat(vectordata,true);
        cv::Mat image(cv::imdecode(data_mat,1));
        contentSize += image.total() * image.elemSize();
        images.push_back(image);
    }
    char* content = new char[contentSize];

    int offset = 0;
    for(cv::Mat image : images){
        memcpy(&(content[offset]), image.data, image.total() * image.elemSize());
        offset += image.total() * image.elemSize();
    }
    tensorContent.set_tensor_content(content, contentSize);
    return StatusCode::OK;
}
}