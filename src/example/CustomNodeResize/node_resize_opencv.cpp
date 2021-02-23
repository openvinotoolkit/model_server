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
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "../../custom_node_interface.h"

static const std::string INPUT_TENSOR_NAME = "image";

auto nchw_to_mat(CustomNodeTensor input) {
    std::vector<float> nchwVector(input.data, input.data + input.dataLength);
    std::vector<float> data(nchwVector.size());
    for(uint64_t c = 0; c < input.dims[1]; ++c) {
        for(uint64_t y = 0; y < input.dims[2]; ++y) {
            for(uint64_t x = 0; x < input.dims[3]; ++x) {
                data[c * (input.dims[2] * input.dims[3]) + y * input.dims[3] + x] = nchwVector[y * input.dims[1] * input.dims[3] + x * input.dims[1] + c];
            }
        }
    }
    cv::Mat image(input.dims[2], input.dims[3] , CV_32F, data.data());
    return image;
}


int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    std::cout << "executing custom node" << std::endl;
    std::cout << "input name " << inputs[0].name << std::endl;
    std::cout << "input dataLength " << inputs[0].dataLength << std::endl;
    std::cout << "input dimsLength " << inputs[0].dimsLength << std::endl;
    if (INPUT_TENSOR_NAME == inputs[0].name) {
        if (inputs[0].dimsLength != 4 ||
            inputs[0].dims[0] != 1 ||
            inputs[0].dims[1] == 3 ) {
            std::cout << "improper " << INPUT_TENSOR_NAME
                      << " dimensions: [" << inputs[0].dims[0]
                      << ", " << inputs[0].dims[1]
                      << ", " << inputs[0].dims[2]
                      << ", " << inputs[0].dims[3] << "]" << std::endl;
            return 1;
        }
        std::cout << "Input valuesPerTensor" << inputs[0].dims[1] << std::endl;
        //numberOfOps = inputs[0].dims[1];
        //valuesPerTensor = inputs[0].dims[2];
        //inputTensor = reinterpret_cast<float*>(inputs[0].data);
    } else {
        std::cout << "Lacking input: " << INPUT_TENSOR_NAME << std::endl;
        return 1;
    }
    // convert input to Mat object
    cv::Mat image = nchw_to_mat(inputs[0]);

    *outputsLength = 1;
 //   *outputs = (struct CustomNodeTensor*)malloc(*outputsLength * sizeof(CustomNodeTensor));
 //   CustomNodeTensor& resultTensor = **outputs;
 //   resultTensor.name = "maximum_tensor";
 //   resultTensor.data = reinterpret_cast<uint8_t*>(result);
 //   resultTensor.dimsLength = 2;
 //   resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsLength * sizeof(uint64_t));
 //   resultTensor.dims[0] = 1;
 //   resultTensor.dims[1] = valuesPerTensor;
 //   resultTensor.dataLength = resultTensor.dims[0] * resultTensor.dims[1] * sizeof(float);
 //   resultTensor.precision = FP32;



    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "input_tensors";
    (*info)->dimsLength = 3;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 4;
    (*info)->dims[2] = 10;
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "maximum_tensor";
    (*info)->dimsLength = 2;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 10;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr) {
    std::cout << "ChooseMaximumCustomLibrary release" << std::endl;
    free(ptr);
    return 0;
}
