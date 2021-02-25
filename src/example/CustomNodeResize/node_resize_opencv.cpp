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

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* SCORES_TENSOR_NAME = "scores";
static constexpr const char* GEOMETRY_TENSOR_NAME = "geometry";
static constexpr const char* TEXT_IMAGES_TENSOR_NAME = "text_images";

cv::Mat nchw_to_mat(const CustomNodeTensor* input) {
    // std::vector<float> data(input->dataLength / sizeof(float));
    // std::cout << "Before: " << data.size() << std::endl;
    // uint64_t C = input->dims[1];
    // uint64_t H = input->dims[2];
    // uint64_t W = input->dims[3];
    // std::cout << "C: " << C << "; H: " << H << "; W: " << W << std::endl;
    // for(uint64_t c = 0; c < C; ++c) {
    //     for(uint64_t h = 0; h < H; ++h) {
    //         for(uint64_t w = 0; w < W; ++w) {
    //             uint64_t nchwOffset = (c * H * W) + (h * W) + w;
    //             uint64_t nhwcOffset = (h * W * C) + (w * C) + c;
    //             data[nhwcOffset] = input->data[nchwOffset];
    //         }
    //     }
    // }
    uint64_t H = input->dims[1];
    uint64_t W = input->dims[2];
    cv::Mat image(H, W, CV_32FC3);
    //std::memcpy((void*)image.data, (void*)data.data(), input->dataLength);
    std::memcpy((void*)image.data, (void*)input->data, input->dataLength);
    return image;
}

std::vector<float> reorder_to_chw(cv::Mat* mat) {
    assert(mat->channels() == 3);
    std::vector<float> data(mat->channels() * mat->rows * mat->cols);
    for(int y = 0; y < mat->rows; ++y) {
        for(int x = 0; x < mat->cols; ++x) {
            for(int c = 0; c < mat->channels(); ++c) {
                data[c * (mat->rows * mat->cols) + y * mat->cols + x] = mat->at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return std::move(data);
}

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    const CustomNodeTensor* imageTensor = nullptr;  // NCHW
    const CustomNodeTensor* scoresTensor = nullptr;
    const CustomNodeTensor* geometryTensor = nullptr;

    for (int i = 0; i < inputsLength; i++) {
        if (std::strcmp(inputs[i].name, IMAGE_TENSOR_NAME) == 0) {
            imageTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, SCORES_TENSOR_NAME) == 0) {
            scoresTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, GEOMETRY_TENSOR_NAME) == 0) {
            geometryTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    if (!imageTensor) {
        std::cout << "Missing input: " << IMAGE_TENSOR_NAME << std::endl;
        return 1;
    }

    // if (!scoresTensor) {
    //     std::cout << "Missing input: " << SCORES_TENSOR_NAME << std::endl;
    //     return 1;
    // }

    // if (!geometryTensor) {
    //     std::cout << "Missing input: " << GEOMETRY_TENSOR_NAME << std::endl;
    //     return 1;
    // }
    cv::Mat image = nchw_to_mat(imageTensor);
    cv::imwrite("/workspace/east_utils/test2.jpg", image);
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(200, 50));
    cv::imwrite("/workspace/east_utils/test.jpg", resizedImage);
    std::vector<float> buffer = reorder_to_chw(&resizedImage);
    uint8_t* data = (uint8_t*)malloc(buffer.size() * sizeof(float));
    std::memcpy(data, buffer.data(), buffer.size() * sizeof(float));

    *outputsLength = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsLength * sizeof(CustomNodeTensor));
    CustomNodeTensor& resultTensor = **outputs;
    resultTensor.name = TEXT_IMAGES_TENSOR_NAME;
    resultTensor.data = data;
    resultTensor.dataLength = buffer.size() * sizeof(float);
    resultTensor.dimsLength = 4;
    resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsLength * sizeof(uint64_t));
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = 3;
    resultTensor.dims[2] = 50;
    resultTensor.dims[3] = 200;
    resultTensor.precision = FP32;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsLength = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 3;
    (*info)[0].dims[2] = 1024;
    (*info)[0].dims[3] = 1920;
    (*info)[0].precision = FP32;

    // (*info)[1].name = SCORES_TENSOR_NAME;
    // (*info)[1].dimsLength = 4;
    // (*info)[1].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    // (*info)[1].dims[0] = 1;
    // (*info)[1].dims[1] = 1;
    // (*info)[1].dims[2] = 256;
    // (*info)[1].dims[3] = 480;
    // (*info)[1].precision = FP32;

    // (*info)[2].name = GEOMETRY_TENSOR_NAME;
    // (*info)[2].dimsLength = 4;
    // (*info)[2].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    // (*info)[2].dims[0] = 1;
    // (*info)[2].dims[1] = 5;
    // (*info)[2].dims[2] = 256;
    // (*info)[2].dims[3] = 480;
    // (*info)[2].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = TEXT_IMAGES_TENSOR_NAME;
    (*info)->dimsLength = 4;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[0] = 3;
    (*info)->dims[0] = 50;
    (*info)->dims[0] = 200;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr) {
    std::cout << "Test release" << std::endl;
    free(ptr);
    return 0;
}
