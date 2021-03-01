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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "./custom_node_interface.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* SCORES_TENSOR_NAME = "scores";
static constexpr const char* GEOMETRY_TENSOR_NAME = "geometry";
static constexpr const char* TEXT_IMAGES_TENSOR_NAME = "text_images";

#define NODE_ASSERT(cond, msg)                       \
    if (!(cond)) {                                   \
        std::cout << "Assert: " << msg << std::endl; \
        return 1;                                    \
    }

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
    // std::memcpy((void*)image.data, (void*)data.data(), input->dataLength);
    std::memcpy((void*)image.data, (void*)input->data, input->dataLength);
    return image;
}

cv::Mat nhwc_to_mat(const CustomNodeTensor* input) {
    uint64_t H = input->dims[1];
    uint64_t W = input->dims[2];
    cv::Mat image(H, W, CV_32FC3, input->data);
    // std::memcpy((void*)image.data, (void*)input->data, input->dataLength);
    return image;
}

std::vector<float> reorder_to_chw(cv::Mat* mat) {
    assert(mat->channels() == 3);
    std::vector<float> data(mat->channels() * mat->rows * mat->cols);
    for (int y = 0; y < mat->rows; ++y) {
        for (int x = 0; x < mat->cols; ++x) {
            for (int c = 0; c < mat->channels(); ++c) {
                data[c * (mat->rows * mat->cols) + y * mat->cols + x] = mat->at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return std::move(data);
}

struct Box {
    uint64_t startX, endX, startY, endY;
    float score;
};

// TODO
void apply_non_max_suppression(std::vector<Box>& boxes) {
    if (boxes.size() == 0) {
        return;
    }
}

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    const CustomNodeTensor* imageTensor = nullptr;
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

    NODE_ASSERT(imageTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(scoresTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(geometryTensor->precision == FP32, "image input is not FP32");

    NODE_ASSERT(imageTensor, "Missing image input");
    NODE_ASSERT(scoresTensor, "Missing scores input");
    NODE_ASSERT(geometryTensor, "Missing geometry input");

    // Image layout = NHWC
    uint64_t imageHeight = imageTensor->dims[1];
    uint64_t imageWidth = imageTensor->dims[2];

    std::cout << "Input image height: " << imageHeight << "; image width: " << imageWidth << std::endl;

    cv::Mat image = nhwc_to_mat(imageTensor);

    NODE_ASSERT(image.cols == imageWidth, "Mat generation failed");
    NODE_ASSERT(image.rows == imageHeight, "Mat generation failed");

    uint64_t numRows = scoresTensor->dims[2];
    uint64_t numCols = scoresTensor->dims[3];

    std::cout << "Rows: " << numRows << "; Cols: " << numCols << std::endl;

    NODE_ASSERT(scoresTensor->dims[1] == 1, "scores has dim 1 not equal to 1");
    NODE_ASSERT(geometryTensor->dims[1] == 5, "geometry has dim 1 not equal to 5");

    NODE_ASSERT(scoresTensor->dims[2] == geometryTensor->dims[2], "scores and geometry has not equal dim 2");
    NODE_ASSERT(scoresTensor->dims[3] == geometryTensor->dims[3], "scores and geometry has not equal dim 3");

    NODE_ASSERT((numRows * 4) == imageHeight, "image is not x4 larger than score/geometry data");
    NODE_ASSERT((numCols * 4) == imageWidth, "image is not x4 larger than score/geometry data");

    std::cout << "Scores bytes: " << scoresTensor->dataLength << std::endl;
    std::cout << "Geometry bytes: " << geometryTensor->dataLength << std::endl;

    std::vector<Box> boxes;

    // Extract the scores (probabilities), followed by the geometrical data used to derive potential bounding box coordinates that surround text
    for (uint64_t y = 0; y < numRows; y++) {
        float* scoresData = (float*)scoresTensor->data + (y * numCols);
        float* xData0 = (float*)geometryTensor->data + ((0 * numRows * numCols) + (y * numCols));
        float* xData1 = (float*)geometryTensor->data + ((1 * numRows * numCols) + (y * numCols));
        float* xData2 = (float*)geometryTensor->data + ((2 * numRows * numCols) + (y * numCols));
        float* xData3 = (float*)geometryTensor->data + ((3 * numRows * numCols) + (y * numCols));
        float* anglesData = (float*)geometryTensor->data + ((4 * numRows * numCols) + (y * numCols));

        for (uint64_t x = 0; x < numCols; x++) {
            static const float maxValInDataset = 1;
            float score = scoresData[x];
            // If our score does not have sufficient probability, ignore it
            if (score < (maxValInDataset * 0.5)) {
                continue;
            }
            std::cout << "Found confidence: " << scoresData[x] << std::endl;

            // Compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            uint64_t offsetX = x * 4;
            uint64_t offsetY = y * 4;

            std::cout << "For coordinate: offsetX: " << offsetX << "; offsetY: " << offsetY << std::endl;

            // Extract the rotation angle for the prediction and then compute the sin and cosine
            float angle = anglesData[x];
            std::cout << "Angle: " << angle << std::endl;
            float cos = std::cos(angle);
            float sin = std::sin(angle);

            // Use the geometry volume to derive the width and height of the bounding box
            float h = xData0[x] + xData2[x];
            float w = xData1[x] + xData3[x];

            // Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            uint64_t endX = (((float)offsetX) + (cos * xData1[x]) + (sin * xData2[x]));
            uint64_t endY = (((float)offsetY) - (sin * xData1[x]) + (cos * xData2[x]));
            uint64_t startX = endX - (uint64_t)w;
            uint64_t startY = endY - (uint64_t)h;
            std::cout << "StartX: " << startX << "; StartY: " << startY << "; EndX: " << endX << "; EndY: " << endY << std::endl;

            boxes.emplace_back(Box{startX, endX, startY, endY, score});

            std::cout << "---------------------------" << std::endl;
        }
    }

    std::cout << "Total findings: " << boxes.size() << std::endl;

    uint64_t outputBatch = boxes.size();
    NODE_ASSERT(outputBatch > 0, "No findings");
    uint64_t byteSize = sizeof(float) * 50 * 200 * 3 * outputBatch;

    uint8_t* data = (uint8_t*)malloc(byteSize);

    std::cout << "Original image size: " << image.size << std::endl;

    for (uint64_t i = 0; i < outputBatch; i++) {
        if (boxes[i].endX >= image.size().width) {
            boxes[i].endX = image.size().width - 1;
        }
        if (boxes[i].endY >= image.size().height) {
            boxes[i].endY = image.size().height - 1;
        }
        uint64_t width = boxes[i].endX - boxes[i].startX;
        uint64_t height = boxes[i].endY - boxes[i].startY;
        cv::Rect myROI(boxes[i].startX, boxes[i].startY, width, height);
        std::cout << myROI << std::endl;
        cv::Mat croppedRef(image, myROI);
        cv::Mat cropped;
        croppedRef.copyTo(cropped);
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(200, 50));
        std::memcpy(((float*)data) + (i * 3 * 200 * 50), resized.data, byteSize / outputBatch);
    }

    *outputsLength = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsLength * sizeof(CustomNodeTensor));
    CustomNodeTensor& resultTensor = **outputs;
    resultTensor.name = TEXT_IMAGES_TENSOR_NAME;
    resultTensor.data = data;
    resultTensor.dataLength = byteSize;
    resultTensor.dimsLength = 4;
    resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsLength * sizeof(uint64_t));
    resultTensor.dims[0] = outputBatch;
    resultTensor.dims[1] = 50;
    resultTensor.dims[2] = 200;
    resultTensor.dims[3] = 3;
    resultTensor.precision = FP32;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsLength = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 1024;
    (*info)[0].dims[2] = 1920;
    (*info)[0].dims[3] = 3;
    (*info)[0].precision = FP32;

    (*info)[1].name = SCORES_TENSOR_NAME;
    (*info)[1].dimsLength = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 256;
    (*info)[1].dims[3] = 480;
    (*info)[1].precision = FP32;

    (*info)[2].name = GEOMETRY_TENSOR_NAME;
    (*info)[2].dimsLength = 4;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[2].dims[0] = 1;
    (*info)[2].dims[1] = 5;
    (*info)[2].dims[2] = 256;
    (*info)[2].dims[3] = 480;
    (*info)[2].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    *infoLength = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = TEXT_IMAGES_TENSOR_NAME;
    (*info)->dimsLength = 4;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 50;
    (*info)->dims[2] = 200;
    (*info)->dims[3] = 3;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr) {
    free(ptr);
    return 0;
}
