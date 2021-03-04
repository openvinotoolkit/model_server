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
#include <iostream>
#include <string>
#include <vector>

#include "../../custom_node_interface.h"
#include "nms.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* SCORES_TENSOR_NAME = "scores";
static constexpr const char* GEOMETRY_TENSOR_NAME = "geometry";
static constexpr const char* TEXT_IMAGES_TENSOR_NAME = "text_images";
static constexpr const char* COORDINATES_TENSOR_NAME = "text_coordinates";
static constexpr const char* CONFIDENCE_TENSOR_NAME = "confidence_levels";

int extract_text_images_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes, const cv::Mat& originalImage, int targetImageHeight, int targetImageWidth, bool convertToGrayScale, const std::string& imageLayout) {
    uint64_t outputBatch = boxes.size();
    int channels = convertToGrayScale ? 1 : 3;

    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * outputBatch;

    float* buffer = (float*)malloc(byteSize);

    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Size taretShape(targetImageWidth, targetImageHeight);
        cv::Mat image = crop_and_resize(originalImage, boxes[i], taretShape);
        if (convertToGrayScale) {
            image = apply_grayscale(image);
        }
        if (imageLayout == "NCHW") {
            auto imgBuffer = reorder_to_nchw((float*)image.data, image.rows, image.cols, image.channels());
            std::memcpy(buffer + (i * channels * targetImageWidth * targetImageHeight), imgBuffer.data(), byteSize / outputBatch);
        } else {
            std::memcpy(buffer + (i * channels * targetImageWidth * targetImageHeight), image.data, byteSize / outputBatch);
        }
    }

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataLength = byteSize;
    output->dimsLength = 5;
    output->dims = (uint64_t*)malloc(output->dimsLength * sizeof(uint64_t));
    output->dims[0] = 1;
    output->dims[1] = outputBatch;
    if (imageLayout == "NCHW") {
        output->dims[2] = channels;
        output->dims[3] = targetImageHeight;
        output->dims[4] = targetImageWidth;
    } else {
        output->dims[2] = targetImageHeight;
        output->dims[3] = targetImageWidth;
        output->dims[4] = channels;
    }
    output->precision = FP32;
    return 0;
}

int extract_coordinates_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes) {
    uint64_t outputBatch = boxes.size();
    uint64_t byteSize = sizeof(int32_t) * 4 * outputBatch;

    int32_t* buffer = (int32_t*)malloc(byteSize);

    for (uint64_t i = 0; i < outputBatch; i++) {
        int32_t entry[] = {
            boxes[i].x,
            boxes[i].y,
            boxes[i].width,
            boxes[i].height};
        std::memcpy(buffer + (i * 4), entry, byteSize / outputBatch);
    }
    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataLength = byteSize;
    output->dimsLength = 3;
    output->dims = (uint64_t*)malloc(output->dimsLength * sizeof(uint64_t));
    output->dims[0] = 1;
    output->dims[1] = outputBatch;
    output->dims[2] = 4;
    output->precision = I32;
    return 0;
}

int extract_scores_into_output(struct CustomNodeTensor* output, const std::vector<float>& scores) {
    uint64_t outputBatch = scores.size();
    uint64_t byteSize = sizeof(float) * outputBatch;

    float* buffer = (float*)malloc(byteSize);
    std::memcpy(buffer, scores.data(), byteSize);

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataLength = byteSize;
    output->dimsLength = 3;
    output->dims = (uint64_t*)malloc(output->dimsLength * sizeof(uint64_t));
    output->dims[0] = 1;
    output->dims[1] = outputBatch;
    output->dims[2] = 1;
    output->precision = FP32;
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    // Parameters reading
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsLength, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsLength, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 4) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 4) == 0, "original image width must be divisible by 4");
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsLength, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsLength, -1);
    NODE_ASSERT(targetImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "original image width must be larger than 0");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsLength) == "true";
    float confidenceThreshold = get_float_parameter("confidence_threshold", params, paramsLength, -1.0);
    NODE_ASSERT(confidenceThreshold >= 0 && confidenceThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    float overlapThreshold = get_float_parameter("overlap_threshold", params, paramsLength, 0.3);
    NODE_ASSERT(overlapThreshold >= 0 && overlapThreshold <= 1.0, "non max suppression filtering overlap threshold must be in 0-1 range");
    int maxOutputBatch = get_int_parameter("max_output_batch", params, paramsLength, 100);
    NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0");
    bool debugMode = get_string_parameter("debug", params, paramsLength) == "true";
    std::string imageLayout = get_string_parameter("image_layout", params, paramsLength, "NCHW");
    NODE_ASSERT(imageLayout == "NCHW" || imageLayout == "NHWC", "image layout can be either NCHW or NHWC for both inputs and outputs");

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

    uint64_t imageHeight, imageWidth;
    if (imageLayout == "NCHW") {
        imageHeight = imageTensor->dims[2];
        imageWidth = imageTensor->dims[3];
    } else {
        imageHeight = imageTensor->dims[1];
        imageWidth = imageTensor->dims[2];
    }

    if (debugMode) {
        std::cout << "Processing input tensor image resolution: " << cv::Size(imageHeight, imageWidth) << "; expected resolution: " << cv::Size(originalImageHeight, originalImageWidth) << std::endl;
    }

    NODE_ASSERT(imageHeight == originalImageHeight, "original image size parameter differs from original image tensor size");
    NODE_ASSERT(imageWidth == originalImageWidth, "original image size parameter differs from original image tensor size");

    cv::Mat image;
    if (imageLayout == "NCHW") {
        image = nchw_to_mat(imageTensor);
    } else {
        image = nhwc_to_mat(imageTensor);
    }

    NODE_ASSERT(image.cols == imageWidth, "Mat generation failed");
    NODE_ASSERT(image.rows == imageHeight, "Mat generation failed");

    uint64_t numRows = scoresTensor->dims[2];
    uint64_t numCols = scoresTensor->dims[3];

    NODE_ASSERT(scoresTensor->dims[1] == 1, "scores has dim 1 not equal to 1");
    NODE_ASSERT(geometryTensor->dims[1] == 5, "geometry has dim 1 not equal to 5");
    NODE_ASSERT(scoresTensor->dims[2] == geometryTensor->dims[2], "scores and geometry has not equal dim 2");
    NODE_ASSERT(scoresTensor->dims[3] == geometryTensor->dims[3], "scores and geometry has not equal dim 3");

    NODE_ASSERT((numRows * 4) == imageHeight, "image is not x4 larger than score/geometry data");
    NODE_ASSERT((numCols * 4) == imageWidth, "image is not x4 larger than score/geometry data");

    std::vector<cv::Rect> rects;
    std::vector<float> scores;

    // Extract the scores (probabilities), followed by the geometrical data used to derive potential bounding box coordinates that surround text
    for (uint64_t y = 0; y < numRows; y++) {
        float* scoresData = (float*)scoresTensor->data + (y * numCols);
        float* xData0 = (float*)geometryTensor->data + ((0 * numRows * numCols) + (y * numCols));
        float* xData1 = (float*)geometryTensor->data + ((1 * numRows * numCols) + (y * numCols));
        float* xData2 = (float*)geometryTensor->data + ((2 * numRows * numCols) + (y * numCols));
        float* xData3 = (float*)geometryTensor->data + ((3 * numRows * numCols) + (y * numCols));
        float* anglesData = (float*)geometryTensor->data + ((4 * numRows * numCols) + (y * numCols));

        for (uint64_t x = 0; x < numCols; x++) {
            float score = scoresData[x];
            // If our score does not have sufficient probability, ignore it
            if (score < confidenceThreshold) {
                continue;
            }

            if (debugMode)
                std::cout << "Found confidence: " << scoresData[x] << std::endl;

            // Compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            uint64_t offsetX = x * 4;
            uint64_t offsetY = y * 4;

            // Extract the rotation angle for the prediction and then compute the sin and cosine
            float angle = anglesData[x];

            if (debugMode)
                std::cout << "Angle: " << angle << std::endl;
            float cos = std::cos(angle);
            float sin = std::sin(angle);

            // Use the geometry volume to derive the width and height of the bounding box
            float h = xData0[x] + xData2[x];
            float w = xData1[x] + xData3[x];

            cv::Point2i p2{offsetX + (cos * xData1[x] + sin * xData2[x]), offsetY + (-sin * xData1[x] + cos * xData2[x])};
            cv::Point2i p1{-sin * h + p2.x, -cos * h + p2.y};
            cv::Point2i p3{-cos * w + p2.x, sin * w + p2.y};
            cv::Point2i p4{p3.x + p1.x - p2.x, p3.y + p1.y - p2.y};

            int x1 = std::min(std::min(std::min(p2.x, p1.x), p3.x), p4.x);
            int x2 = std::max(std::max(std::max(p2.x, p1.x), p3.x), p4.x);
            int y1 = std::min(std::min(std::min(p2.y, p1.y), p3.y), p4.y);
            int y2 = std::max(std::max(std::max(p2.y, p1.y), p3.y), p4.y);

            if (debugMode) {
                std::cout << "Angled polygon coordinates: " << std::endl;
                std::cout << p4 << p3 << p1 << p2 << std::endl;

                std::cout << "Polygon bounding box with no rotation: " << std::endl;
                std::cout << cv::Point2i(x1, y1) << cv::Point2i(x2, y2) << std::endl;
                std::cout << "---------------------------" << std::endl;
            }

            rects.emplace_back(x1, y1, x2 - x1 - 1, y2 - y1 - 1);
            scores.emplace_back(score);
        }
    }

    if (debugMode)
        std::cout << "Total findings: " << rects.size() << std::endl;

    std::vector<cv::Rect> filteredBoxes;
    std::vector<float> filteredScores;
    nms2(rects, scores, filteredBoxes, filteredScores, overlapThreshold);
    NODE_ASSERT(filteredBoxes.size() == filteredScores.size(), "filtered boxes and scores are not equal length");
    if (filteredBoxes.size() > maxOutputBatch) {
        filteredBoxes.resize(maxOutputBatch);
        filteredScores.resize(maxOutputBatch);
    }

    if (debugMode)
        std::cout << "Total findings after NMS2 (non max suppression) filter: " << filteredBoxes.size() << std::endl;

    *outputsLength = 3;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsLength * sizeof(CustomNodeTensor));

    CustomNodeTensor& textImagesTensor = (*outputs)[0];
    textImagesTensor.name = TEXT_IMAGES_TENSOR_NAME;
    if (extract_text_images_into_output(&textImagesTensor, filteredBoxes, image, targetImageHeight, targetImageWidth, convertToGrayScale, imageLayout)) {
        return 1;
    }

    CustomNodeTensor& coordinatesTensor = (*outputs)[1];
    coordinatesTensor.name = COORDINATES_TENSOR_NAME;
    if (extract_coordinates_into_output(&coordinatesTensor, filteredBoxes)) {
        return 1;
    }

    CustomNodeTensor& confidenceTensor = (*outputs)[2];
    confidenceTensor.name = CONFIDENCE_TENSOR_NAME;
    if (extract_scores_into_output(&confidenceTensor, filteredScores)) {
        return 1;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsLength, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsLength, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 4) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 4) == 0, "original image width must be divisible by 4");
    std::string imageLayout = get_string_parameter("image_layout", params, paramsLength, "NCHW");
    NODE_ASSERT(imageLayout == "NCHW" || imageLayout == "NHWC", "image layout can be either NCHW or NHWC for both inputs and outputs");

    *infoLength = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsLength = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    if (imageLayout == "NCHW") {
        (*info)[0].dims[1] = 3;
        (*info)[0].dims[2] = originalImageHeight;
        (*info)[0].dims[3] = originalImageWidth;
    } else {
        (*info)[0].dims[1] = originalImageHeight;
        (*info)[0].dims[2] = originalImageWidth;
        (*info)[0].dims[3] = 3;
    }
    (*info)[0].precision = FP32;

    (*info)[1].name = SCORES_TENSOR_NAME;
    (*info)[1].dimsLength = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = originalImageHeight / 4;
    (*info)[1].dims[3] = originalImageWidth / 4;
    (*info)[1].precision = FP32;

    (*info)[2].name = GEOMETRY_TENSOR_NAME;
    (*info)[2].dimsLength = 4;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[2].dims[0] = 1;
    (*info)[2].dims[1] = 5;
    (*info)[2].dims[2] = originalImageHeight / 4;
    (*info)[2].dims[3] = originalImageWidth / 4;
    (*info)[2].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoLength, const struct CustomNodeParam* params, int paramsLength) {
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsLength, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsLength, -1);
    NODE_ASSERT(targetImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "original image width must be larger than 0");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsLength) == "true";
    std::string imageLayout = get_string_parameter("image_layout", params, paramsLength, "NCHW");
    NODE_ASSERT(imageLayout == "NCHW" || imageLayout == "NHWC", "image layout can be either NCHW or NHWC for both inputs and outputs");

    *infoLength = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoLength * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = TEXT_IMAGES_TENSOR_NAME;
    (*info)[0].dimsLength = 5;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 0;
    if (imageLayout == "NCHW") {
        (*info)[0].dims[2] = convertToGrayScale ? 1 : 3;
        (*info)[0].dims[3] = targetImageHeight;
        (*info)[0].dims[4] = targetImageWidth;
    } else {
        (*info)[0].dims[2] = targetImageHeight;
        (*info)[0].dims[3] = targetImageWidth;
        (*info)[0].dims[4] = convertToGrayScale ? 1 : 3;
    }
    (*info)[0].precision = FP32;

    (*info)[1].name = COORDINATES_TENSOR_NAME;
    (*info)[1].dimsLength = 3;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 0;
    (*info)[1].dims[2] = 5;
    (*info)[1].precision = I32;

    (*info)[2].name = CONFIDENCE_TENSOR_NAME;
    (*info)[2].dimsLength = 3;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsLength * sizeof(uint64_t));
    (*info)[2].dims[0] = 1;
    (*info)[2].dims[1] = 0;
    (*info)[2].dims[2] = 1;
    (*info)[2].precision = FP32;
    return 0;
}

int release(void* ptr) {
    free(ptr);
    return 0;
}
