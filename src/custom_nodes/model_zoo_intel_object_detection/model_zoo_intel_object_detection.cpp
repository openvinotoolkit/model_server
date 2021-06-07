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
#include "opencv2/opencv.hpp"
#include "utils.hpp"

static constexpr const char* INPUT_IMAGE_TENSOR_NAME = "image";
static constexpr const char* INPUT_DETECTION_TENSOR_NAME = "detection";
static constexpr const char* OUTPUT_IMAGES_TENSOR_NAME = "images";
static constexpr const char* OUTPUT_COORDINATES_TENSOR_NAME = "coordinates";
static constexpr const char* OUTPUT_CONFIDENCES_TENSOR_NAME = "confidences";

bool copy_images_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes, const cv::Mat& originalImage, int targetImageHeight, int targetImageWidth, const std::string& targetImageLayout, bool convertToGrayScale) {
    const uint64_t outputBatch = boxes.size();
    int channels = convertToGrayScale ? 1 : 3;

    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * outputBatch;

    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    if (buffer == nullptr) {
        return false;
    }

    cv::Size targetShape(targetImageWidth, targetImageHeight);
    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Mat image;

        if (!crop_rotate_resize(originalImage, image, boxes[i], 0.0, boxes[i].width, boxes[i].height, targetShape)) {
            std::cout << "box is outside of original image" << std::endl;
            free(buffer);
            return false;
        }
        if (convertToGrayScale) {
            image = apply_grayscale(image);
        }
        if (targetImageLayout == "NCHW") {
            auto imgBuffer = reorder_to_nchw((float*)image.data, image.rows, image.cols, image.channels());
            std::memcpy(buffer + (i * channels * targetImageWidth * targetImageHeight), imgBuffer.data(), byteSize / outputBatch);
        } else {
            std::memcpy(buffer + (i * channels * targetImageWidth * targetImageHeight), image.data, byteSize / outputBatch);
        }
    }

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 5;
    output->dims = (uint64_t*)malloc(output->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output->dims != nullptr, "malloc has failed");
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    if (targetImageLayout == "NCHW") {
        output->dims[2] = channels;
        output->dims[3] = targetImageHeight;
        output->dims[4] = targetImageWidth;
    } else {
        output->dims[2] = targetImageHeight;
        output->dims[3] = targetImageWidth;
        output->dims[4] = channels;
    }
    output->precision = FP32;
    return true;
}

bool copy_coordinates_into_output(struct CustomNodeTensor* output, const std::vector<cv::Vec4f>& detections) {
    const uint64_t outputBatch = detections.size();
    uint64_t byteSize = sizeof(int32_t) * 4 * outputBatch;

    int32_t* buffer = (int32_t*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    if (buffer == nullptr) {
        return false;
    }

    for (size_t i = 0; i < outputBatch; i++) {
        float entry[] = {
            detections[i][0], detections[i][1], detections[i][2], detections[i][3]};
        std::memcpy(buffer + (i * 4), entry, byteSize / outputBatch);
    }

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 3;
    output->dims = (uint64_t*)malloc(output->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output->dims != nullptr, "malloc has failed");
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 4;
    output->precision = FP32;
    return true;
}

bool copy_confidences_into_output(struct CustomNodeTensor* output, const std::vector<float>& confidences) {
    const uint64_t outputBatch = confidences.size();
    uint64_t byteSize = sizeof(float) * outputBatch;

    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    std::memcpy(buffer, confidences.data(), byteSize);

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 3;
    output->dims = (uint64_t*)malloc(output->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output->dims != nullptr, "malloc has failed");
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 1;
    output->precision = FP32;
    return true;
}

void cleanup(CustomNodeTensor& tensor) {
    free(tensor.data);
    free(tensor.dims);
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount) {
    // Parameters reading
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsCount) == "true";
    float confidenceThreshold = get_float_parameter("confidence_threshold", params, paramsCount, -1.0);
    NODE_ASSERT(confidenceThreshold >= 0 && confidenceThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    uint64_t maxOutputBatch = get_int_parameter("max_output_batch", params, paramsCount, 100);
    NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0");
    int filterLabelId = get_int_parameter("filter_label_id", params, paramsCount, -1);
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";

    const CustomNodeTensor* imageTensor = nullptr;
    const CustomNodeTensor* detectionTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, INPUT_IMAGE_TENSOR_NAME) == 0) {
            imageTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, INPUT_DETECTION_TENSOR_NAME) == 0) {
            detectionTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    NODE_ASSERT(imageTensor != nullptr, "Missing input image");
    NODE_ASSERT(detectionTensor != nullptr, "Missing input scores");
    NODE_ASSERT(imageTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(detectionTensor->precision == FP32, "image input is not FP32");

    NODE_ASSERT(imageTensor->dimsCount == 4, "input image shape must have 4 dimensions");
    NODE_ASSERT(imageTensor->dims[0] == 1, "input image batch must be 1");
    NODE_ASSERT(imageTensor->dims[originalImageLayout == "NCHW" ? 1 : 3] == 3, "input image needs to have 3 color channels");

    NODE_ASSERT(detectionTensor->dimsCount == 4, "input detection shape must have 4 dimensions");
    NODE_ASSERT(detectionTensor->dims[0] == 1, "input detection dim[0] must be 1");
    NODE_ASSERT(detectionTensor->dims[1] == 1, "input detection dim[1] must be 1");
    NODE_ASSERT(detectionTensor->dims[2] == 200, "input detection dim[2] must be 200");
    NODE_ASSERT(detectionTensor->dims[3] == 7, "input detection dim[3] must be 7");

    uint64_t _imageHeight = imageTensor->dims[originalImageLayout == "NCHW" ? 2 : 1];
    uint64_t _imageWidth = imageTensor->dims[originalImageLayout == "NCHW" ? 3 : 2];
    NODE_ASSERT(_imageHeight <= std::numeric_limits<int>::max(), "image height is too large");
    NODE_ASSERT(_imageWidth <= std::numeric_limits<int>::max(), "image width is too large");
    int imageHeight = static_cast<int>(_imageHeight);
    int imageWidth = static_cast<int>(_imageWidth);

    if (debugMode) {
        std::cout << "Processing input tensor image resolution: " << cv::Size(imageHeight, imageWidth) << "; expected resolution: " << cv::Size(originalImageHeight, originalImageWidth) << std::endl;
    }

    NODE_ASSERT(imageHeight == originalImageHeight, "original image size parameter differs from original image tensor size");
    NODE_ASSERT(imageWidth == originalImageWidth, "original image size parameter differs from original image tensor size");

    cv::Mat image;
    if (originalImageLayout == "NHWC") {
        image = nhwc_to_mat(imageTensor);
    } else {
        image = nchw_to_mat(imageTensor);
    }

    NODE_ASSERT(image.cols == imageWidth, "Mat generation failed");
    NODE_ASSERT(image.rows == imageHeight, "Mat generation failed");

    uint64_t detectionsCount = detectionTensor->dims[2];
    uint64_t featuresCount = detectionTensor->dims[3];

    std::vector<cv::Rect> boxes;
    std::vector<cv::Vec4f> detections;
    std::vector<float> confidences;

    for (uint64_t i = 0; i < detectionsCount; i++) {
        float* detection = (float*)(detectionTensor->data + (i * featuresCount * sizeof(float)));
        int imageId = static_cast<int>(detection[0]);
        int labelId = static_cast<int>(detection[1]);
        float confidence = detection[2];
        int xMin = static_cast<int>(detection[3] * imageWidth);
        int yMin = static_cast<int>(detection[4] * imageHeight);
        int xMax = static_cast<int>(detection[5] * imageWidth);
        int yMax = static_cast<int>(detection[6] * imageHeight);
        if (imageId == 0 && confidence >= confidenceThreshold) {
            if (filterLabelId != -1 && filterLabelId != labelId) {
                if (debugMode) {
                    std::cout << "Skipping label ID: " << labelId << std::endl;
                }
                continue;
            }
            auto box = cv::Rect(cv::Point(xMin, yMin), cv::Point(xMax, yMax));
            boxes.emplace_back(box);
            detections.emplace_back(detection[3], detection[4], detection[5], detection[6]);
            confidences.emplace_back(confidence);
            if (debugMode) {
                std::cout << "Detection:\nImageID: " << imageId << "; LabelID:" << labelId << "; Confidence:" << confidence << "; Box:" << box << std::endl;
            }
        }
    }

    NODE_ASSERT(boxes.size() == confidences.size(), "boxes and confidences are not equal length");
    if (boxes.size() > maxOutputBatch) {
        boxes.resize(maxOutputBatch);
        confidences.resize(maxOutputBatch);
    }

    *outputsCount = 3;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    NODE_ASSERT((*outputs) != nullptr, "malloc has failed");

    CustomNodeTensor& imagesTensor = (*outputs)[0];
    imagesTensor.name = OUTPUT_IMAGES_TENSOR_NAME;
    if (!copy_images_into_output(&imagesTensor, boxes, image, targetImageHeight, targetImageWidth, targetImageLayout, convertToGrayScale)) {
        free(*outputs);
        return 1;
    }

    CustomNodeTensor& coordinatesTensor = (*outputs)[1];
    coordinatesTensor.name = OUTPUT_COORDINATES_TENSOR_NAME;
    if (!copy_coordinates_into_output(&coordinatesTensor, detections)) {
        cleanup(imagesTensor);
        free(*outputs);
        return 1;
    }

    CustomNodeTensor& confidencesTensor = (*outputs)[2];
    confidencesTensor.name = OUTPUT_CONFIDENCES_TENSOR_NAME;
    if (!copy_confidences_into_output(&confidencesTensor, confidences)) {
        cleanup(imagesTensor);
        cleanup(coordinatesTensor);
        free(*outputs);
        return 1;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");

    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = INPUT_IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    if (originalImageLayout == "NCHW") {
        (*info)[0].dims[1] = 3;
        (*info)[0].dims[2] = originalImageHeight;
        (*info)[0].dims[3] = originalImageWidth;
    } else {
        (*info)[0].dims[1] = originalImageHeight;
        (*info)[0].dims[2] = originalImageWidth;
        (*info)[0].dims[3] = 3;
    }
    (*info)[0].precision = FP32;

    (*info)[1].name = INPUT_DETECTION_TENSOR_NAME;
    (*info)[1].dimsCount = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 200;
    (*info)[1].dims[3] = 7;
    (*info)[1].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount) {
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsCount) == "true";

    *infoCount = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = OUTPUT_IMAGES_TENSOR_NAME;
    (*info)[0].dimsCount = 5;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 0;
    (*info)[0].dims[1] = 1;
    if (targetImageLayout == "NCHW") {
        (*info)[0].dims[2] = convertToGrayScale ? 1 : 3;
        (*info)[0].dims[3] = targetImageHeight;
        (*info)[0].dims[4] = targetImageWidth;
    } else {
        (*info)[0].dims[2] = targetImageHeight;
        (*info)[0].dims[3] = targetImageWidth;
        (*info)[0].dims[4] = convertToGrayScale ? 1 : 3;
    }
    (*info)[0].precision = FP32;

    (*info)[1].name = OUTPUT_COORDINATES_TENSOR_NAME;
    (*info)[1].dimsCount = 3;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 0;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 4;
    (*info)[1].precision = FP32;

    (*info)[2].name = OUTPUT_CONFIDENCES_TENSOR_NAME;
    (*info)[2].dimsCount = 3;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[2].dims) != nullptr, "malloc has failed");
    (*info)[2].dims[0] = 0;
    (*info)[2].dims[1] = 1;
    (*info)[2].dims[2] = 1;
    (*info)[2].precision = FP32;
    return 0;
}

int release(void* ptr) {
    free(ptr);
    return 0;
}
