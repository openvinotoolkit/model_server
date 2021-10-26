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
#include <shared_mutex>
#include <string>
#include <vector>

#include "../../custom_node_interface.h"
#include "../common/buffersqueue.hpp"
#include "../common/customNodeLibraryInternalManager.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"

using CustomNodeLibraryInternalManager = ovms::custom_nodes_common::CustomNodeLibraryInternalManager;
using BuffersQueue = ovms::custom_nodes_common::BuffersQueue;

static constexpr const char* INPUT_IMAGE_TENSOR_NAME = "image";
static constexpr const char* INPUT_DETECTION_TENSOR_NAME = "detection";
static constexpr const char* INPUT_IMAGE_DIMS_NAME = "image_dims";
static constexpr const char* INPUT_DETECTION_DIMS_NAME = "detection_dims";
static constexpr const char* INPUT_TENSOR_INFO_NAME = "input_info";

static constexpr const char* OUTPUT_TENSOR_NAME = "output";
static constexpr const char* OUTPUT_IMAGES_TENSOR_NAME = "images";
static constexpr const char* OUTPUT_COORDINATES_TENSOR_NAME = "coordinates";
static constexpr const char* OUTPUT_CONFIDENCES_TENSOR_NAME = "confidences";
static constexpr const char* OUTPUT_IMAGES_DIMS_NAME = "images_dims";
static constexpr const char* OUTPUT_COORDINATES_DIMS_NAME = "coordinates_dims";
static constexpr const char* OUTPUT_CONFIDENCES_DIMS_NAME = "confidences_dims";
static constexpr const char* OUTPUT_TENSOR_INFO_NAME = "output_info";
static constexpr const char* OUTPUT_COORDINATES_INFO_DIMS_NAME = "coordinates_info_dims";
static constexpr const char* OUTPUT_IMAGES_INFO_DIMS_NAME = "images_info_dims";
static constexpr const char* OUTPUT_CONFIDENCES_INFO_DIMS_NAME = "confidences_info_dims";

static constexpr const int QUEUE_SIZE = 1;

std::shared_mutex internalManagerLock;

void cleanup(CustomNodeTensor& tensor, CustomNodeLibraryInternalManager* internalManager) {
    release(tensor.data, internalManager);
    release(tensor.dims, internalManager);
}

template <typename T>
bool get_buffer(CustomNodeLibraryInternalManager* internalManager, T** buffer, const char* buffersQueueName, uint64_t byte_size) {
    auto buffersQueue = internalManager->getBuffersQueue(buffersQueueName);
    if (!(buffersQueue == nullptr))
        *buffer = static_cast<T*>(buffersQueue->getBuffer());
    if (*buffer == nullptr || buffersQueue == nullptr) {
        *buffer = (T*)malloc(byte_size);
        if (*buffer == nullptr) {
            std::cout << "allocation for buffer: " << buffersQueueName << "FAILED" << std::endl;
            return false;
        }
    }
    return true;
}

bool copy_images_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes, const cv::Mat& originalImage, int targetImageHeight, int targetImageWidth, const std::string& targetImageLayout, bool convertToGrayScale, CustomNodeLibraryInternalManager* internalManager) {
    const uint64_t outputBatch = boxes.size();
    int channels = convertToGrayScale ? 1 : 3;
    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * outputBatch;

    float* buffer = nullptr;
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_IMAGES_TENSOR_NAME, byteSize))
        return false;

    cv::Size targetShape(targetImageWidth, targetImageHeight);
    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Mat image;

        if (!crop_rotate_resize(originalImage, image, boxes[i], 0.0, boxes[i].width, boxes[i].height, targetShape)) {
            std::cout << "box is outside of original image" << std::endl;
            release(buffer, internalManager);
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

    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_IMAGES_DIMS_NAME, 5 * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
    output->dimsCount = 5;
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

bool copy_coordinates_into_output(struct CustomNodeTensor* output, const std::vector<cv::Vec4f>& detections, CustomNodeLibraryInternalManager* internalManager) {
    const uint64_t outputBatch = detections.size();
    uint64_t byteSize = sizeof(int32_t) * 4 * outputBatch;

    int32_t* buffer = nullptr;
    if (!get_buffer<int32_t>(internalManager, &buffer, OUTPUT_COORDINATES_TENSOR_NAME, byteSize))
        return false;
    for (size_t i = 0; i < outputBatch; i++) {
        float entry[] = {
            detections[i][0], detections[i][1], detections[i][2], detections[i][3]};
        std::memcpy(buffer + (i * 4), entry, byteSize / outputBatch);
    }
    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;

    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_COORDINATES_DIMS_NAME, 3 * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
    output->dimsCount = 3;
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 4;
    output->precision = FP32;
    return true;
}

bool copy_confidences_into_output(struct CustomNodeTensor* output, const std::vector<float>& confidences, CustomNodeLibraryInternalManager* internalManager) {
    const uint64_t outputBatch = confidences.size();
    uint64_t byteSize = sizeof(float) * outputBatch;

    float* buffer = nullptr;
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_CONFIDENCES_TENSOR_NAME, byteSize))
        return false;
    std::memcpy(buffer, confidences.data(), byteSize);
    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;

    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_CONFIDENCES_DIMS_NAME, 3 * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
    output->dimsCount = 3;
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 1;
    output->precision = FP32;
    return true;
}

int initializeInternalManager(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // creating InternalManager instance
    std::unique_ptr<CustomNodeLibraryInternalManager> internalManager = std::make_unique<CustomNodeLibraryInternalManager>();
    NODE_ASSERT(internalManager != nullptr, "internalManager allocation failed");

    // reading parameters to determine size of pre-allocated buffers
    uint64_t maxOutputBatch = get_int_parameter("max_output_batch", params, paramsCount, 100);
    NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsCount) == "true";
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");

    // creating BuffersQueues for output: images
    int channels = convertToGrayScale ? 1 : 3;
    uint64_t imagesByteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * maxOutputBatch;
    imagesByteSize = imagesByteSize;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_IMAGES_TENSOR_NAME, imagesByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_IMAGES_DIMS_NAME, 5 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output: coordinates
    uint64_t coordinatesByteSize = sizeof(int32_t) * 4 * maxOutputBatch;
    coordinatesByteSize = coordinatesByteSize;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_TENSOR_NAME, coordinatesByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output: confidences
    uint64_t confidenceByteSize = sizeof(float) * maxOutputBatch;
    confidenceByteSize = confidenceByteSize;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCES_TENSOR_NAME, confidenceByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCES_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for inputs
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_IMAGE_DIMS_NAME, 4 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_DETECTION_DIMS_NAME, 4 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output tensor
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_NAME, 3 * sizeof(CustomNodeTensor), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for info tensors
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_TENSOR_INFO_NAME, 2 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_INFO_NAME, 3 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for outputs dims in getOutputsInfo
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_IMAGES_INFO_DIMS_NAME, 5 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_INFO_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCES_INFO_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    *customNodeLibraryInternalManager = internalManager.release();
    return 0;
}

int reinitializeInternalManagerIfNeccessary(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // reading parameters to determine new sizes of pre-allocated buffers
    uint64_t maxOutputBatch = get_int_parameter("max_output_batch", params, paramsCount, 100);
    NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsCount) == "true";
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");

    std::unique_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(*customNodeLibraryInternalManager);

    // replace buffers if their sizes need to change
    int channels = convertToGrayScale ? 1 : 3;
    uint64_t imagesByteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * maxOutputBatch;
    NODE_ASSERT(internalManager->recreateBuffersQueue(OUTPUT_IMAGES_TENSOR_NAME, imagesByteSize, QUEUE_SIZE), "buffer recreation failed");

    uint64_t coordinatesByteSize = sizeof(int32_t) * 4 * maxOutputBatch;
    NODE_ASSERT(internalManager->recreateBuffersQueue(OUTPUT_COORDINATES_TENSOR_NAME, coordinatesByteSize, QUEUE_SIZE), "buffer recreation failed");

    uint64_t confidenceByteSize = sizeof(float) * maxOutputBatch;
    NODE_ASSERT(internalManager->recreateBuffersQueue(OUTPUT_CONFIDENCES_TENSOR_NAME, confidenceByteSize, QUEUE_SIZE), "buffer recreation failed");

    return 0;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    std::cout << "Started initialize of new API" << std::endl;
    auto status = 0;
    if (*customNodeLibraryInternalManager == nullptr) {
        status = initializeInternalManager(customNodeLibraryInternalManager, params, paramsCount);
    } else {
        status = reinitializeInternalManagerIfNeccessary(customNodeLibraryInternalManager, params, paramsCount);
    }
    NODE_ASSERT(status == 0, "initialize failed");
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    // deallocate InternalManager and its contents
    if (customNodeLibraryInternalManager != nullptr) {
        CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
        delete internalManager;
    }
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
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

    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    *outputsCount = 3;
    if (!get_buffer<struct CustomNodeTensor>(internalManager, outputs, OUTPUT_TENSOR_NAME, 3 * sizeof(CustomNodeTensor))) {
        return 1;
    }

    CustomNodeTensor& imagesTensor = (*outputs)[0];
    imagesTensor.name = OUTPUT_IMAGES_TENSOR_NAME;
    if (!copy_images_into_output(&imagesTensor, boxes, image, targetImageHeight, targetImageWidth, targetImageLayout, convertToGrayScale, internalManager)) {
        release(*outputs, internalManager);
        return 1;
    }

    CustomNodeTensor& coordinatesTensor = (*outputs)[1];
    coordinatesTensor.name = OUTPUT_COORDINATES_TENSOR_NAME;
    if (!copy_coordinates_into_output(&coordinatesTensor, detections, internalManager)) {
        cleanup(imagesTensor, internalManager);
        release(*outputs, internalManager);
        return 1;
    }

    CustomNodeTensor& confidencesTensor = (*outputs)[2];
    confidencesTensor.name = OUTPUT_CONFIDENCES_TENSOR_NAME;
    if (!copy_confidences_into_output(&confidencesTensor, confidences, internalManager)) {
        cleanup(coordinatesTensor, internalManager);
        cleanup(imagesTensor, internalManager);
        release(*outputs, internalManager);
        return 1;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");

    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    *infoCount = 2;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, INPUT_TENSOR_INFO_NAME, 2 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }

    (*info)[0].name = INPUT_IMAGE_TENSOR_NAME;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[0].dims), INPUT_IMAGE_DIMS_NAME, 4 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)[0].dimsCount = 4;
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
    if (!get_buffer<uint64_t>(internalManager, &((*info)[1].dims), INPUT_DETECTION_DIMS_NAME, 4 * sizeof(uint64_t))) {
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
    (*info)[1].dimsCount = 4;
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 200;
    (*info)[1].dims[3] = 7;
    (*info)[1].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");
    bool convertToGrayScale = get_string_parameter("convert_to_gray_scale", params, paramsCount) == "true";

    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    *infoCount = 3;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, OUTPUT_TENSOR_INFO_NAME, 3 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }

    (*info)[0].name = OUTPUT_IMAGES_TENSOR_NAME;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[0].dims), OUTPUT_IMAGES_INFO_DIMS_NAME, 5 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)[0].dimsCount = 5;
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
    if (!get_buffer<uint64_t>(internalManager, &((*info)[1].dims), OUTPUT_COORDINATES_INFO_DIMS_NAME, 3 * sizeof(uint64_t))) {
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
    (*info)[1].dimsCount = 3;
    (*info)[1].dims[0] = 0;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 4;
    (*info)[1].precision = FP32;

    (*info)[2].name = OUTPUT_CONFIDENCES_TENSOR_NAME;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[2].dims), OUTPUT_CONFIDENCES_INFO_DIMS_NAME, 3 * sizeof(uint64_t))) {
        release((*info)[1].dims, internalManager);
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
    (*info)[2].dimsCount = 3;
    (*info)[2].dims[0] = 0;
    (*info)[2].dims[1] = 1;
    (*info)[2].dims[2] = 1;
    (*info)[2].precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    if (!internalManager->releaseBuffer(ptr)) {
        free(ptr);
        return 0;
    }
    return 0;
}
