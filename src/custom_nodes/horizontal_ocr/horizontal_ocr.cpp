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
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#include "../common/buffersqueue.hpp"
#include "../common/custom_node_library_internal_manager.hpp"
#include "opencv2/opencv.hpp"

using CustomNodeLibraryInternalManager = ovms::custom_nodes_common::CustomNodeLibraryInternalManager;

static constexpr const char* INPUT_IMAGE_TENSOR_NAME = "image";
static constexpr const char* INPUT_GEOMETRY_TENSOR_NAME = "boxes";
static constexpr const char* INPUT_TENSOR_INFO_NAME = "input_info";
static constexpr const char* INPUT_IMAGE_INFO_DIMS_NAME = "image_info_dims";
static constexpr const char* INPUT_GEOMETRY_INFO_DIMS_NAME = "boxes_info_dims";

static constexpr const char* OUTPUT_TENSOR_NAME = "output";
static constexpr const char* OUTPUT_CONFIDENCE_TENSOR_NAME = "confidence_levels";
static constexpr const char* OUTPUT_TEXT_IMAGES_TENSOR_NAME = "text_images";
static constexpr const char* OUTPUT_COORDINATES_TENSOR_NAME = "text_coordinates";
static constexpr const char* OUTPUT_CONFIDENCE_DIMS_NAME = "confidence_levels_dims";
static constexpr const char* OUTPUT_TEXT_IMAGES_DIMS_NAME = "text_images_dims";
static constexpr const char* OUTPUT_COORDINATES_DIMS_NAME = "text_coordinates_dims";
static constexpr const char* OUTPUT_TENSOR_INFO_NAME = "output_info";
static constexpr const char* OUTPUT_CONFIDENCE_INFO_DIMS_NAME = "confidence_levels_info_dims";
static constexpr const char* OUTPUT_TEXT_IMAGES_INFO_DIMS_NAME = "text_images_info_dims";
static constexpr const char* OUTPUT_COORDINATES_INFO_DIMS_NAME = "text_coordinates_info_dims";

static constexpr const int QUEUE_SIZE = 1;

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
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_TEXT_IMAGES_TENSOR_NAME, byteSize))
        return false;

    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Size targetShape(targetImageWidth, targetImageHeight);
        cv::Mat image;

        cv::Mat cropped = originalImage(boxes[i]);
        cv::resize(cropped, image, targetShape);
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
    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_TEXT_IMAGES_DIMS_NAME, output->dimsCount * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
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

bool copy_coordinates_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes, CustomNodeLibraryInternalManager* internalManager) {
    const uint64_t outputBatch = boxes.size();
    uint64_t byteSize = sizeof(int32_t) * 4 * outputBatch;

    int32_t* buffer = nullptr;
    if (!get_buffer<int32_t>(internalManager, &buffer, OUTPUT_COORDINATES_TENSOR_NAME, byteSize))
        return false;
    for (uint64_t i = 0; i < outputBatch; i++) {
        int32_t entry[] = {
            boxes[i].x,
            boxes[i].y,
            boxes[i].width,
            boxes[i].height};
        std::memcpy(buffer + (i * 4), entry, byteSize / outputBatch);
    }
    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 3;
    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_COORDINATES_DIMS_NAME, output->dimsCount * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 4;
    output->precision = I32;
    return true;
}

bool copy_scores_into_output(struct CustomNodeTensor* output, const std::vector<float>& scores, CustomNodeLibraryInternalManager* internalManager) {
    const uint64_t outputBatch = scores.size();
    uint64_t byteSize = sizeof(float) * outputBatch;

    float* buffer = nullptr;
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_CONFIDENCE_TENSOR_NAME, byteSize))
        return false;
    std::memcpy(buffer, scores.data(), byteSize);

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 3;
    if (!get_buffer<uint64_t>(internalManager, &(output->dims), OUTPUT_CONFIDENCE_DIMS_NAME, output->dimsCount * sizeof(uint64_t))) {
        release(buffer, internalManager);
        return false;
    }
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 1;
    output->precision = FP32;
    return true;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
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

    // creating BuffersQueues for output: text_images
    int channels = convertToGrayScale ? 1 : 3;
    uint64_t textImagesByteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * maxOutputBatch;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TEXT_IMAGES_TENSOR_NAME, textImagesByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TEXT_IMAGES_DIMS_NAME, 5 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output: coordinates
    uint64_t coordinatesByteSize = sizeof(int32_t) * 4 * maxOutputBatch;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_TENSOR_NAME, coordinatesByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output: confidence
    uint64_t confidenceByteSize = sizeof(float) * maxOutputBatch;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCE_TENSOR_NAME, confidenceByteSize, QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCE_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for output tensor
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_NAME, 3 * sizeof(CustomNodeTensor), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for info tensors
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_TENSOR_INFO_NAME, 2 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_INFO_NAME, 3 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for inputs dims in getInputsInfo
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_IMAGE_INFO_DIMS_NAME, 4 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_GEOMETRY_INFO_DIMS_NAME, 2 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    // creating BuffersQueues for outputs dims in getOutputsInfo
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TEXT_IMAGES_INFO_DIMS_NAME, 5 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_COORDINATES_INFO_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_CONFIDENCE_INFO_DIMS_NAME, 3 * sizeof(uint64_t), QUEUE_SIZE), "buffer creation failed");

    *customNodeLibraryInternalManager = internalManager.release();
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    // deallocate InternalManager and its contents
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    delete internalManager;
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
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";

    const CustomNodeTensor* imageTensor = nullptr;
    const CustomNodeTensor* boxesTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, INPUT_IMAGE_TENSOR_NAME) == 0) {
            imageTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, INPUT_GEOMETRY_TENSOR_NAME) == 0) {
            boxesTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    NODE_ASSERT(imageTensor != nullptr, "Missing input image");
    NODE_ASSERT(boxesTensor != nullptr, "Missing input boxes");
    NODE_ASSERT(imageTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(boxesTensor->precision == FP32, "image input is not FP32");

    NODE_ASSERT(imageTensor->dimsCount == 4, "input image shape must have 4 dimensions");
    NODE_ASSERT(imageTensor->dims[0] == 1, "input image batch must be 1");
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

    uint64_t _numDetections = boxesTensor->dims[0];
    uint64_t _numItems = boxesTensor->dims[1];
    NODE_ASSERT(boxesTensor->dims[0] == 100, "boxes has dim 0 not equal to 100");
    NODE_ASSERT(boxesTensor->dims[1] == 5, "boxes has dim 1 not equal to 5");
    int numDetections = static_cast<int>(_numDetections);
    int numItems = static_cast<int>(_numItems);

    std::vector<cv::Rect> rects;
    std::vector<float> scores;

    for (int i = 0; i < numDetections; i++) {
        float* boxesData = (float*)boxesTensor->data + (i * numItems);
        float score = boxesData[4];
        if (score < confidenceThreshold) {
            continue;
        }

        if (debugMode) {
            std::cout << "Found confidence: " << score << std::endl;
        }

        float x1 = boxesData[0];
        float y1 = boxesData[1];
        float x2 = boxesData[2];
        float y2 = boxesData[3];
        NODE_ASSERT(x2 > x1, "detected box width must be greater than 0");
        NODE_ASSERT(y2 > y1, "detected box height must be greater than 0");

        rects.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.emplace_back(score);
    }

    NODE_ASSERT(rects.size() == scores.size(), "rects and scores are not equal length");
    if (rects.size() > maxOutputBatch) {
        rects.resize(maxOutputBatch);
        scores.resize(maxOutputBatch);
    }

    if (debugMode)
        std::cout << "Total findings: " << rects.size() << std::endl;

    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    std::shared_lock lock(internalManager->getInternalManagerLock());

    *outputsCount = 3;
    if (!get_buffer<struct CustomNodeTensor>(internalManager, outputs, OUTPUT_TENSOR_NAME, *outputsCount * sizeof(CustomNodeTensor))) {
        return 1;
    }

    CustomNodeTensor& textImagesTensor = (*outputs)[0];
    textImagesTensor.name = OUTPUT_TEXT_IMAGES_TENSOR_NAME;
    if (!copy_images_into_output(&textImagesTensor, rects, image, targetImageHeight, targetImageWidth, targetImageLayout, convertToGrayScale, internalManager)) {
        release(*outputs, internalManager);
        return 1;
    }

    CustomNodeTensor& coordinatesTensor = (*outputs)[1];
    coordinatesTensor.name = OUTPUT_COORDINATES_TENSOR_NAME;
    if (!copy_coordinates_into_output(&coordinatesTensor, rects, internalManager)) {
        cleanup(textImagesTensor, internalManager);
        release(*outputs, internalManager);
        return 1;
    }

    CustomNodeTensor& confidenceTensor = (*outputs)[2];
    confidenceTensor.name = OUTPUT_CONFIDENCE_TENSOR_NAME;
    if (!copy_scores_into_output(&confidenceTensor, scores, internalManager)) {
        cleanup(coordinatesTensor, internalManager);
        cleanup(textImagesTensor, internalManager);
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

    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    std::shared_lock lock(internalManager->getInternalManagerLock());

    *infoCount = 2;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, INPUT_TENSOR_INFO_NAME, *infoCount * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }

    (*info)[0].name = INPUT_IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[0].dims), INPUT_IMAGE_INFO_DIMS_NAME, (*info)[0].dimsCount * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
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

    (*info)[1].name = INPUT_GEOMETRY_TENSOR_NAME;
    (*info)[1].dimsCount = 2;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[1].dims), INPUT_GEOMETRY_INFO_DIMS_NAME, (*info)[1].dimsCount * sizeof(uint64_t))) {
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
    (*info)[1].dims[0] = 100;
    (*info)[1].dims[1] = 5;
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

    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    std::shared_lock lock(internalManager->getInternalManagerLock());

    *infoCount = 3;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, OUTPUT_TENSOR_INFO_NAME, *infoCount * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }

    (*info)[0].name = OUTPUT_TEXT_IMAGES_TENSOR_NAME;
    (*info)[0].dimsCount = 5;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[0].dims), OUTPUT_TEXT_IMAGES_INFO_DIMS_NAME, (*info)[0].dimsCount * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
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
    if (!get_buffer<uint64_t>(internalManager, &((*info)[1].dims), OUTPUT_COORDINATES_INFO_DIMS_NAME, (*info)[1].dimsCount * sizeof(uint64_t))) {
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
    (*info)[1].dims[0] = 0;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 4;
    (*info)[1].precision = I32;

    (*info)[2].name = OUTPUT_CONFIDENCE_TENSOR_NAME;
    (*info)[2].dimsCount = 3;
    if (!get_buffer<uint64_t>(internalManager, &((*info)[2].dims), OUTPUT_CONFIDENCE_INFO_DIMS_NAME, (*info)[2].dimsCount * sizeof(uint64_t))) {
        release((*info)[1].dims, internalManager);
        release((*info)[0].dims, internalManager);
        release(*info, internalManager);
        return 1;
    }
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
