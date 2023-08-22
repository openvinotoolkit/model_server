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
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#include "nms.hpp"
#include "opencv2/opencv.hpp"

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* SCORES_TENSOR_NAME = "scores";
static constexpr const char* GEOMETRY_TENSOR_NAME = "geometry";
static constexpr const char* TEXT_IMAGES_TENSOR_NAME = "text_images";
static constexpr const char* COORDINATES_TENSOR_NAME = "text_coordinates";
static constexpr const char* CONFIDENCE_TENSOR_NAME = "confidence_levels";

struct BoxMetadata {
    float angle;
    float originalWidth;
    float originalHeight;
};

static bool copy_images_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes, const std::vector<BoxMetadata>& metadata, const cv::Mat& originalImage, int targetImageHeight, int targetImageWidth, const std::string& targetImageLayout, bool convertToGrayScale, int rotationAngleThreshold) {
    const uint64_t outputBatch = boxes.size();
    int channels = convertToGrayScale ? 1 : 3;

    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * outputBatch;

    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");

    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Size targetShape(targetImageWidth, targetImageHeight);
        cv::Mat image;
        float degree = metadata[i].angle * (180.0 / M_PI);

        if (!crop_rotate_resize(originalImage, image, boxes[i], (abs(degree) > rotationAngleThreshold) ? -degree : 0.0, metadata[i].originalWidth, metadata[i].originalHeight, targetShape)) {
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

static bool copy_boxes_into_output(struct CustomNodeTensor* output, const std::vector<cv::Rect>& boxes) {
    const uint64_t outputBatch = boxes.size();
    uint64_t byteSize = sizeof(int32_t) * 4 * outputBatch;

    int32_t* buffer = (int32_t*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");

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
    output->dims = (uint64_t*)malloc(output->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output->dims != nullptr, "malloc has failed");
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = 4;
    output->precision = I32;
    return true;
}

static bool copy_confidences_into_output(struct CustomNodeTensor* output, const std::vector<float>& confidences) {
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

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // Parameters reading
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 4) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 4) == 0, "original image width must be divisible by 4");
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
    float overlapThreshold = get_float_parameter("overlap_threshold", params, paramsCount, 0.3);
    NODE_ASSERT(overlapThreshold >= 0 && overlapThreshold <= 1.0, "non max suppression filtering overlap threshold must be in 0-1 range");
    uint64_t maxOutputBatch = get_int_parameter("max_output_batch", params, paramsCount, 100);
    NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0");
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";
    float boxWidthAdjustment = get_float_parameter("box_width_adjustment", params, paramsCount, 0.0);
    float boxHeightAdjustment = get_float_parameter("box_height_adjustment", params, paramsCount, 0.0);
    NODE_ASSERT(boxWidthAdjustment >= 0.0, "box width adjustment must be positive");
    NODE_ASSERT(boxHeightAdjustment >= 0.0, "box height adjustment must be positive");
    int rotationAngleThreshold = get_int_parameter("rotation_angle_threshold", params, paramsCount, 20);
    NODE_ASSERT(rotationAngleThreshold >= 0, "rotation angle threshold must be positive");

    const CustomNodeTensor* imageTensor = nullptr;
    const CustomNodeTensor* scoresTensor = nullptr;
    const CustomNodeTensor* geometryTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
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

    NODE_ASSERT(imageTensor != nullptr, "Missing input image");
    NODE_ASSERT(scoresTensor != nullptr, "Missing input scores");
    NODE_ASSERT(geometryTensor != nullptr, "Missing input geometry");
    NODE_ASSERT(imageTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(scoresTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(geometryTensor->precision == FP32, "image input is not FP32");

    NODE_ASSERT(imageTensor->dimsCount == 4, "input image shape must have 4 dimensions");
    NODE_ASSERT(imageTensor->dims[0] == 1, "input image batch must be 1");
    uint64_t _imageHeight = imageTensor->dims[originalImageLayout == "NCHW" ? 2 : 1];
    uint64_t _imageWidth = imageTensor->dims[originalImageLayout == "NCHW" ? 3 : 2];
    NODE_ASSERT(_imageHeight <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image height is too large");
    NODE_ASSERT(_imageWidth <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image width is too large");
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

    uint64_t _numRows = scoresTensor->dims[1];
    uint64_t _numCols = scoresTensor->dims[2];
    NODE_ASSERT(_numRows <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "score  rows is too large");
    NODE_ASSERT(_numCols <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "score columns is too large");
    int numRows = static_cast<int>(_numRows);
    int numCols = static_cast<int>(_numCols);

    NODE_ASSERT(scoresTensor->dims[3] == 1, "scores has dim 3 not equal to 1");
    NODE_ASSERT(geometryTensor->dims[3] == 5, "geometry has dim 3 not equal to 5");
    NODE_ASSERT(scoresTensor->dims[1] == geometryTensor->dims[1], "scores and geometry has not equal dim 2");
    NODE_ASSERT(scoresTensor->dims[2] == geometryTensor->dims[2], "scores and geometry has not equal dim 3");

    NODE_ASSERT((numRows * 4) == imageHeight, "image is not x4 larger than score/geometry data");
    NODE_ASSERT((numCols * 4) == imageWidth, "image is not x4 larger than score/geometry data");

    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<BoxMetadata> metadata;

    // Extract the scores (probabilities), followed by the geometrical data used to derive potential bounding box coordinates that surround text
    for (int y = 0; y < numRows; y++) {
        float* scoresData = (float*)scoresTensor->data + (y * numCols);
        float* geometryData = (float*)geometryTensor->data + (y * numCols * 5);

        for (int x = 0; x < numCols; x++) {
            float score = scoresData[x];
            // If our score does not have sufficient probability, ignore it
            if (score < confidenceThreshold) {
                continue;
            }

            if (debugMode)
                std::cout << "Found confidence: " << scoresData[x] << std::endl;

            // Compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            int offsetX = x * 4;
            int offsetY = y * 4;

            // Extract the rotation angle for the prediction and then compute the sin and cosine
            int dataOffset = (x * 5);
            float angle = geometryData[dataOffset + 4];

            if (debugMode)
                std::cout << "Angle: " << angle << std::endl;
            float cos = std::cos(angle);
            float sin = std::sin(angle);

            // Use the geometry volume to derive the width and height of the bounding box
            float h = geometryData[dataOffset + 0] + geometryData[dataOffset + 2];
            float w = geometryData[dataOffset + 1] + geometryData[dataOffset + 3];

            cv::Point2i p2{
                offsetX + static_cast<int>(cos * geometryData[dataOffset + 1] + sin * geometryData[dataOffset + 2]),
                offsetY + static_cast<int>(-sin * geometryData[dataOffset + 1] + cos * geometryData[dataOffset + 2])};
            cv::Point2i p1{
                static_cast<int>(-sin * h) + p2.x,
                static_cast<int>(-cos * h) + p2.y};
            cv::Point2i p3{
                static_cast<int>(-cos * w) + p2.x,
                static_cast<int>(sin * w) + p2.y};
            cv::Point2i p4{
                p3.x + p1.x - p2.x,
                p3.y + p1.y - p2.y};

            int x1 = std::min(std::min(std::min(p2.x, p1.x), p3.x), p4.x);
            int x2 = std::max(std::max(std::max(p2.x, p1.x), p3.x), p4.x);
            int y1 = std::min(std::min(std::min(p2.y, p1.y), p3.y), p4.y);
            int y2 = std::max(std::max(std::max(p2.y, p1.y), p3.y), p4.y);

            x1 = std::max(0, (int)(x1 - (x2 - x1) * boxWidthAdjustment));
            x2 = std::min(originalImageWidth, (int)(x2 + (x2 - x1) * boxWidthAdjustment));
            y1 = std::max(0, (int)(y1 - (y2 - y1) * boxHeightAdjustment));
            y2 = std::min(originalImageHeight, (int)(y2 + (y2 - y1) * boxHeightAdjustment));

            if (debugMode) {
                std::stringstream ss;
                ss << "Angled polygon coordinates: " << std::endl;
                ss << p4 << p3 << p1 << p2 << std::endl;
                ss << "Polygon bounding box with no rotation: " << std::endl;
                ss << cv::Point2i(x1, y1) << cv::Point2i(x2, y2) << std::endl;
                ss << "---------------------------" << std::endl;
                std::cout << ss.str() << std::endl;
            }

            NODE_ASSERT(x2 > x1, "detected box width must be greater than 0");
            NODE_ASSERT(y2 > y1, "detected box height must be greater than 0");

            NODE_ASSERT(x2 > x1, "detected box width must be greater than 0");
            NODE_ASSERT(y2 > y1, "detected box height must be greater than 0");

            rects.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.emplace_back(score);
            metadata.emplace_back(BoxMetadata{angle, w * (1.0f + boxWidthAdjustment), h * (1.0f + boxHeightAdjustment)});
        }
    }

    if (debugMode)
        std::cout << "Total findings: " << rects.size() << std::endl;

    std::vector<cv::Rect> filteredBoxes;
    std::vector<float> filteredScores;
    std::vector<BoxMetadata> filteredMetadata;
    nms2(rects, scores, metadata, filteredBoxes, filteredScores, filteredMetadata, overlapThreshold);
    NODE_ASSERT(filteredBoxes.size() == filteredScores.size(), "filtered boxes and scores are not equal length");
    if (filteredBoxes.size() > maxOutputBatch) {
        filteredBoxes.resize(maxOutputBatch);
        filteredScores.resize(maxOutputBatch);
    }

    if (debugMode) {
        std::cout << "Total findings after NMS2 (non max suppression) filter: " << filteredBoxes.size() << std::endl;
    }

    *outputsCount = 3;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));

    NODE_ASSERT((*outputs) != nullptr, "malloc has failed");
    CustomNodeTensor& textImagesTensor = (*outputs)[0];
    textImagesTensor.name = TEXT_IMAGES_TENSOR_NAME;
    if (!copy_images_into_output(&textImagesTensor, filteredBoxes, filteredMetadata, image, targetImageHeight, targetImageWidth, targetImageLayout, convertToGrayScale, rotationAngleThreshold)) {
        free(*outputs);
        return 1;
    }

    CustomNodeTensor& coordinatesTensor = (*outputs)[1];
    coordinatesTensor.name = COORDINATES_TENSOR_NAME;
    if (!copy_boxes_into_output(&coordinatesTensor, filteredBoxes)) {
        free(*outputs);
        cleanup(textImagesTensor);
        return 1;
    }

    CustomNodeTensor& confidenceTensor = (*outputs)[2];
    confidenceTensor.name = CONFIDENCE_TENSOR_NAME;
    if (!copy_confidences_into_output(&confidenceTensor, filteredScores)) {
        free(*outputs);
        cleanup(textImagesTensor);
        cleanup(coordinatesTensor);
        return 1;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 4) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 4) == 0, "original image width must be divisible by 4");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");

    *infoCount = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = IMAGE_TENSOR_NAME;
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

    (*info)[1].name = SCORES_TENSOR_NAME;
    (*info)[1].dimsCount = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = originalImageHeight / 4;
    (*info)[1].dims[2] = originalImageWidth / 4;
    (*info)[1].dims[3] = 1;
    (*info)[1].precision = FP32;

    (*info)[2].name = GEOMETRY_TENSOR_NAME;
    (*info)[2].dimsCount = 4;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[2].dims) != nullptr, "malloc has failed");
    (*info)[2].dims[0] = 1;
    (*info)[2].dims[1] = originalImageHeight / 4;
    (*info)[2].dims[2] = originalImageWidth / 4;
    (*info)[2].dims[3] = 5;
    (*info)[2].precision = FP32;
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

    *infoCount = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TEXT_IMAGES_TENSOR_NAME;
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

    (*info)[1].name = COORDINATES_TENSOR_NAME;
    (*info)[1].dimsCount = 3;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 0;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 4;
    (*info)[1].precision = I32;

    (*info)[2].name = CONFIDENCE_TENSOR_NAME;
    (*info)[2].dimsCount = 3;
    (*info)[2].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[2].dims) != nullptr, "malloc has failed");
    (*info)[2].dims[0] = 0;
    (*info)[2].dims[1] = 1;
    (*info)[2].dims[2] = 1;
    (*info)[2].precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
