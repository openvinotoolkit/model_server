//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include "../../custom_node_interface.h"
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#pragma warning(push)
#pragma warning(disable : 6269)
#include "opencv2/opencv.hpp"
#pragma warning(pop)

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* DETECTION_TENSOR_NAME = "detection";

DLL_PUBLIC int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

DLL_PUBLIC int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

DLL_PUBLIC int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
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
    uint64_t targetImageColorChannels = 3;
    float confidenceThreshold = get_float_parameter("confidence_threshold", params, paramsCount, -1.0);
    NODE_ASSERT(confidenceThreshold >= 0 && confidenceThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";
    int gaussianBlurKernelSize = get_int_parameter("gaussian_blur_kernel_size", params, paramsCount, -1);
    NODE_ASSERT(gaussianBlurKernelSize > 0, "gaussian blur kernel size must be larger than 0");
    NODE_ASSERT(gaussianBlurKernelSize % 2 == 1, "gaussian blur kernel size must be odd");

    // Inputs reading
    const CustomNodeTensor* imageTensor = nullptr;
    const CustomNodeTensor* detectionTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, IMAGE_TENSOR_NAME) == 0) {
            imageTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, DETECTION_TENSOR_NAME) == 0) {
            detectionTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    // Validating inputs
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
    NODE_ASSERT(_imageHeight <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image height is too large");
    NODE_ASSERT(_imageWidth <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image width is too large");
    int imageHeight = static_cast<int>(_imageHeight);
    int imageWidth = static_cast<int>(_imageWidth);

    // Processing
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
            auto box = cv::Rect(cv::Point(xMin, yMin), cv::Point(xMax, yMax));
            boxes.emplace_back(box);
            if (debugMode) {
                std::cout << "Detection:\nImageID: " << imageId << "; LabelID:" << labelId << "; Confidence:" << confidence << "; Box:" << box << std::endl;
            }
        }
    }

    // Applying blur on detected areas
    for (const auto& box : boxes) {
        cv::GaussianBlur(image(box), image(box), cv::Size(gaussianBlurKernelSize, gaussianBlurKernelSize), 0);
    }

    // Perform resize operation.
    if (originalImageHeight != targetImageHeight || originalImageWidth != targetImageWidth) {
        cv::resize(image, image, cv::Size(targetImageWidth, targetImageHeight));
    }

    // Preparing output tensor
    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * targetImageColorChannels;
    NODE_ASSERT(image.total() * image.elemSize() == byteSize, "buffer size differs");
    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");

    if (targetImageLayout == "NCHW") {
        reorder_to_nchw_2<float>((float*)image.data, (float*)buffer, image.rows, image.cols, image.channels());
    } else {
        std::memcpy((uint8_t*)buffer, image.data, byteSize);
    }

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cout << "malloc has failed" << std::endl;
        free(buffer);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = IMAGE_TENSOR_NAME;
    output.data = reinterpret_cast<uint8_t*>(buffer);
    output.dataBytes = byteSize;
    output.dimsCount = 4;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = 1;
    output.dims[0] = 1;
    if (targetImageLayout == "NCHW") {
        output.dims[1] = targetImageColorChannels;
        output.dims[2] = targetImageHeight;
        output.dims[3] = targetImageWidth;
    } else {
        output.dims[1] = targetImageHeight;
        output.dims[2] = targetImageWidth;
        output.dims[3] = targetImageColorChannels;
    }
    output.precision = FP32;
    return 0;
}

DLL_PUBLIC int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");

    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
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

    (*info)[1].name = DETECTION_TENSOR_NAME;
    (*info)[1].dimsCount = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)[1].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = 200;
    (*info)[1].dims[3] = 7;
    (*info)[1].precision = FP32;
    return 0;
}

DLL_PUBLIC int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;

    if (targetImageLayout == "NHWC") {
        (*info)[0].dims[1] = targetImageHeight;
        (*info)[0].dims[2] = targetImageWidth;
        (*info)[0].dims[3] = 3;
    } else {
        (*info)[0].dims[1] = 3;
        (*info)[0].dims[2] = targetImageHeight;
        (*info)[0].dims[3] = targetImageWidth;
    }

    (*info)[0].precision = FP32;

    return 0;
}

DLL_PUBLIC int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
