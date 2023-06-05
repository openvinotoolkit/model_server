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
#include <map>
#include <string>

#include "../../custom_node_interface.h"
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#include "opencv2/opencv.hpp"

static constexpr const char* TENSOR_NAME = "image";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // Parameters reading

    // Image size.
    //
    // If not specified (-1), the image will not be resized.
    // When specified, cv::resize is used to resize an image.
    // Original image size must not specified, input size is dynamic.
    int _targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int _targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(_targetImageHeight > 0 || _targetImageHeight == -1, "target image height - when specified, must be larger than 0");
    NODE_ASSERT(_targetImageWidth > 0 || _targetImageWidth == -1, "target image width - when specified, must be larger than 0");

    // Color order.
    //
    // Possible orders: BGR (default), RGB and GRAY.
    // Depending on the order, number of color channels will be selected - 3 for BGR/RGB and 1 for GRAY.
    std::string originalImageColorOrder = get_string_parameter("original_image_color_order", params, paramsCount, "BGR");
    std::string targetImageColorOrder = get_string_parameter("target_image_color_order", params, paramsCount);
    targetImageColorOrder = targetImageColorOrder.empty() ? originalImageColorOrder : targetImageColorOrder;
    NODE_ASSERT(originalImageColorOrder == "BGR" || originalImageColorOrder == "RGB" || originalImageColorOrder == "GRAY", "original image layout must be BGR, RGB or GRAY");
    NODE_ASSERT(targetImageColorOrder == "BGR" || targetImageColorOrder == "RGB" || targetImageColorOrder == "GRAY", "target image layout must be BGR, RGB or GRAY");
    uint64_t targetImageColorChannels = targetImageColorOrder == "GRAY" ? 1 : 3;

    // Image layout.
    //
    // Possible layouts: NCHW and NHWC.
    // Since OpenCV is used for transformations, image will be converted to cv::Mat.
    // The container requires the data in NHWC format, so selecting input layout NCHW will convert data to NHWC, therefore decrease performance.
    // Selecting target layout NCHW will also perform conversion before copying data into output.
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount);
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount);
    targetImageLayout = targetImageLayout.empty() ? originalImageLayout : targetImageLayout;
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");

    // Scale.
    //
    // When specified, all pixel values will be divided by this value.
    bool isScaleDefined = false;
    float scale = get_float_parameter("scale", params, paramsCount, isScaleDefined, -1);
    NODE_ASSERT(scale != 0, "cannot divide by scale equal to 0");

    // Scale values.
    //
    // Smilar to scale but scale value should be provided per color channel.
    std::vector<float> scaleValues = get_float_list_parameter("scale_values", params, paramsCount);
    for (auto scale : scaleValues) {
        NODE_ASSERT(scale != 0, "cannot divide by scale equal to 0");
    }

    // Mean values.
    //
    // If not specified, the image will not be scaled.
    // When specified, all pixel values will be subtracted by this value per channel.
    // The exact meaning and order of channels depend on input image.
    std::vector<float> meanValues = get_float_list_parameter("mean_values", params, paramsCount);

    // Debug flag for additional logging.
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";

    // ------------ validation start -------------
    NODE_ASSERT(inputsCount == 1, "there must be exactly one input");
    const CustomNodeTensor* imageTensor = inputs;
    NODE_ASSERT(std::strcmp(imageTensor->name, TENSOR_NAME) == 0, "node input name is wrong");
    NODE_ASSERT(imageTensor->dimsCount == 4, "image tensor shape must have 4 dimensions")
    NODE_ASSERT(imageTensor->dims[0] == 1, "image tensor must have batch size equal to 1")

    uint64_t originalImageHeight = 0;
    uint64_t originalImageWidth = 0;
    uint64_t originalImageColorChannels = 0;

    if (originalImageLayout == "NCHW") {
        originalImageColorChannels = imageTensor->dims[1];
        originalImageHeight = imageTensor->dims[2];
        originalImageWidth = imageTensor->dims[3];
    } else if (originalImageLayout == "NHWC") {
        originalImageHeight = imageTensor->dims[1];
        originalImageWidth = imageTensor->dims[2];
        originalImageColorChannels = imageTensor->dims[3];
    } else {
        std::cout << "original image layout: " << originalImageLayout << " is unsupported" << std::endl;
        return 1;
    }

    NODE_ASSERT(originalImageHeight > 0 && originalImageWidth > 0, "original image size must be positive");
    NODE_ASSERT(originalImageColorChannels == 1 || originalImageColorChannels == 3, "original image color channels must be 1 or 3");
    NODE_ASSERT(originalImageHeight * originalImageWidth * originalImageColorChannels * sizeof(float) == imageTensor->dataBytes, "number of input bytes does not match input shape");

    if (originalImageColorOrder == "GRAY") {
        NODE_ASSERT(originalImageColorChannels == 1, "for color order GRAY color channels must be equal 1");
    }
    if (originalImageColorOrder == "BGR" || originalImageColorOrder == "RGB") {
        NODE_ASSERT(originalImageColorChannels == 3, "for color order BGR/RGB color channels must be equal to 3");
    }

    uint64_t targetImageHeight = _targetImageHeight == -1 ? originalImageHeight : _targetImageHeight;
    uint64_t targetImageWidth = _targetImageWidth == -1 ? originalImageWidth : _targetImageWidth;

    NODE_ASSERT(scaleValues.size() == 0 || targetImageColorChannels == scaleValues.size(), "number of scale values must be equal to number of target image channels");
    NODE_ASSERT(meanValues.size() == 0 || targetImageColorChannels == meanValues.size(), "number of mean values must be equal to number of target image channels");

    auto originalImageResolution = originalImageHeight * originalImageWidth;
    auto targetImageResolution = targetImageHeight * targetImageWidth;

    if (debugMode) {
        std::cout << "Original image size: " << cv::Size2i(originalImageWidth, originalImageHeight) << std::endl;
        std::cout << "Original image resolution: " << originalImageResolution << std::endl;
        std::cout << "Original image color channels: " << originalImageColorChannels << std::endl;
        std::cout << "Original image color order: " << originalImageColorOrder << std::endl;
        std::cout << "Original image layout: " << originalImageLayout << std::endl;
        std::cout << "Target image size: " << cv::Size2i(targetImageWidth, targetImageHeight) << std::endl;
        std::cout << "Target image resolution: " << targetImageResolution << std::endl;
        std::cout << "Target image color channels: " << targetImageColorChannels << std::endl;
        std::cout << "Target image color order: " << targetImageColorOrder << std::endl;
        std::cout << "Target image layout: " << targetImageLayout << std::endl;
        std::cout << "Scale: " << (isScaleDefined ? std::to_string(scale) : "not defined") << std::endl;
        std::cout << "Scale values: " << floatListToString(scaleValues) << std::endl;
        std::cout << "Mean values: " << floatListToString(meanValues) << std::endl;
    }
    // ------------- validation end ---------------

    // Prepare cv::Mat out of imageTensor input.
    // In case input is in NCHW format, perform reordering to NHWC.
    cv::Mat image = cv::Mat(originalImageHeight, originalImageWidth, originalImageColorChannels == 1 ? CV_32FC1 : CV_32FC3);
    if (originalImageLayout == "NCHW") {
        reorder_to_nhwc_2<float>((float*)imageTensor->data, (float*)image.data, originalImageHeight, originalImageWidth, originalImageColorChannels);
    } else {
        std::memcpy(image.data, imageTensor->data, imageTensor->dataBytes);
    }

    // Change color order and number of channels.
    static const std::map<std::pair<std::string, std::string>, int> colors = {
        {{"GRAY", "BGR"}, cv::COLOR_GRAY2BGR},
        {{"GRAY", "RGB"}, cv::COLOR_GRAY2RGB},
        {{"BGR", "RGB"}, cv::COLOR_BGR2RGB},
        {{"BGR", "GRAY"}, cv::COLOR_BGR2GRAY},
        {{"RGB", "BGR"}, cv::COLOR_RGB2BGR},
        {{"RGB", "GRAY"}, cv::COLOR_RGB2GRAY},
    };

    if (originalImageColorOrder != targetImageColorOrder) {
        const auto& colorIt = colors.find({originalImageColorOrder, targetImageColorOrder});
        NODE_ASSERT(colorIt != colors.end(), "unsupported color conversion");
        cv::cvtColor(image, image, colorIt->second);
    }

    // Perform procesesing with scale and mean values. If scale and scaleValues provided only scaleValues are used for scaling.
    // If scale and meanValues provided mean values are subtracted from pixels first then scaling is made.
    // Scaling will be applied before resize if target resolution is smaller.
    if ((isScaleDefined || scaleValues.size() > 0 || meanValues.size() > 0) && originalImageResolution < targetImageResolution) {
        if (debugMode) {
            std::cout << "Performing scaling before resize operation" << std::endl;
        }
        NODE_ASSERT(scale_image(isScaleDefined, scale, meanValues, scaleValues, image), "Error during image scaling");
    }

    // Perform resize operation.
    if (originalImageHeight != targetImageHeight || originalImageWidth != targetImageWidth) {
        cv::resize(image, image, cv::Size(targetImageWidth, targetImageHeight));
    }

    // Scaling should be applied after resize if target resolution is smaller.
    if ((isScaleDefined || scaleValues.size() > 0 || meanValues.size() > 0) && originalImageResolution >= targetImageResolution) {
        if (debugMode) {
            std::cout << "Performing scaling after resize operation" << std::endl;
        }
        NODE_ASSERT(scale_image(isScaleDefined, scale, meanValues, scaleValues, image), "Error during image scaling");
    }

    // Prepare output tensor
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
    output.name = TENSOR_NAME;
    output.data = reinterpret_cast<uint8_t*>(buffer);
    output.dataBytes = byteSize;
    output.dimsCount = 4;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
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

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 0;
    (*info)[0].dims[2] = 0;
    (*info)[0].dims[3] = 0;
    (*info)[0].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // Parameters reading
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0 || targetImageHeight == -1, "target image height - when specified, must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0 || targetImageWidth == -1, "target image width - when specified, must be larger than 0");

    std::string originalImageColorOrder = get_string_parameter("original_image_color_order", params, paramsCount);
    std::string targetImageColorOrder = get_string_parameter("target_image_color_order", params, paramsCount);
    targetImageColorOrder = targetImageColorOrder.empty() ? originalImageColorOrder : targetImageColorOrder;
    NODE_ASSERT(originalImageColorOrder == "BGR" || originalImageColorOrder == "RGB" || originalImageColorOrder == "GRAY", "original image layout must be BGR, RGB or GRAY");
    NODE_ASSERT(targetImageColorOrder == "BGR" || targetImageColorOrder == "RGB" || targetImageColorOrder == "GRAY", "target image layout must be BGR, RGB or GRAY");

    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount);
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount);
    targetImageLayout = targetImageLayout.empty() ? originalImageLayout : targetImageLayout;
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;

    if (targetImageLayout == "NHWC") {
        (*info)[0].dims[1] = targetImageHeight == -1 ? 0 : targetImageHeight;
        (*info)[0].dims[2] = targetImageWidth == -1 ? 0 : targetImageWidth;
        (*info)[0].dims[3] = targetImageColorOrder == "GRAY" ? 1 : 3;
    } else {
        (*info)[0].dims[1] = targetImageColorOrder == "GRAY" ? 1 : 3;
        (*info)[0].dims[2] = targetImageHeight == -1 ? 0 : targetImageHeight;
        (*info)[0].dims[3] = targetImageWidth == -1 ? 0 : targetImageWidth;
    }

    (*info)[0].precision = FP32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
