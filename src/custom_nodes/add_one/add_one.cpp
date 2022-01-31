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
#include <cstring>
#include <iostream>

#include "../../custom_node_interface.h"
#include "../common/custom_node_library_internal_manager.hpp"
#include "../common/utils.hpp"
#include "add_one_internal_manager.hpp"

using InternalManager = ovms::custom_nodes_common::AddOneInternalManager;

static constexpr const char* INPUT_NUMBERS_TENSOR_NAME = "input_numbers";
static constexpr const char* INPUT_TENSOR_INFO_NAME = "input_info";
static constexpr const char* INPUT_NUMBERS_INFO_DIMS_NAME = "input_info_dims";

static constexpr const char* OUTPUT_TENSOR_NAME = "output";
static constexpr const char* OUTPUT_NUMBERS_TENSOR_NAME = "output_numbers";
static constexpr const char* OUTPUT_NUMBERS_DIMS_NAME = "output_dims";
static constexpr const char* OUTPUT_TENSOR_INFO_NAME = "output_info";
static constexpr const char* OUTPUT_NUMBERS_INFO_DIMS_NAME = "output_info_dims";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // creating InternalManager instance
    std::unique_ptr<InternalManager> internalManager = std::make_unique<InternalManager>();
    NODE_ASSERT(internalManager != nullptr, "internalManager allocation failed");

    // reading parameters to determine size of pre-allocated buffers
    int output_queue_size = get_int_parameter("output_queue_size", params, paramsCount, internalManager->getCurrentOutputQueueSize());
    NODE_ASSERT(output_queue_size > 0, "output_queue_size should be greater than 0");
    internalManager->setCurrentOutputQueueSize(output_queue_size);

    int info_queue_size = get_int_parameter("info_queue_size", params, paramsCount, internalManager->getCurrentInfoQueueSize());
    NODE_ASSERT(info_queue_size > 0, "info_queue_size should be greater than 0");
    internalManager->setCurrentInfoQueueSize(info_queue_size);

    // creating BuffersQueues for output tensor
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_NAME, 1 * sizeof(CustomNodeTensor), output_queue_size), "output buffer creation failed");

    // creating BuffersQueues for output: output_numbers
    uint64_t byteSize = sizeof(float) * internalManager->getOutputSize();
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_NUMBERS_TENSOR_NAME, byteSize, output_queue_size), "output tensor buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_NUMBERS_DIMS_NAME, 2 * sizeof(uint64_t), output_queue_size), "output tensor dims buffer creation failed");

    // creating BuffersQueues for info tensors
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_TENSOR_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo), info_queue_size), "input info buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo), info_queue_size), "output info buffer creation failed");

    // creating BuffersQueues for inputs dims in getInputsInfo
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_NUMBERS_INFO_DIMS_NAME, 2 * sizeof(uint64_t), info_queue_size), "input info dims buffer creation failed");

    // creating BuffersQueues for outputs dims in getOutputsInfo
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_NUMBERS_INFO_DIMS_NAME, 2 * sizeof(uint64_t), info_queue_size), "output info dims buffer creation failed");

    *customNodeLibraryInternalManager = internalManager.release();
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    InternalManager* internalManager = static_cast<InternalManager*>(customNodeLibraryInternalManager);
    delete internalManager;
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    InternalManager* internalManager = static_cast<InternalManager*>(customNodeLibraryInternalManager);
    NODE_ASSERT(internalManager != nullptr, "internalManager is not initialized");

    NODE_ASSERT(inputsCount == 1, "too many inputs provided");
    NODE_ASSERT(std::strcmp(inputs[0].name, INPUT_NUMBERS_TENSOR_NAME) == 0, "invalid input name");

    const CustomNodeTensor* inputTensor = &(inputs[0]);

    NODE_ASSERT(inputTensor->precision == FP32, "input precision is not FP32");
    NODE_ASSERT(inputTensor->dimsCount == 2, "input shape must have 2 dimensions");
    NODE_ASSERT(inputTensor->dims[0] == 1, "input batch size must be 1");
    NODE_ASSERT(inputTensor->dims[1] == 10, "input dim[1] must be 10");

    int add_number = get_int_parameter("add_number", params, paramsCount, 1);
    NODE_ASSERT(add_number >= 0, "add_number should be equal or greater than 0");
    int sub_number = get_int_parameter("sub_number", params, paramsCount, 0);
    NODE_ASSERT(sub_number >= 0, "sub_number should be equal or greater than 0");

    *outputsCount = 1;
    if (!get_buffer<struct CustomNodeTensor>(internalManager, outputs, OUTPUT_TENSOR_NAME, *outputsCount * sizeof(CustomNodeTensor))) {
        return 1;
    }

    struct CustomNodeTensor* outputTensor = (&(*outputs))[0];
    outputTensor->name = OUTPUT_NUMBERS_TENSOR_NAME;

    float* buffer = nullptr;
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_NUMBERS_TENSOR_NAME, inputTensor->dataBytes * sizeof(uint8_t))) {
        release(*outputs, internalManager);
        return 1;
    }
    memcpy(buffer, inputTensor->data, inputTensor->dataBytes);
    outputTensor->data = reinterpret_cast<uint8_t*>(buffer);
    outputTensor->dataBytes = inputTensor->dataBytes;

    outputTensor->dimsCount = inputTensor->dimsCount;
    if (!get_buffer<uint64_t>(internalManager, &(outputTensor->dims), OUTPUT_NUMBERS_DIMS_NAME, outputTensor->dimsCount * sizeof(uint64_t))) {
        release(buffer, internalManager);
        release(*outputs, internalManager);
        return 1;
    }
    outputTensor->dims[0] = 1;
    outputTensor->dims[1] = 10;
    outputTensor->precision = inputTensor->precision;

    for (uint64_t i = 0; i < outputTensor->dataBytes; i += sizeof(float)) {
        *(float*)(outputTensor->data + i) = *(float*)(inputTensor->data + i) + add_number - sub_number;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    InternalManager* internalManager = static_cast<InternalManager*>(customNodeLibraryInternalManager);
    NODE_ASSERT(internalManager != nullptr, "internalManager is not initialized");

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, INPUT_TENSOR_INFO_NAME, *infoCount * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = INPUT_NUMBERS_TENSOR_NAME;
    (*info)->dimsCount = 2;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), INPUT_NUMBERS_INFO_DIMS_NAME, (*info)->dimsCount * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->dims[1] = internalManager->getInputSize();
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    InternalManager* internalManager = static_cast<InternalManager*>(customNodeLibraryInternalManager);
    NODE_ASSERT(internalManager != nullptr, "internalManager is not initialized");

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, OUTPUT_TENSOR_INFO_NAME, *infoCount * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = "output_numbers";
    (*info)->dimsCount = 2;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), OUTPUT_NUMBERS_INFO_DIMS_NAME, (*info)->dimsCount * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->dims[1] = internalManager->getOutputSize();
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    InternalManager* internalManager = static_cast<InternalManager*>(customNodeLibraryInternalManager);
    if (!internalManager->releaseBuffer(ptr)) {
        free(ptr);
        return 0;
    }
    return 0;
}
