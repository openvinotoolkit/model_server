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
#include <shared_mutex>

#include "../../custom_node_interface.h"
#include "../common/buffersqueue.hpp"
#include "../common/customNodeLibraryInternalManager.hpp"
#include "utils.hpp"

using CustomNodeLibraryInternalManager = ovms::custom_nodes_common::CustomNodeLibraryInternalManager;
using BuffersQueue = ovms::custom_nodes_common::BuffersQueue;

static constexpr const char* INPUT_TENSOR_NAME = "input_numbers";
static constexpr const char* INPUT_INFO_NAME = "input_info";
static constexpr const char* INPUT_INFO_DIMS_NAME = "input_info_dims";

static constexpr const char* OUTPUT_NAME = "output";
static constexpr const char* OUTPUT_TENSOR_NAME = "output_numbers";
static constexpr const char* OUTPUT_TENSOR_DIMS_NAME = "output_dims";
static constexpr const char* OUTPUT_INFO_NAME = "output_info";
static constexpr const char* OUTPUT_INFO_DIMS_NAME = "output_info_dims";

static constexpr const int QUEUE_SIZE = 1;

std::shared_mutex internalManagerLock;

template <typename T>
bool get_buffer(CustomNodeLibraryInternalManager* internalManager, T** buffer, const char* buffersQueueName, uint64_t byte_size) {
    auto buffersQueue = internalManager->getBuffersQueue(buffersQueueName);
    if (!(buffersQueue == nullptr)) {
        *buffer = static_cast<T*>(buffersQueue->getBuffer());
    }
    if (*buffer == nullptr || buffersQueue == nullptr) {
        *buffer = (T*)malloc(byte_size);
        if (*buffer == nullptr) {
            std::cout << "allocation for buffer: " << buffersQueueName << "FAILED" << std::endl;
            return false;
        }
    }
    return true;
}

int initializeInternalManager(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // creating InternalManager instance
    std::unique_ptr<CustomNodeLibraryInternalManager> internalManager = std::make_unique<CustomNodeLibraryInternalManager>();
    NODE_ASSERT(internalManager != nullptr, "internalManager allocation failed");

    // creating BuffersQueues for output
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_NAME, 1 * sizeof(CustomNodeTensor), QUEUE_SIZE), "output buffer creation failed");

    // creating BuffersQueues for output tensor
    uint64_t byteSize = sizeof(float) * 10;
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_NAME, byteSize, QUEUE_SIZE), "output tensor buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_TENSOR_DIMS_NAME, 2 * sizeof(uint64_t), QUEUE_SIZE), "output tensor dimsbuffer creation failed");

    // creating BuffersQueues for info tensors
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "input info buffer creation failed");
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo), QUEUE_SIZE), "output info buffer creation failed");

    // creating BuffersQueues for input info dims
    NODE_ASSERT(internalManager->createBuffersQueue(INPUT_INFO_DIMS_NAME, 2 * sizeof(uint64_t), QUEUE_SIZE), "input info dims buffer creation failed");

    // creating BuffersQueues for output info dims
    NODE_ASSERT(internalManager->createBuffersQueue(OUTPUT_INFO_DIMS_NAME, 2 * sizeof(uint64_t), QUEUE_SIZE), "output info dims buffer creation failed");

    *customNodeLibraryInternalManager = internalManager.release();
    return 0;
}

int reinitializeInternalManagerIfNeccessary(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // std::cout << "started reinitialize" << std::endl;
    // std::unique_lock lock(internalManagerLock);
    // CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(*customNodeLibraryInternalManager);

    // uint64_t byteSize = sizeof(float) * 10;
    // NODE_ASSERT(internalManager->recreateBuffersQueue(OUTPUT_TENSOR_NAME, byteSize, QUEUE_SIZE), "buffer recreation failed");

    return 0;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
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
    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    NODE_ASSERT(inputsCount == 1, "too many inputs provided");
    NODE_ASSERT(std::strcmp(inputs[0].name, INPUT_TENSOR_NAME) == 0, "invalid input name");

    const CustomNodeTensor* inputTensor = &(inputs[0]);

    NODE_ASSERT(inputTensor->precision == FP32, "input precision is not FP32");
    NODE_ASSERT(inputTensor->dimsCount == 2, "input shape must have 2 dimensions");
    NODE_ASSERT(inputTensor->dims[0] == 1, "input batch size must be 1");
    NODE_ASSERT(inputTensor->dims[1] == 10, "input dim[1] must be 10");

    int multiplier = get_int_parameter("multiplier", params, paramsCount, 1);
    NODE_ASSERT(multiplier > 0, "multiplier should be greater than 0");

    *outputsCount = 1;
    if (!get_buffer<struct CustomNodeTensor>(internalManager, outputs, OUTPUT_NAME, 1 * sizeof(CustomNodeTensor))) {
        return 1;
    }

    struct CustomNodeTensor* outputTensor = (&(*outputs))[0];
    outputTensor->name = OUTPUT_TENSOR_NAME;

    float* buffer = nullptr;
    if (!get_buffer<float>(internalManager, &buffer, OUTPUT_TENSOR_NAME, inputTensor->dataBytes * sizeof(uint8_t))) {
        release(*outputs, internalManager);
        return 1;
    }
    memcpy(buffer, inputTensor->data, inputTensor->dataBytes);
    outputTensor->data = reinterpret_cast<uint8_t*>(buffer);
    outputTensor->dataBytes = inputTensor->dataBytes;

    if (!get_buffer<uint64_t>(internalManager, &(outputTensor->dims), OUTPUT_TENSOR_DIMS_NAME, 2 * sizeof(uint64_t))) {
        release(buffer, internalManager);
        release(*outputs, internalManager);
        return 1;
    }
    outputTensor->dimsCount = inputTensor->dimsCount;
    outputTensor->dims[0] = 1;
    outputTensor->dims[1] = 10;
    outputTensor->precision = inputTensor->precision;

    for (uint64_t i = 0; i < outputTensor->dataBytes; i += sizeof(float)) {
        *(float*)(outputTensor->data + i) = (*(float*)(inputTensor->data + i) + 1.0f) * multiplier;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, INPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = INPUT_TENSOR_NAME;
    (*info)->dimsCount = 2;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), INPUT_INFO_DIMS_NAME, 2 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 10;
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::shared_lock lock(internalManagerLock);
    CustomNodeLibraryInternalManager* internalManager = static_cast<CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, OUTPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = "output_numbers";
    (*info)->dimsCount = 2;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), OUTPUT_INFO_DIMS_NAME, 2 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 10;
    (*info)->precision = FP32;
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
