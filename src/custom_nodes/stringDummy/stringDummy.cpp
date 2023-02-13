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
#include "../common/custom_node_library_internal_manager.hpp"
#include "../common/utils.hpp"

static constexpr const char* INPUT_TENSOR_NAME = "input_numbers";
static constexpr const char* INPUT_INFO_NAME = "input_info";
static constexpr const char* INPUT_INFO_DIMS_NAME = "input_info_dims";

static constexpr const char* OUTPUT_NAME = "output";
static constexpr const char* OUTPUT_TENSOR_NAME = "output_numbers";
static constexpr const char* OUTPUT_TENSOR_DIMS_NAME = "output_dims";
static constexpr const char* OUTPUT_INFO_NAME = "output_info";
static constexpr const char* OUTPUT_INFO_DIMS_NAME = "output_info_dims";

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager = static_cast<ovms::custom_nodes_common::CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    std::shared_lock<std::shared_timed_mutex> lock(internalManager->getInternalManagerLock());

    const CustomNodeTensor* inputTensor = &(inputs[0]);

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
    outputTensor->precision = inputTensor->precision;

    for (uint64_t i = 0; i < outputTensor->dataBytes; i += sizeof(char)) {
        *(char*)(outputTensor->data + i) = *(char*)(inputTensor->data + i);
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager = static_cast<ovms::custom_nodes_common::CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    NODE_ASSERT(internalManager != nullptr, "internalManager is not initialized");
    std::shared_lock<std::shared_timed_mutex> lock(internalManager->getInternalManagerLock());

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, INPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = INPUT_TENSOR_NAME;
    (*info)->dimsCount = 1;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), INPUT_INFO_DIMS_NAME, 1 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->precision = UNSPECIFIED;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager = static_cast<ovms::custom_nodes_common::CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    NODE_ASSERT(internalManager != nullptr, "internalManager is not initialized");
    std::shared_lock<std::shared_timed_mutex> lock(internalManager->getInternalManagerLock());

    *infoCount = 1;
    if (!get_buffer<struct CustomNodeTensorInfo>(internalManager, info, OUTPUT_INFO_NAME, 1 * sizeof(CustomNodeTensorInfo))) {
        return 1;
    }
    (*info)->name = "output_numbers";
    (*info)->dimsCount = 1;
    if (!get_buffer<uint64_t>(internalManager, &((*info)->dims), OUTPUT_INFO_DIMS_NAME, 1 * sizeof(uint64_t))) {
        release(*info, internalManager);
        return 1;
    }
    (*info)->dims[0] = 1;
    (*info)->precision = UNSPECIFIED;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager = static_cast<ovms::custom_nodes_common::CustomNodeLibraryInternalManager*>(customNodeLibraryInternalManager);
    if (!internalManager->releaseBuffer(ptr)) {
        free(ptr);
        return 0;
    }
    return 0;
}
