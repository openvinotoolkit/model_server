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

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }


static constexpr const char* IMAGE_TENSOR_NAME = "tensor";
static constexpr const char* TENSOR_OUT = "tensor_out";
const uint64_t DIM_0 = 4;
const uint64_t DIM_1 = 4;
const uint64_t DIM_2 = 1;
const uint64_t DIM_3 = 10;

bool prepare_output_tensor(struct CustomNodeTensor** outputs, int* outputsCount)
{
    bool result = true;
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    NODE_ASSERT((*outputs) != nullptr, "malloc has failed");
    CustomNodeTensor& tensor = (*outputs)[0];
    uint64_t byte_size = sizeof(float) * DIM_0 * DIM_1 * DIM_2 * DIM_3;

    float* buffer;
    tensor.name = TENSOR_OUT;
    buffer = (float*)malloc(byte_size);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    if(buffer == nullptr) {
        result = false;
    } else {
        tensor.data = reinterpret_cast<uint8_t*>(buffer);
        tensor.dataBytes = byte_size;
        tensor.dimsCount = 4;
        tensor.dims = (uint64_t*)malloc(tensor.dimsCount * sizeof(uint64_t));
        NODE_ASSERT(tensor.dims != nullptr, "malloc has failed");
        tensor.dims[0] = DIM_0;
        tensor.dims[1] = DIM_1;
        tensor.dims[2] = DIM_2;
        tensor.dims[3] = DIM_3;
        tensor.precision = FP32;
    }
    return result;
}

void cleanup(CustomNodeTensor& tensor) {
    free(tensor.data);
    free(tensor.dims);
}

int execute(const struct CustomNodeTensor* inputs,
            int inputsCount,
            struct CustomNodeTensor** outputs,
            int* outputsCount,
            const struct CustomNodeParam* params,
            int paramsCount,
            void* customNodeLibraryInternalManager)
{
    prepare_output_tensor(outputs, outputsCount);
    return 0;
}

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, int defaultValue) {
    int result = defaultValue;
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                result = std::stoi(params[i].value);
                break;
            } catch (std::invalid_argument& e) {
                result = defaultValue;
            } catch (std::out_of_range& e) {
                result = defaultValue;
            }
        }
    }
    return result;
}

int getInputsInfo(struct CustomNodeTensorInfo** info,
                  int* infoCount,
                  const struct CustomNodeParam* params,
                  int paramsCount,
                  void* customNodeLibraryInternalManager)
{
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = DIM_1;
    (*info)[0].dims[1] = DIM_2;
    (*info)[0].dims[2] = DIM_3;
    (*info)[0].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info,
                   int* infoCount,
                   const struct CustomNodeParam* params,
                   int paramsCount,
                   void* customNodeLibraryInternalManager)
{
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    (*info)[0].name = TENSOR_OUT;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = get_int_parameter("demultiply_size", params, paramsCount, 0);
    (*info)[0].dims[1] = DIM_1;
    (*info)[0].dims[2] = DIM_2;
    (*info)[0].dims[3] = DIM_3;
    (*info)[0].precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager)
{
    free(ptr);
    return 0;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}
