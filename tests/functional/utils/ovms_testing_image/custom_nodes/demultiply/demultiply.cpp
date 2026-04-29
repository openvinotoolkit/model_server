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
#include <vector>
#include <string>
#include <regex>

#include "../../custom_node_interface.h"
#include "opencv2/opencv.hpp"

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }

#define LOG_ENTRY_M(method) std::cout << "[CustomNode] [" << __LINE__ << "] demultiply." << method << " entry" << std::endl;
#define LOG_EXIT_M(method) std::cout << "[CustomNode] [" << __LINE__ << "] demultiply." << method << " exit" << std::endl;

static constexpr const char* IMAGE_TENSOR_NAME = "tensor";
static constexpr const char* TENSOR_OUT = "tensor_out";
const uint64_t NEW_LAYER_DIM = 3;
const uint64_t BATCH_SIZE = 1;
const uint64_t NR_OF_CHANNELS = 3;

bool prepare_output_tensor(struct CustomNodeTensor** outputs, int* outputsCount, int img_height, int img_width)
{
    bool result = true;
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    NODE_ASSERT((*outputs) != nullptr, "malloc has failed");
    CustomNodeTensor& tensor = (*outputs)[0];
    uint64_t byte_size = sizeof(float) * NEW_LAYER_DIM * BATCH_SIZE * NR_OF_CHANNELS * img_height * img_width;

    float* buffer;
    tensor.name = TENSOR_OUT;
    buffer = (float*)malloc(byte_size);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    if(buffer == nullptr) {
        result = false;
    } else {
        tensor.data = reinterpret_cast<uint8_t*>(buffer);
        tensor.dataBytes = byte_size;
        tensor.dimsCount = 5;
        tensor.dims = (uint64_t*)malloc(tensor.dimsCount * sizeof(uint64_t));
        NODE_ASSERT(tensor.dims != nullptr, "malloc has failed");
        int dims_content[] = {NEW_LAYER_DIM, BATCH_SIZE, NR_OF_CHANNELS, img_height, img_width};
        for(uint64_t i = 0; i < tensor.dimsCount; i++)
            tensor.dims[(int)i] = dims_content[(int)i];
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
            void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("execute enter");
    int dimsCount = inputs[0].dimsCount;
    int img_height = inputs[0].dims[dimsCount - 2];
    int img_width = inputs[0].dims[dimsCount - 1];
    std::cout << "demultiply load img_height: " << img_height << "; img_width: " << img_width << std::endl;
    prepare_output_tensor(outputs, outputsCount, img_height, img_width);
    LOG_EXIT_M("execute");
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
                  void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("getInputsInfo");
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 3;
    (*info)[0].dims[2] = 224;
    (*info)[0].dims[3] = 224;
    (*info)[0].precision = FP32;
    LOG_EXIT_M("getInputsInfo");
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info,
                   int* infoCount,
                   const struct CustomNodeParam* params,
                   int paramsCount,
                   void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("getOutputsInfo");
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    (*info)[0].name = TENSOR_OUT;
    (*info)[0].dimsCount = 5;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = get_int_parameter("demultiply_size", params, paramsCount, 0);
    (*info)[0].dims[1] = 1;
    (*info)[0].dims[2] = 3;
    (*info)[0].dims[3] = 224;
    (*info)[0].dims[4] = 224;
    (*info)[0].precision = FP32;
    std::cout << "[CustomNode] [" << __LINE__ << "] demultiply.getOutputsInfo " << TENSOR_OUT << std::endl;
    std::cout << "    ["<<(*info)[0].dims[0]<<", "<<(*info)[0].dims[1]<<", "<<(*info)[0].dims[2]<<", "<<(*info)[0].dims[3]<<", "<<(*info)[0].dims[3]<<"]"<< std::endl;
    LOG_EXIT_M("getOutputsInfo");
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager)
{
    LOG_ENTRY_M("release");
    free(ptr);
    LOG_EXIT_M("release");
    return 0;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}
