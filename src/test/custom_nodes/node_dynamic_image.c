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
#include <stdlib.h>
#include <string.h>

#include "../../custom_node_interface.h"

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager){
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    if (inputsCount != 1) {
        return 1;
    }

    if (strcmp(inputs[0].name, "input_numbers") != 0) {
        return 2;
    }


    const struct CustomNodeTensor* input = &inputs[0];

    if (input->dimsCount != 4) {
        return 3;
    }

    if (input->precision != FP32) {
        return 4;
    }

    float addValue = 8.0f;

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor) * (*outputsCount));
    struct CustomNodeTensor* output = (&(*outputs))[0];

    output->name = "output_numbers";
    output->data = (uint8_t*)malloc(input->dataBytes * sizeof(uint8_t));
    output->dataBytes = input->dataBytes;
    output->dims = (uint64_t*)malloc(input->dimsCount * sizeof(uint64_t));
    output->dimsCount = input->dimsCount;
    memcpy((void*)output->dims, (void*)input->dims, input->dimsCount * sizeof(uint64_t));
    output->precision = input->precision;

    for (uint64_t i = 0; i < output->dataBytes; i += sizeof(float)) {
        *(float*)(output->data + i) = *(float*)(input->data + i) + addValue;
    }
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "input_numbers";
    (*info)->dimsCount = 4;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 0;
    (*info)->dims[1] = 0;
    (*info)->dims[2] = 0;
    (*info)->dims[3] = 0;
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "output_numbers";
    (*info)->dimsCount = 4;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 0;
    (*info)->dims[1] = 0;
    (*info)->dims[2] = 0;
    (*info)->dims[3] = 0;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
