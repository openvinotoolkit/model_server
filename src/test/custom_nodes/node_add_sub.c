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
#include <stdio.h>
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
    if (paramsCount != 2) {
        return 1;
    }

    if (inputsCount != 1) {
        return 2;
    }

    if (strcmp(inputs[0].name, "input_numbers") != 0) {
        return 3;
    }

    const struct CustomNodeTensor* input = &inputs[0];

    if (input->precision != FP32) {
        return 4;
    }

    float addValue = 0.0f;
    float subValue = 0.0f;

    for (int i = 0; i < paramsCount; i++) {
        if (strcmp(params[i].key, "add_value") == 0) {
            addValue = atof(params[i].value);
        }
        if (strcmp(params[i].key, "sub_value") == 0) {
            subValue = atof(params[i].value);
        }
    }

    printf("CUSTOM ADD_SUB NODE => Parameters passed: add_value:(%f); sub_value:(%f)\n", addValue, subValue);
    printf("CUSTOM ADD_SUB NODE => Number of input tensors passed: (%d)\n", inputsCount);

    for (int i = 0; i < inputsCount; i++) {
        printf("CUSTOM ADD_SUB NODE => Input Name(%s) DataLen(%ld) DimLen(%ld)\n", inputs[i].name, inputs[i].dataBytes, inputs[i].dimsCount);
    }

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
        *(float*)(output->data + i) = *(float*)(input->data + i) + addValue - subValue;
    }

    fflush(stdout);
    return 0;
}

// Some unit tests are based on a fact that this node library is dynamic and can take shape{1,3} as input.
int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "input_numbers";
    (*info)->dimsCount = 2;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 0;
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "output_numbers";
    (*info)->dimsCount = 2;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 0;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    printf("CUSTOM ADD_SUB RELEASE\n");
    fflush(stdout);
    free(ptr);
    return 0;
}
