//*****************************************************************************
// Copyright 2023 Intel Corporation
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
    if (paramsCount != 1) {
        return 1;
    }

    if (inputsCount != 1) {
        return 2;
    }

    if (strcmp(inputs[0].name, "scalar") != 0) {
        return 3;
    }

    const struct CustomNodeTensor* input = &inputs[0];

    if (input->precision != FP32) {
        return 4;
    }

    if (input->dataBytes != sizeof(float)) {
        return 5;
    }

    if (input->dimsCount != 0) {
        return 6;
    }

    float addValue = 0.0f;

    for (int i = 0; i < paramsCount; i++) {
        if (strcmp(params[i].key, "scalar_add_value") == 0) {
            addValue = atof(params[i].value);
        }
    }

    printf("CUSTOM SCALAR NODE => Parameters passed: scalar_add_value:(%f)\n", addValue);

    for (int i = 0; i < inputsCount; i++) {
        printf("CUSTOM SCALAR NODE => Input Name(%s) DataLen(%ld) DimLen(%ld)\n", inputs[i].name, inputs[i].dataBytes, inputs[i].dimsCount);
    }

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor) * (*outputsCount));
    struct CustomNodeTensor* output = (&(*outputs))[0];

    output->name = "result_scalar";
    output->data = (uint8_t*)malloc(input->dataBytes);
    output->dataBytes = input->dataBytes;
    output->dims = (uint64_t*)malloc(input->dimsCount * sizeof(uint64_t));  // TODO: malloc of size 0, what does it return?
    output->dimsCount = input->dimsCount;
    memcpy((void*)output->dims, (void*)input->dims, input->dimsCount * sizeof(uint64_t));
    output->precision = input->precision;

    for (uint64_t i = 0; i < output->dataBytes; i += sizeof(float)) {
        *(float*)(output->data + i) = *(float*)(input->data + i) + addValue;
    }

    fflush(stdout);
    return 0;
}

// Some unit tests are based on a fact that this node library is dynamic and can take shape{1,3} as input.
int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "scalar";
    (*info)->dimsCount = 0;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "result_scalar";
    (*info)->dimsCount = 0;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    printf("CUSTOM SCALAR RELEASE\n");
    fflush(stdout);
    free(ptr);
    return 0;
}
