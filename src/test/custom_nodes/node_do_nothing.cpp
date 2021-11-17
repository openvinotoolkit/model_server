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
#include <string>
#include <vector>
#include <iostream>

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

    if (strcmp(inputs[0].name, "input") != 0) {
        return 2;
    }

    const struct CustomNodeTensor* input = &inputs[0];

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor) * (*outputsCount));
    struct CustomNodeTensor* output = (&(*outputs))[0];

    output->name = "output";
    output->data = (uint8_t*)malloc(input->dataBytes * sizeof(uint8_t));
    output->dataBytes = input->dataBytes;
    memcpy((void*)output->data, (void*)input->data, input->dataBytes * sizeof(uint8_t));
    output->dims = (uint64_t*)malloc(input->dimsCount * sizeof(uint64_t));
    output->dimsCount = input->dimsCount;
    memcpy((void*)output->dims, (void*)input->dims, input->dimsCount * sizeof(uint64_t));
    output->precision = input->precision;
    return 0;
}

// parse dims parameter from string representing list of ints, e.g. "dims" = "[3,5,10]"
int parametrizeDimensions(std::vector<int>& dims, const struct CustomNodeParam* params, int paramsCount) {
    for (int i = 0; i < paramsCount; i++) {
        if (strcmp(params[i].key, "dims") == 0) {
            std::string dimsString = params[i].value;
            if(dimsString.front() != '[' || dimsString.back() != ']') {
                std::cout << "dims param wrong format, should be [int, ...]" << std::endl;
                return 1;
            }
            dims.clear();
            // remove '[' and ']'
            dimsString.erase(0, 1).pop_back();
            size_t pos = 0;
            while ((pos = dimsString.find(',')) != std::string::npos) {
                dims.emplace_back(atoi(dimsString.substr(0, pos).c_str()));
                dimsString.erase(0, pos + 1);
            }
            if (!dimsString.empty()) {
                dims.emplace_back(atoi(dimsString.c_str()));
            }
        }
    }
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::vector<int> dims{3, 5, 10};
    if (parametrizeDimensions(dims, params, paramsCount) != 0) {
        return 1;
    }

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "input";
    (*info)->dimsCount = dims.size();
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    for (int i = 0; i < dims.size(); i++) {
        (*info)->dims[i] = dims.at(i);
    }
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::vector<int> dims{3, 5, 10};
    if (parametrizeDimensions(dims, params, paramsCount) != 0) {
        return 1;
    }
    
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "output";
    (*info)->dimsCount = dims.size();
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    for (int i = 0; i < dims.size(); i++) {
        (*info)->dims[i] = dims.at(i);
    }
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
