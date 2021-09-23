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
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

#include "../../custom_node_interface.h"

extern "C" {

static const std::string INPUT_TENSOR_NAME = "input_numbers";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::stringstream ss;
    // validate inputs
    float* inputTensor = nullptr;
    size_t valuesPerTensor = 0;
    for (size_t i = 0; i < inputsCount; ++i) {
        if (INPUT_TENSOR_NAME == inputs[i].name) {
            if (inputs[i].dimsCount != 2 ||
                inputs[i].dims[0] != 1 ||
                inputs[i].dims[1] == 0) {
                ss << "improper " << INPUT_TENSOR_NAME
                   << " dimensions: [" << inputs[i].dims[0] << ", " << inputs[i].dims[1] << "]" << std::endl;
                std::cout << ss.str() << std::endl;
                return 1;
            }
            ss << "Input valuesPerTensor:" << inputs[i].dims[1] << std::endl;
            valuesPerTensor = inputs[i].dims[1];
            inputTensor = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "Unexpected input" << inputs[i].name << std::endl;
            return 1;
        }
    }
    if (!inputTensor) {
        std::cout << "lacking inputs" << std::endl;
        return 1;
    }
    uint64_t demultiplyCount = static_cast<uint64_t>(inputTensor[0]);
    ss << "Will demultiply with count = " << demultiplyCount << std::endl;

    // prepare outputs
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    float* result = (float*)malloc(demultiplyCount * valuesPerTensor * sizeof(float));  // dummy input size * number of ops

    CustomNodeTensor& resultTensor = (*outputs)[0];
    resultTensor.name = "dynamic_demultiplex_results";
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsCount = 3;
    resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsCount * sizeof(uint64_t));
    resultTensor.dims[0] = demultiplyCount;
    resultTensor.dims[1] = 1;
    resultTensor.dims[2] = valuesPerTensor;
    resultTensor.dataBytes = resultTensor.dims[0] * resultTensor.dims[1] * resultTensor.dims[2] * sizeof(float);
    resultTensor.precision = FP32;

    // perform operations - copy input tensor n times
    for (size_t demultiplyCopyId = 0; demultiplyCopyId < demultiplyCount; ++demultiplyCopyId) {
        for (size_t dummyPos = 0; dummyPos < valuesPerTensor; ++dummyPos) {
            auto resultIndex = demultiplyCopyId * valuesPerTensor + dummyPos;
            result[resultIndex] = inputTensor[dummyPos];
            if (demultiplyCount < 100 || ((demultiplyCopyId % 100) == 0)) {
                ss << "demultiplyCopyId:" << demultiplyCopyId
                   << " dummyPos:" << dummyPos
                   << " resultIndex:" << resultIndex
                   << " result:" << result[resultIndex]
                   << " inputTensor:" << inputTensor[dummyPos]
                   << std::endl;
            }
        }
    }
    std::cout << ss.str() << std::endl;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = "input_numbers";
    (*info)[0].precision = FP32;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 10;

    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = "dynamic_demultiplex_results";
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = 0;
    (*info)[0].dims[1] = 1;
    (*info)[0].dims[2] = 10;
    (*info)[0].precision = FP32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    std::cout << "Dynamic demultiplexer release" << std::endl;
    free(ptr);
    return 0;
}
}
