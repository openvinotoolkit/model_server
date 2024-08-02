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
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../custom_node_interface.h"

extern "C" {
enum Method {
    MAXIMUM_MINIMUM,
    MAXIMUM_AVERAGE,
    MAXIMUM_MAXIMUM,
    METHOD_COUNT
};

static const std::string INPUT_TENSOR_NAME = "input_tensors";
static const std::string OUTPUT_TENSOR_NAME = "maximum_tensor";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::stringstream ss;
    // choose selection criteria
    Method selectionMethod = Method::METHOD_COUNT;
    if (paramsCount != 1) {
        std::cout << "Wrong number of parameters - expected 1" << std::endl;
        return 1;
    }
    if (strcmp(params[0].key, "selection_criteria") == 0) {
        if (strcmp(params[0].value, "MAXIMUM_MINIMUM") == 0) {
            selectionMethod = Method::MAXIMUM_MINIMUM;
        } else if (strcmp(params[0].value, "MAXIMUM_MAXIMUM") == 0) {
            selectionMethod = Method::MAXIMUM_MAXIMUM;
        } else if (strcmp(params[0].value, "MAXIMUM_AVERAGE") == 0) {
            selectionMethod = Method::MAXIMUM_AVERAGE;
        } else {
            ss << "Not allowed selection criteria chosen:" << params[0].value << std::endl;
            std::cout << ss.str() << std::endl;
            return 1;
        }
    } else {
        std::cout << "Non recognized param string" << std::endl;
        return 1;
    }
    // get input tensor
    if (inputsCount != 1) {
        std::cout << "Wrong number of inputs - expected 1" << std::endl;
        return 1;
    }
    float* inputTensor;
    size_t valuesPerTensor = 0;
    size_t numberOfOps = 0;
    if (INPUT_TENSOR_NAME == inputs[0].name) {
        if (inputs[0].dimsCount != 3 ||
            inputs[0].dims[1] != 1 ||
            inputs[0].dims[0] == 0 ||
            inputs[0].dims[2] == 0) {
            ss << "improper " << INPUT_TENSOR_NAME
               << " dimensions: [" << inputs[0].dims[0]
               << ", " << inputs[0].dims[1]
               << ", " << inputs[0].dims[2] << "]" << std::endl;
            std::cout << ss.str() << std::endl;
            return 1;
        }
        ss << "Input valuesPerTensor: " << inputs[0].dims[1] << std::endl;
        numberOfOps = inputs[0].dims[0];
        valuesPerTensor = inputs[0].dims[2];
        inputTensor = reinterpret_cast<float*>(inputs[0].data);
    } else {
        ss << "Lacking input: " << INPUT_TENSOR_NAME << std::endl;
        std::cout << ss.str() << std::endl;
        return 1;
    }
    // prepare output
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    float* result = (float*)malloc(valuesPerTensor * sizeof(float));

    CustomNodeTensor& resultTensor = **outputs;
    resultTensor.name = OUTPUT_TENSOR_NAME.c_str();
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsCount = 2;
    resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsCount * sizeof(uint64_t));
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = valuesPerTensor;
    resultTensor.dataBytes = resultTensor.dims[0] * resultTensor.dims[1] * sizeof(float);
    resultTensor.precision = FP32;

    // perform operations
    std::vector<float> minimums(numberOfOps, std::numeric_limits<int>::max());
    std::vector<float> maximums(numberOfOps, std::numeric_limits<int>::lowest());
    std::vector<float> averages(numberOfOps, 0);
    for (size_t opId = 0; opId < numberOfOps; ++opId) {
        for (size_t dummyPos = 0; dummyPos < valuesPerTensor; ++dummyPos) {
            auto index = opId * valuesPerTensor + dummyPos;
            switch (selectionMethod) {
            case Method::MAXIMUM_MAXIMUM:
                maximums[opId] = std::max(maximums[opId], inputTensor[index]);
                break;
            case Method::MAXIMUM_MINIMUM:
                minimums[opId] = std::min(minimums[opId], inputTensor[index]);
                break;
            case Method::MAXIMUM_AVERAGE:
                averages[opId] += inputTensor[index];
                break;
            default:
                return 1;
            }
            ss << "opId:" << opId
               << " dummyPos:" << dummyPos
               << " input:" << inputTensor[index]
               << " minimums:" << minimums[opId]
               << " averages:" << averages[opId]
               << " maximums:" << maximums[opId]
               << " selected method:" << selectionMethod
               << std::endl;
        }
        averages[opId] /= valuesPerTensor;
    }
    // find which tensor to choose
    const std::vector<float>* fromWhichContainerToChoose = &maximums;
    switch (selectionMethod) {
    case Method::MAXIMUM_MAXIMUM:
        fromWhichContainerToChoose = &maximums;
        break;
    case Method::MAXIMUM_MINIMUM:
        fromWhichContainerToChoose = &minimums;
        break;
    case Method::MAXIMUM_AVERAGE:
        fromWhichContainerToChoose = &averages;
        break;
    default:
        return 1;
    }
    size_t whichTensor = std::distance(fromWhichContainerToChoose->begin(),
        std::max_element(fromWhichContainerToChoose->begin(),
            fromWhichContainerToChoose->end()));
    ss << "Selected tensor pos: " << whichTensor << std::endl;
    // copy appropriate tensor
    for (size_t i = 0; i < valuesPerTensor; ++i) {
        size_t index = whichTensor * valuesPerTensor + i;
        ss << "Putting tensor:" << whichTensor
           << " index:" << index
           << " with value:" << inputTensor[index] << std::endl;
        result[i] = inputTensor[index];
    }
    std::cout << ss.str() << std::endl;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = INPUT_TENSOR_NAME.c_str();
    (*info)->dimsCount = 3;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 4;
    (*info)->dims[1] = 1;
    (*info)->dims[2] = 10;
    (*info)->precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = OUTPUT_TENSOR_NAME.c_str();
    (*info)->dimsCount = 2;
    (*info)->dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 1;
    (*info)->dims[1] = 10;
    (*info)->precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    std::cout << "ChooseMaximumCustomLibrary release" << std::endl;
    free(ptr);
    return 0;
}
}
