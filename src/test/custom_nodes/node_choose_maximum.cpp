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

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    // choose selection criteria
    Method selectionMethod = Method::METHOD_COUNT;
    if (paramsLength != 1) {
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
            std::cout << "Not allowed selection criteria chosen:" << params[0].value << std::endl;
            return 1;
        }
    } else {
        std::cout << "Non recognized param string" << std::endl;
        return 1;
    }
    // get input tensor
    if (inputsLength != 1) {
        std::cout << "Wrong number of inputs - expected 1" << std::endl;
        return 1;
    }
    float* inputTensor;
    size_t valuesPerTensor = 0;
    size_t numberOfOps = 0;
    if (INPUT_TENSOR_NAME == inputs[0].name) {
        if (inputs[0].dimsLength != 3 ||
            inputs[0].dims[0] != 1 ||
            inputs[0].dims[1] == 0 ||
            inputs[0].dims[2] == 0) {
            std::cout << "improper " << INPUT_TENSOR_NAME
                      << " dimensions: [" << inputs[0].dims[0]
                      << ", " << inputs[0].dims[1]
                      << ", " << inputs[0].dims[2] << "]" << std::endl;
            return 1;
        }
        std::cout << "Input valuesPerTensor" << inputs[0].dims[1] << std::endl;
        numberOfOps = inputs[0].dims[1];
        valuesPerTensor = inputs[0].dims[2];
        inputTensor = reinterpret_cast<float*>(inputs[0].data);
    } else {
        std::cout << "Lacking input: " << INPUT_TENSOR_NAME << std::endl;
        return 1;
    }
    // prepare output
    *outputsLength = 1;
    *outputs = new CustomNodeTensor[*outputsLength];
    float* result = new float[valuesPerTensor];

    CustomNodeTensor& resultTensor = **outputs;
    resultTensor.name = "maximum_tensor";
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsLength = 2;
    resultTensor.dims = new uint64_t[resultTensor.dimsLength];
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = valuesPerTensor;
    resultTensor.dataLength = resultTensor.dims[0] * resultTensor.dims[1] * sizeof(float);
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
                minimums[opId] = std::min(maximums[opId], inputTensor[index]);
                break;
            case Method::MAXIMUM_AVERAGE:
                averages[opId] += inputTensor[index];
                break;
            default:
                return 1;
            }
            std::cout << "opId:" << opId
                      << " dummyPos:" << dummyPos
                      << " input:" << inputTensor[index]
                      << " minimums:" << minimums[opId]
                      << " averages:" << averages[opId]
                      << " maximums:" << maximums[opId]
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
    std::cout << "Selected tensor pos: " << whichTensor << std::endl;
    // copy appropiate tensor
    for (size_t i = 0; i < valuesPerTensor; ++i) {
        size_t index = whichTensor * valuesPerTensor + i;
        std::cout << "Putting tensor:" << whichTensor
                  << " index:" << index
                  << " with value:" << inputTensor[index] << std::endl;
        result[i] = inputTensor[index];
    }
    return 0;
}

int releaseBuffer(struct CustomNodeTensor* output) {
    std::cout << "DifferentOperationsCustomLibrary deleting "
              << output->name << " buffer" << std::endl;
    delete output->data;
    delete output->dims;
    return 0;
}

int releaseTensors(struct CustomNodeTensor* outputs) {
    std::cout << "DifferentOperationsCustomLibrary deleting outputs:" << outputs << std::endl;
    delete[] outputs;
    return 0;
}
}
