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

static const size_t DUMMY_INPUT_SIZE = 10;

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
        if (strcmp(params[0].value, "MAXIMUM_MINIMUM") == Method::MAXIMUM_MINIMUM) {
            selectionMethod = Method::MAXIMUM_MINIMUM;
        } else if (strcmp(params[0].value, "MAXIMUM_MAXIMUM") == Method::MAXIMUM_MAXIMUM) {
            selectionMethod = Method::MAXIMUM_MAXIMUM;
        } else if (strcmp(params[0].value, "MAXIMUM_AVERAGE") == Method::MAXIMUM_AVERAGE) {
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
    for (size_t i = 0; i < inputsLength; ++i) {
        if (INPUT_TENSOR_NAME == inputs[i].name) {
            inputTensor = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "Lacking input:" << INPUT_TENSOR_NAME << std::endl;
            return 1;  // TODO free
        }
    }

    // prepare output
    *outputsLength = 1;
    *outputs = new CustomNodeTensor[*outputsLength];
    float* result = new float[DUMMY_INPUT_SIZE];

    CustomNodeTensor& resultTensor = **outputs;
    resultTensor.name = "maximum_tensor";
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsLength = 2;
    resultTensor.dims = new uint64_t[resultTensor.dimsLength];
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = DUMMY_INPUT_SIZE;
    resultTensor.dataLength = resultTensor.dims[0] * resultTensor.dims[1] * sizeof(float);
    resultTensor.precision = FP32;

    // get adjustable dim
    size_t tensorsCount = inputs[0].dims[1];
    // perform operations
    std::vector<float> minimums(tensorsCount, std::numeric_limits<int>::max());
    std::vector<float> maximums(tensorsCount, std::numeric_limits<int>::lowest());
    std::vector<float> averages(tensorsCount, 0);
    for (size_t opId = 0; opId < tensorsCount; ++opId) {  // TODO adjustable opsId
        for (size_t dummyPos = 0; dummyPos < DUMMY_INPUT_SIZE; ++dummyPos) {
            auto index = opId * DUMMY_INPUT_SIZE + dummyPos;
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
            }
            std::cout << "opId:" << opId
                      << " dummyPos:" << dummyPos
                      << " input:" << inputTensor[index]
                      << " minimums:" << minimums[opId]
                      << " averages:" << averages[opId]
                      << " maximums:" << maximums[opId]
                      << std::endl;
        }
        averages[opId] /= DUMMY_INPUT_SIZE;
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
    }
    size_t whichTensor = std::distance(fromWhichContainerToChoose->begin(),
        std::max_element(fromWhichContainerToChoose->begin(),
            fromWhichContainerToChoose->end()));
    std::cout << "Selected tensor pos: " << whichTensor << std::endl;
    // copy appropiate tensor
    for (size_t i = 0; i < DUMMY_INPUT_SIZE; ++i) {
        size_t index = whichTensor * DUMMY_INPUT_SIZE + i;
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
