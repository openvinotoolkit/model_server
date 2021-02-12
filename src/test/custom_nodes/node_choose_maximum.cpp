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

static const size_t dummyInputSize = 10;

extern "C" {
enum METHOD {
    MAXIMUM_MINIMUM,
    MAXIMUM_AVERAGE,
    MAXIMUM_MAXIMUM,
    METHOD_COUNT
};

static const std::string INPUT_TENSOR_NAME = "input_tensors";

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    // choose selection criteria
    uint16_t selection_method = METHOD_COUNT;
    if (paramsLength != 1) {
        std::cout << "Wrong number of parameters - expected 1" << std::endl;
    }
    for (int i = 0; i < paramsLength; i++) {
        if (strcmp(params[i].key, "selection_criterium") == 0) {
            if (strcmp(params[i].value, "MAXIMUM_MINIMUM") == 0) {
                selection_method = MAXIMUM_MINIMUM;
            } else if (strcmp(params[i].value, "MAXIMUM_MAXIMUM") == 0) {
                selection_method = MAXIMUM_MAXIMUM;
            } else if (strcmp(params[i].value, "MAXIMUM_AVERAGE") == 0) {
                selection_method = MAXIMUM_AVERAGE;
            } else {
                std::cout << "Not allowed method chosen:" << params[i].value << std::endl;
                return 1;
            }
        } else {
            std::cout << "Non recognized param string" << std::endl;
            return 1;
        }
    }
    // TODO validate inputs
    *outputsLength = 1;
    *outputs = new CustomNodeTensor[*outputsLength];
    std::cout << "Creating outputs:" << *outputs << std::endl;
    // TODO check result
    CustomNodeTensor& resultTensor = *outputs[*outputsLength - 1];
    resultTensor.name = "maximum_tensor";
    float* result = new float[dummyInputSize];
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsLength = 2;
    resultTensor.dims = new uint64_t[resultTensor.dimsLength];
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = dummyInputSize;
    resultTensor.dataLength = 1 * dummyInputSize * sizeof(float);
    resultTensor.precision = FP32;
    // get input tensor
    float* inputTensor;
    for (size_t i = 0; i < inputsLength; ++i) {
        if (INPUT_TENSOR_NAME == inputs[i].name) {
            inputTensor = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "Lacking input:" << INPUT_TENSOR_NAME << std::endl;
            return 1;  // TODO free
        }
    }
    // get adjustable dim
    size_t tensorsCount = inputs[0].dims[1];
    // perform operations
    std::vector<float> minimums(tensorsCount, std::numeric_limits<int>::max());
    std::vector<float> maximums(tensorsCount, std::numeric_limits<int>::lowest());
    std::vector<float> averages(tensorsCount, 0);
    for (size_t opId = 0; opId < tensorsCount; ++opId) {  // TODO adjustable opsId
        for (size_t dummyPos = 0; dummyPos < dummyInputSize; ++dummyPos) {
            auto index = opId * dummyInputSize + dummyPos;
            switch (selection_method) {
            case METHOD::MAXIMUM_MAXIMUM:
                maximums[opId] = std::max(maximums[opId], inputTensor[index]);
                break;
            case METHOD::MAXIMUM_MINIMUM:
                minimums[opId] = std::min(maximums[opId], inputTensor[index]);
                break;
            case METHOD::MAXIMUM_AVERAGE:
                averages[opId] += inputTensor[index];
                break;
            default:
                return 2;
            }
            std::cout << "opId:" << opId
                      << " dummyPos:" << dummyPos
                      << " input:" << inputTensor[index]
                      << " minimums:" << minimums[opId]
                      << " averages:" << averages[opId]
                      << " maximums:" << maximums[opId]
                      << std::endl;
        }
        averages[opId] /= dummyInputSize;
    }
    // find which tensor to choose
    size_t whichTensor = 42;
    const std::vector<float>* fromWhichContainerToChoose = &maximums;
    switch (selection_method) {
    case METHOD::MAXIMUM_MAXIMUM:
        fromWhichContainerToChoose = &maximums;
        break;
    case METHOD::MAXIMUM_MINIMUM:
        fromWhichContainerToChoose = &minimums;
        break;
    case METHOD::MAXIMUM_AVERAGE:
        fromWhichContainerToChoose = &averages;
        break;
    default:
        // TODO cleanup of ptrs
        return 2;
    }
    whichTensor = std::distance(fromWhichContainerToChoose->begin(),
        std::max_element(fromWhichContainerToChoose->begin(),
            fromWhichContainerToChoose->end()));
    std::cout << "Selected tensor pos: " << whichTensor << std::endl;
    // copy appropiate tensor
    for (size_t i = 0; i < dummyInputSize; ++i) {
        size_t index = whichTensor * dummyInputSize + i;
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
