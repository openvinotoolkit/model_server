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
#include <string>

#include "../../custom_node_interface.h"

static const size_t dummyInputSize = 10;

extern "C" {
enum OPS {
ADD,
SUB,
MULTIPLY,
DIVIDE
};

static const std::string INPUT_TENSOR_NAME = "input_tensors";

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    // TODO validate inputs
    // TODO validate no params
    *outputsLength = 1;
    *outputs = new CustomNodeTensor[*outputsLength];
    std::cout << "Creating outputs:" << *outputs << std::endl;
    // TODO check result
    CustomNodeTensor& resultTensor = *outputs[*outputsLength - 1];
    resultTensor.name = "maximum_tensor";
    float* result = new float[4 * dummyInputSize]; // dummy input size * numbr of ops
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsLength = 2;
    resultTensor.dims = new uint64_t[resultTensor.dimsLength];
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = dummyInputSize;
    resultTensor.dataLength = 1 * dummyInputSize * sizeof(float);
    resultTensor.precision = FP32;

    uint16_t selection_method = METHOD_COUNT;
    if ( paramsLength != 1) {
        std::cout << "Wrong number of parameters - expected 1" << std::endl;
    }
    for (int i = 0; i < paramsLength; i++) {
        if (strcmp(params[i].key, "selection_criterium") == 0) {
   //         addValue = atof(params[i].value);
            selection_method = 1;
        }
    }
    if (selection_method == METHOD_COUNT) {
        std::cout << "Non recognized method string" << std::endl;
        return 1;
    }
    // perform operations
    float* inputTensor;
    float* inputFactors;
    for (size_t i = 0; i < inputsLength; ++i) {
        if (INPUT_TENSOR_NAME == inputs[i].name) {
            inputTensor = reinterpret_cast<float*>(inputs[i].data);
        } else if (FACTORS_TENSOR_NAME == inputs[i].name) {
            inputFactors = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "lacking inputs" << std::endl;
            return 1; // TODO free
        }
    }
    for (size_t opId = 0; opId < 4; ++opId) {
        for (size_t dummyPos = 0; dummyPos < dummyInputSize; ++dummyPos) {
            // TODO get dummyInputSize from input
            auto resultIndex = opId * dummyInputSize + dummyPos;
            switch(opId) {
                case OPS::ADD:
                    result[resultIndex] = inputTensor[dummyPos] + inputFactors[opId];
                    break;
                case OPS::SUB:
                    result[resultIndex] = inputTensor[dummyPos] - inputFactors[opId];
                    break;
                case MULTIPLY:
                    result[resultIndex] = inputTensor[dummyPos] * inputFactors[opId];
                    break;
                case DIVIDE:
                    result[resultIndex] = inputTensor[dummyPos] / inputFactors[opId];
                    break;
                default:
                    // TODO cleanup of ptrs
                    return 2;
            }
            std::cout << "opId:" << opId
                      << " dummyPos:" << dummyPos
                      << " resultIndex:" << resultIndex
                      << " result:" << result[resultIndex]
                      << " inputTensor:" << inputTensor[dummyPos]
                      << " inputFactor:" << inputFactors[opId]
                      << std::endl;
        }
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
