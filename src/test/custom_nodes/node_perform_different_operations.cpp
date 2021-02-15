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

extern "C" {
enum OPS {
    ADD,
    SUB,
    MULTIPLY,
    DIVIDE,
    OPS_END
};

static const std::string INPUT_TENSOR_NAME = "input_numbers";
static const std::string FACTORS_TENSOR_NAME = "op_factors";

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    // validate inputs
    float* inputTensor;
    float* inputFactors;
    size_t valuesPerTensor = 0;
    for (size_t i = 0; i < inputsLength; ++i) {
        if (INPUT_TENSOR_NAME == inputs[i].name) {
            if (inputs[i].dimsLength != 2 ||
                inputs[i].dims[0] != 1 ||
                inputs[i].dims[1] == 0) {
                std::cout << "improper " << INPUT_TENSOR_NAME
                          << " dimensions: [" << inputs[i].dims[0] << ", " << inputs[i].dims[1] << "]" << std::endl;
                return 1;
            }
            std::cout << "Input valuesPerTensor" << inputs[i].dims[1] << std::endl;
            valuesPerTensor = inputs[i].dims[1];
            inputTensor = reinterpret_cast<float*>(inputs[i].data);
        } else if (FACTORS_TENSOR_NAME == inputs[i].name) {
            if (inputs[i].dimsLength != 2 ||
                inputs[i].dims[0] != 1 ||
                inputs[i].dims[1] != OPS_END) {
                std::cout << "improper " << FACTORS_TENSOR_NAME
                          << " dimensions:[" << inputs[i].dims[0] << ", " << inputs[i].dims[1] << "]" << std::endl;
                return 1;
            }
            inputFactors = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "lacking inputs" << std::endl;
            return 1;
        }
    }

    // prepare outputs
    *outputsLength = 1;
    *outputs = new CustomNodeTensor[*outputsLength];
    float* result = new float[OPS_END * valuesPerTensor];  // dummy input size * numbr of ops

    CustomNodeTensor& resultTensor = *outputs[*outputsLength - 1];
    resultTensor.name = "different_ops_results";
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsLength = 3;
    resultTensor.dims = new uint64_t[resultTensor.dimsLength];
    resultTensor.dims[0] = 1;
    resultTensor.dims[1] = OPS_END;
    resultTensor.dims[2] = valuesPerTensor;
    resultTensor.dataLength = resultTensor.dims[0] * resultTensor.dims[1] * resultTensor.dims[2] * sizeof(float);
    resultTensor.precision = FP32;

    // perform operations
    for (size_t opId = 0; opId < OPS_END; ++opId) {
        for (size_t dummyPos = 0; dummyPos < valuesPerTensor; ++dummyPos) {
            auto resultIndex = opId * valuesPerTensor + dummyPos;
            switch (opId) {
            case OPS::ADD:
                result[resultIndex] = inputTensor[dummyPos] + inputFactors[opId];
                break;
            case OPS::SUB:
                result[resultIndex] = inputTensor[dummyPos] - inputFactors[opId];
                break;
            case OPS::MULTIPLY:
                result[resultIndex] = inputTensor[dummyPos] * inputFactors[opId];
                break;
            case OPS::DIVIDE:
                result[resultIndex] = inputTensor[dummyPos] / inputFactors[opId];
                break;
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
