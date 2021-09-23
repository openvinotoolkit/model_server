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
enum OPS {
    ADD,
    SUB,
    MULTIPLY,
    DIVIDE,
    OPS_END
};

static const std::string INPUT_TENSOR_NAME = "input_numbers";
static const std::string FACTORS_TENSOR_NAME = "op_factors";

static const char* OUTPUT_TENSOR_NAME = "different_ops_results";
static const char* OUTPUT_FACTORS_NAME = "different_ops_factors";

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
    float* inputFactors = nullptr;
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
        } else if (FACTORS_TENSOR_NAME == inputs[i].name) {
            if (inputs[i].dimsCount != 2 ||
                inputs[i].dims[0] != 1 ||
                inputs[i].dims[1] != OPS_END) {
                ss << "improper " << FACTORS_TENSOR_NAME
                   << " dimensions:[" << inputs[i].dims[0] << ", " << inputs[i].dims[1] << "]" << std::endl;
                std::cout << ss.str() << std::endl;
                return 1;
            }
            inputFactors = reinterpret_cast<float*>(inputs[i].data);
        } else {
            std::cout << "Unexpected input" << inputs[i].name << std::endl;
            return 1;
        }
    }
    if ((!inputTensor) || (!inputFactors)) {
        std::cout << "lacking inputs" << std::endl;
        return 1;
    }

    // prepare outputs
    *outputsCount = 2;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    float* result = (float*)malloc(OPS_END * valuesPerTensor * sizeof(float));  // dummy input size * number of ops
    float* resultFactors = (float*)malloc(OPS_END * OPS_END * sizeof(float));   // dummy input size * number of ops

    CustomNodeTensor& resultTensor = (*outputs)[0];
    resultTensor.name = OUTPUT_TENSOR_NAME;
    resultTensor.data = reinterpret_cast<uint8_t*>(result);
    resultTensor.dimsCount = 3;
    resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsCount * sizeof(uint64_t));
    resultTensor.dims[0] = OPS_END;
    resultTensor.dims[1] = 1;
    resultTensor.dims[2] = valuesPerTensor;
    resultTensor.dataBytes = resultTensor.dims[0] * resultTensor.dims[1] * resultTensor.dims[2] * sizeof(float);
    resultTensor.precision = FP32;

    CustomNodeTensor& resultFactorsTensor = (*outputs)[1];
    resultFactorsTensor.name = OUTPUT_FACTORS_NAME;
    resultFactorsTensor.data = reinterpret_cast<uint8_t*>(resultFactors);
    resultFactorsTensor.dimsCount = 3;
    resultFactorsTensor.dims = (uint64_t*)malloc(resultFactorsTensor.dimsCount * sizeof(uint64_t));
    resultFactorsTensor.dims[0] = OPS_END;
    resultFactorsTensor.dims[1] = 1;
    resultFactorsTensor.dims[2] = OPS_END;
    resultFactorsTensor.dataBytes = resultFactorsTensor.dims[0] * resultFactorsTensor.dims[1] * resultFactorsTensor.dims[2] * sizeof(float);
    resultFactorsTensor.precision = FP32;
    // perform operations
    for (size_t opId = 0; opId < OPS_END; ++opId) {
        for (size_t factorsPos = 0; factorsPos < OPS_END; ++factorsPos) {
            resultFactors[opId * OPS_END + factorsPos] = inputFactors[factorsPos];
        }
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
            ss << "opId:" << opId
               << " dummyPos:" << dummyPos
               << " resultIndex:" << resultIndex
               << " result:" << result[resultIndex]
               << " inputTensor:" << inputTensor[dummyPos]
               << " inputFactor:" << inputFactors[opId]
               << std::endl;
        }
    }
    std::cout << ss.str() << std::endl;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = INPUT_TENSOR_NAME.c_str();
    (*info)[0].precision = FP32;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 10;

    (*info)[1].name = FACTORS_TENSOR_NAME.c_str();
    (*info)[1].precision = FP32;
    (*info)[1].dimsCount = 2;
    (*info)[1].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = 4;

    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)[0].name = OUTPUT_TENSOR_NAME;
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = OPS_END;
    (*info)[0].dims[1] = 1;
    (*info)[0].dims[2] = 10;
    (*info)[0].precision = FP32;

    (*info)[1].name = OUTPUT_FACTORS_NAME;
    (*info)[1].dimsCount = 3;
    (*info)[1].dims = (uint64_t*)malloc((*info)[1].dimsCount * sizeof(uint64_t));
    (*info)[1].dims[0] = OPS_END;
    (*info)[1].dims[1] = 1;
    (*info)[1].dims[2] = OPS_END;
    (*info)[1].precision = FP32;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    std::cout << "DifferentOperationsCustomLibrary release" << std::endl;
    free(ptr);
    return 0;
}
}
