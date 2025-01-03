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
#pragma once

#ifdef __linux__
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
#elif _WIN32
#define DLL_PUBLIC __declspec(dllexport)
#define DLL_LOCAL __declspec(dllimport)
#endif

#include <stdint.h>
typedef enum {
    UNSPECIFIED,
    FP32,
    FP16,
    U8,
    I8,
    I16,
    U16,
    I32,
    FP64,
    I64
} CustomNodeTensorPrecision;

struct CustomNodeTensor {
    const char* name;
    uint8_t* data;
    uint64_t dataBytes;
    uint64_t* dims;
    uint64_t dimsCount;
    CustomNodeTensorPrecision precision;
};

struct CustomNodeTensorInfo {
    const char* name;
    uint64_t* dims;
    uint64_t dimsCount;
    CustomNodeTensorPrecision precision;
};

struct CustomNodeParam {
    const char *key, *value;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Custom node library initialize enables creation of resources to be reused between predictions.
 * Potential use cases include optimized temporary buffers allocation.
 * Using initialize is optional and not required for custom node to work.
 * CustomNodeLibraryInternalManager should be created here if initialize is used.
 * On initialize failure status not equal to zero is returned and error log is printed.
 */
DLL_PUBLIC DLL_PUBLIC int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
/**
 * @brief Custom node library deinitialize enables destruction of resources that were used between predictions.
 * Using deinitialize is optional and not required for custom node to work.
 * CustomNodeLibraryInternalManager should be destroyed here if deinitialize is used.
 * On deinitialize failure only error log is printed.
 */
DLL_PUBLIC DLL_PUBLIC int deinitialize(void* customNodeLibraryInternalManager);
DLL_PUBLIC DLL_PUBLIC int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
DLL_PUBLIC DLL_PUBLIC int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
DLL_PUBLIC DLL_PUBLIC int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
DLL_PUBLIC DLL_PUBLIC int release(void* ptr, void* customNodeLibraryInternalManager);

#ifdef __cplusplus
}
#endif
