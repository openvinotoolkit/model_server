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
#include <iostream>
#include <vector>
#include <string>
#include <regex>

#include "../../custom_node_interface.h"
#include "opencv2/opencv.hpp"

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }

#define LOG_PREFIX "[CustomNode] [" << __LINE__ << "] elastic: "

#ifdef LOG
#define LOG_PREFIX "[CustomNode] [" << __LINE__ << "] elastic: "
#define LOG_ENTRY_M(method) std::cout << LOG_PREFIX << "Method: " << method << " entry" << std::endl;
#define LOG_EXIT_M(method) std::cout << LOG_PREFIX << "Method: " << method << " exit" << std::endl;
#else
#define LOG_ENTRY_M(method) 
;
#define LOG_EXIT_M(method) ;
#endif

static constexpr const char* TENSOR_IN = "tensor_in";
static constexpr const char* TENSOR_OUT = "tensor_out";

const char* get_param_value(const std::string& param_name, const struct CustomNodeParam* params, int params_count, const char* default_value) {
    LOG_ENTRY_M("get_param_value");
    const char* result = default_value;
    for (int i = 0; i < params_count; i++) {
        if(param_name == params[i].key) {
            result = params[i].value;
            break;
        }
    }
    std::cout << LOG_PREFIX << "name: " << param_name << "; value: " << result << std::endl;
    LOG_EXIT_M("get_param_value");
    return result;
}

std::vector<std::string> parse_tokens(const char* str, std::smatch &match, std::regex reg)
{
    std::vector<std::string> result;

    std::string input_copy(str);
    while(std::regex_search(input_copy, match, reg))
    {
        result.push_back(match[1].str());
        input_copy = match.suffix().str();
    }
    return result;
}

std::vector<int> parse_shape(const char* msg)
{
    std::vector<int> result;
    std::smatch match;   
    const std::regex dim_re_value("(\\d+)");

    std::vector<std::string> dim_string_value = parse_tokens(msg, match, dim_re_value);
    for(size_t i = 0; i < dim_string_value.size(); i++){
        int dim_size = std::stoi(dim_string_value[(int)i]);
        result.push_back(dim_size);
    }
    return result;
}

int get_tensor_info(struct CustomNodeTensorInfo** info,
                        int* info_count,
                        const CustomNodeParam* params,
                        int params_count,
                        const char* param_name,
                        const char* default_value,
                        const char* tensor_name)
{
    LOG_ENTRY_M("get_tensor_info");
    const char* shape_str = get_param_value(param_name, params, params_count, default_value);
    std::vector<int> shape = parse_shape(shape_str);

    *info_count = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*info_count * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    std::cout << LOG_PREFIX << "allocate info: "<< info << std::endl;

    CustomNodeTensorInfo& tensor = (*info)[0];
    tensor.name = tensor_name;
    tensor.dimsCount = static_cast<int>(shape.size());
    tensor.dims = (uint64_t*)malloc(tensor.dimsCount * sizeof(uint64_t));
    NODE_ASSERT((tensor.dims) != nullptr, "malloc has failed");
    std::cout << LOG_PREFIX << "allocate dims: "<< tensor.dims << std::endl;

    for(int i = 0; i < (int)tensor.dimsCount; i++)
        tensor.dims[i] = shape[i];
    tensor.precision = FP32;

    LOG_EXIT_M("get_tensor_info");
    return 0;
}

void release_tensor_info(struct CustomNodeTensorInfo* info, int info_count)
{
    LOG_ENTRY_M("release_tensor_info");
    for(int i = 0; i < info_count; i ++)
        free(info[i].dims);
    free(info);
    LOG_EXIT_M("release_tensor_info");
}

int execute(const struct CustomNodeTensor* inputs,
            int inputsCount,
            struct CustomNodeTensor** outputs,
            int* outputsCount,
            const struct CustomNodeParam* params,
            int paramsCount,
            void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("execute");
    struct CustomNodeTensorInfo* info = NULL;
    int result = get_tensor_info(&info, outputsCount, params, paramsCount, "output_shape", "[10, 10]", TENSOR_OUT);

    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    NODE_ASSERT((*outputs) != nullptr, "malloc has failed");
    std::cout << LOG_PREFIX << "allocate outputs: "<< outputs << std::endl;

    CustomNodeTensor& tensor = (*outputs)[0];
    uint32_t byte_size = sizeof(uint32_t);
    for(int i = 0; i < (int)info[0].dimsCount; i++)
        byte_size *= (uint32_t)info[0].dims[i];
    tensor.name = TENSOR_OUT;

    float* buffer = (float*)malloc(byte_size);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");
    std::cout << LOG_PREFIX << "allocate data: " << buffer << std::endl;

    tensor.data = reinterpret_cast<uint8_t*>(buffer);
    tensor.dataBytes = byte_size;
    tensor.dimsCount = info[0].dimsCount;
    tensor.dims = (uint64_t*)malloc(tensor.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(tensor.dims != nullptr, "malloc has failed");
    std::cout << LOG_PREFIX << "allocate shape: " << tensor.dims << std::endl;
    for(uint64_t i = 0; i < info[0].dimsCount; i++)
        tensor.dims[i] = info[0].dims[(int)i];
    tensor.precision = info[0].precision;

    release_tensor_info(info, *outputsCount);
    LOG_EXIT_M("execute");
    return result;
}

int getInputsInfo(struct CustomNodeTensorInfo** info,
                  int* infoCount,
                  const struct CustomNodeParam* params,
                  int paramsCount,
                  void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("getInputsInfo");
    int result = get_tensor_info(info, infoCount, params, paramsCount, "input_shape", "[1, 10]", TENSOR_IN);
    LOG_EXIT_M("getInputsInfo");
    return result;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info,
                   int* infoCount,
                   const struct CustomNodeParam* params,
                   int paramsCount,
                   void* customNodeLibraryInternalManager){
    LOG_ENTRY_M("getOutputsInfo");
    int result = get_tensor_info(info, infoCount, params, paramsCount, "output_shape", "[1, 10]", TENSOR_OUT);
    LOG_EXIT_M("getOutputsInfo");
    return result;
}

int release(void* ptr, void* customNodeLibraryInternalManager)
{
    std::cout << LOG_PREFIX << "release(" << ptr << ")" << std::endl;
    free(ptr);
    return 0;
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}
