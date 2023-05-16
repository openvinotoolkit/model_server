//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define __STDC_WANT_LIB_EXT1__ 1  // to ensure existence of strnlen
#include <string.h>

#include "custom_node_interface.h"  // NOLINT
#include "model.hpp"
#include "utils.hpp"

#define INPUT_NAME_TEXTS "texts"

#define OUTPUT_NAME_TOKENS "input_ids"
#define OUTPUT_NAME_ATTENTION "attention_mask"

// Size of memory allocation on the heap for generated tokens.
// If the size of the output is larger than this value, the output is truncated.
// Consider using memory pool.
#define DEFAULT_MAX_ID_ARR_LEN 1024

using namespace custom_nodes::tokenizer;

#define DEBUG_MSG(str)                                   \
    if (debugMode) {                                     \
        std::cout << "[tokenizer] " << str << std::endl; \
    }

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";
    std::string modelPath = get_string_parameter("model_path", params, paramsCount, "");
    NODE_ASSERT(!modelPath.empty(), "model_path cannot be empty");
    try {
        auto cnlim = std::make_unique<BlingFireModel>(modelPath, debugMode);
        if (!cnlim->isValid())
            throw std::exception();
        *customNodeLibraryInternalManager = cnlim.release();
    } catch (...) {
        std::cerr << "[tokenizer] initialize() fail: Cannot load tokenization model from path: " << modelPath << std::endl;
        return 1;
    }
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    if (customNodeLibraryInternalManager != nullptr) {
        BlingFireModel* manager = static_cast<BlingFireModel*>(customNodeLibraryInternalManager);
        delete manager;
    }
    return 0;
}

static int retrieveInputs(
    // in
    const struct CustomNodeTensor* inputs,
    int inputsCount,
    // out
    const CustomNodeTensor** textTensor) {
    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, INPUT_NAME_TEXTS) == 0) {
            *textTensor = &(inputs[i]);
        } else {
            std::cerr << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }
    return 0;
}

static int validateInputs(const CustomNodeTensor* textTensor) {
    NODE_ASSERT(textTensor != nullptr, "Missing " INPUT_NAME_TEXTS " input");
    NODE_ASSERT(textTensor->precision == U8, INPUT_NAME_TEXTS " input is not U8");

    NODE_ASSERT(textTensor->dimsCount == 2, INPUT_NAME_TEXTS " inout shape must have 2 dimensions");
    NODE_ASSERT(textTensor->dims[0] > 0, INPUT_NAME_TEXTS " input dimension 1 must be larger than 0 (number of texts)");
    NODE_ASSERT(textTensor->dims[1] > 0, INPUT_NAME_TEXTS " input dimension 2 must be larger than 0 (max null terminated text length)");
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto start = std::chrono::steady_clock::now();
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";
    DEBUG_MSG("execute() start");
    // Parameters reading
    int maxIdsArrLength = get_int_parameter("max_ids_arr_length", params, paramsCount, DEFAULT_MAX_ID_ARR_LEN);
    NODE_ASSERT(maxIdsArrLength > 0, "max_ids_arr_length param must be larger than 0");

    const CustomNodeTensor* textTensor = nullptr;

    NODE_ASSERT(retrieveInputs(inputs, inputsCount, &textTensor) == 0, "retrieveInputs() failed");
    NODE_ASSERT(validateInputs(textTensor) == 0, "validateInputs() failed");

    BlingFireModel* model = static_cast<BlingFireModel*>(customNodeLibraryInternalManager);

    *outputsCount = 2;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        return 1;
    }

    std::vector<std::vector<int64_t>> ids(textTensor->dims[0]);
    // For each batch, sequentially
    for (uint64_t batch = 0; batch < textTensor->dims[0]; batch++) {
        DEBUG_MSG("tokenizing batch " << batch);
        const char* strStart = (const char*)textTensor->data + batch * textTensor->dims[1];
        std::string text(strStart, strnlen(strStart, textTensor->dims[1]));
        ids[batch] = model->tokenize(text, maxIdsArrLength);
        DEBUG_MSG("tokenized batch " << batch << "; of string: " << text);
    }

    DEBUG_MSG("getting max token size");
    size_t maxTokenSize = 0;
    for (const auto& id : ids) {
        maxTokenSize = std::max(maxTokenSize, id.size());
    }

    DEBUG_MSG("preparing output tensors");
    CustomNodeTensor& tokens = (*outputs)[0];
    tokens.name = OUTPUT_NAME_TOKENS;
    tokens.dataBytes = sizeof(int64_t) * maxTokenSize * ids.size();
    tokens.data = (uint8_t*)malloc(tokens.dataBytes);
    tokens.dimsCount = 2;
    tokens.dims = (uint64_t*)malloc(tokens.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(tokens.dims != nullptr, "malloc has failed");
    tokens.dims[0] = ids.size();
    tokens.dims[1] = maxTokenSize;
    tokens.precision = I64;

    CustomNodeTensor& attention = (*outputs)[1];
    attention.name = OUTPUT_NAME_ATTENTION;
    attention.dataBytes = sizeof(int64_t) * maxTokenSize * ids.size();
    attention.data = (uint8_t*)malloc(attention.dataBytes);
    attention.dimsCount = 2;
    attention.dims = (uint64_t*)malloc(attention.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(attention.dims != nullptr, "malloc has failed");
    attention.dims[0] = ids.size();
    attention.dims[1] = maxTokenSize;
    attention.precision = I64;

    DEBUG_MSG("writing output");
    for (size_t i = 0; i < ids.size(); i++) {
        std::memcpy(tokens.data + i * maxTokenSize * sizeof(int64_t), ids[i].data(), ids[i].size() * sizeof(int64_t));
        for (size_t j = 0; j < ids[i].size(); j++) {
            ((int64_t*)attention.data)[i * maxTokenSize + j] = 1;
        }
        for (size_t j = ids[i].size(); j < maxTokenSize; j++) {
            ((int64_t*)attention.data)[i * maxTokenSize + j] = 0;
        }
    }
    auto end = std::chrono::steady_clock::now();
    DEBUG_MSG("execute() end; took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << " ms");
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = INPUT_NAME_TEXTS;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = U8;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = OUTPUT_NAME_TOKENS;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = I64;

    (*info)[1].name = OUTPUT_NAME_ATTENTION;
    (*info)[1].dimsCount = 2;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = -1;
    (*info)[1].dims[1] = -1;
    (*info)[1].precision = I64;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
