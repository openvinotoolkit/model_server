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
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "custom_node_interface.h"  // NOLINT
#include "model.hpp"
#include "utils.hpp"

#define DEBUG_MSG(str)                                     \
    if (debugMode) {                                       \
        std::cout << "[detokenizer] " << str << std::endl; \
    }

#define INPUT_NAME_LOGITS "logits"
#define INPUT_NAME_PREVIOUS_TOKENS "input_ids"
#define INPUT_NAME_PREVIOUS_ATTENTION "attention_mask"

#define OUTPUT_NAME_TEXTS "texts"

// Size of memory allocation on the heap for generated text.
// If the size of the output is larger than this value, the output is truncated.
// Consider using memory pool.
#define DEFAULT_MAX_BUF_LEN 4096

using namespace custom_nodes::tokenizer;

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
        std::cerr << "[detokenizer] initialize() fail: Cannot load tokenization model from path: " << modelPath << std::endl;
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
    const CustomNodeTensor** logitsTensor,
    const CustomNodeTensor** inputIdsTensor,
    const CustomNodeTensor** attentionMaskTensor) {
    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, INPUT_NAME_LOGITS) == 0) {
            *logitsTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, INPUT_NAME_PREVIOUS_TOKENS) == 0) {
            *inputIdsTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, INPUT_NAME_PREVIOUS_ATTENTION) == 0) {
            *attentionMaskTensor = &(inputs[i]);
        } else {
            std::cerr << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }
    return 0;
}

static int validateInputs(
    const CustomNodeTensor* logitsTensor,
    const CustomNodeTensor* inputIdsTensor,
    const CustomNodeTensor* attentionMaskTensor) {
    NODE_ASSERT(logitsTensor != nullptr, "Missing " INPUT_NAME_LOGITS " input");
    NODE_ASSERT(logitsTensor->precision == FP32, INPUT_NAME_LOGITS " input is not FP32");
    NODE_ASSERT(logitsTensor->dimsCount == 3, "input " INPUT_NAME_LOGITS " shape must have 3 dimensions");
    NODE_ASSERT(logitsTensor->dims[0] > 0, "input " INPUT_NAME_LOGITS " dimension 1 must be larger than 0");
    NODE_ASSERT(logitsTensor->dims[1] > 0, "input " INPUT_NAME_LOGITS " dimension 2 must be larger than 0");
    NODE_ASSERT(logitsTensor->dims[2] > 0, "input " INPUT_NAME_LOGITS " text dimension 3 must be larger than 0");

    NODE_ASSERT(inputIdsTensor != nullptr, "Missing " INPUT_NAME_PREVIOUS_TOKENS " input");
    NODE_ASSERT(inputIdsTensor->precision == I64, INPUT_NAME_PREVIOUS_TOKENS " input is not I64");
    NODE_ASSERT(inputIdsTensor->dimsCount == 2, INPUT_NAME_PREVIOUS_TOKENS " shape must have 2 dimensions");
    NODE_ASSERT(inputIdsTensor->dims[0] > 0, INPUT_NAME_PREVIOUS_TOKENS " dimension 1 must be larger than 0");
    NODE_ASSERT(inputIdsTensor->dims[1] > 0, INPUT_NAME_PREVIOUS_TOKENS " dimension 2 must be larger than 0");

    NODE_ASSERT(attentionMaskTensor != nullptr, "Missing " INPUT_NAME_PREVIOUS_ATTENTION " input");
    NODE_ASSERT(attentionMaskTensor->precision == I64, INPUT_NAME_PREVIOUS_ATTENTION " input is not I64");
    NODE_ASSERT(attentionMaskTensor->dimsCount == 2, INPUT_NAME_PREVIOUS_ATTENTION " shape must have 2 dimensions");
    NODE_ASSERT(attentionMaskTensor->dims[0] > 0, INPUT_NAME_PREVIOUS_ATTENTION " dimension 1 must be larger than 0");
    NODE_ASSERT(attentionMaskTensor->dims[1] > 0, INPUT_NAME_PREVIOUS_ATTENTION " dimension 2 must be larger than 0");

    NODE_ASSERT(logitsTensor->dims[0] == inputIdsTensor->dims[0], INPUT_NAME_LOGITS " and " INPUT_NAME_PREVIOUS_TOKENS " need to have matching batch dimension");
    NODE_ASSERT(logitsTensor->dims[0] == attentionMaskTensor->dims[0], INPUT_NAME_LOGITS " and " INPUT_NAME_PREVIOUS_ATTENTION " need to have matching batch dimension");

    NODE_ASSERT(logitsTensor->dims[1] == inputIdsTensor->dims[1], INPUT_NAME_LOGITS " and " INPUT_NAME_PREVIOUS_TOKENS " need to have matching second dimension");
    NODE_ASSERT(logitsTensor->dims[1] == attentionMaskTensor->dims[1], INPUT_NAME_LOGITS " and " INPUT_NAME_PREVIOUS_ATTENTION " need to have matching second dimension");
    return 0;
}

// in:  [-1, -1, 50400]
// out: [Batch, MaxLength]
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto start = std::chrono::steady_clock::now();
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";
    DEBUG_MSG("execute() start");
    // Parameters reading
    int maxBufferLength = get_int_parameter("max_buffer_length", params, paramsCount, DEFAULT_MAX_BUF_LEN);
    NODE_ASSERT(maxBufferLength > 0, "max_buffer_length param must be larger than 0");

    const CustomNodeTensor* logitsTensor = nullptr;
    const CustomNodeTensor* inputIdsTensor = nullptr;
    const CustomNodeTensor* attentionMaskTensor = nullptr;

    NODE_ASSERT(retrieveInputs(inputs, inputsCount, &logitsTensor, &inputIdsTensor, &attentionMaskTensor) == 0, "retrieveInputs() failed");
    NODE_ASSERT(validateInputs(logitsTensor, inputIdsTensor, attentionMaskTensor) == 0, "validateInputs() failed");

    BlingFireModel* model = static_cast<BlingFireModel*>(customNodeLibraryInternalManager);

    std::vector<std::string> results;
    for (uint64_t batch = 0; batch < logitsTensor->dims[0]; batch++) {
        // get previous tokens of current batch for context
        DEBUG_MSG("get previous tokens of batch " << batch);
        int64_t* inputIds = reinterpret_cast<int64_t*>(
            inputIdsTensor->data +
            batch * (inputIdsTensor->dims[1] * sizeof(int64_t)));
        int64_t* attentionMask = reinterpret_cast<int64_t*>(
            attentionMaskTensor->data +
            batch * (attentionMaskTensor->dims[1] * sizeof(int64_t)));

        int64_t* it = std::find(attentionMask, attentionMask + attentionMaskTensor->dims[1], 0);
        std::ptrdiff_t distance = std::distance(attentionMask, it);
        std::ptrdiff_t lastNonZeroIndex = distance - 1;

        // case for empty string being in a batch (attention mask all zeros)
        if (lastNonZeroIndex < 0)
            lastNonZeroIndex = 0;

        std::vector<int64_t> previousTokens(inputIds, inputIds + distance);

        // slice
        DEBUG_MSG("slicing batch " << batch);
        float* logits = reinterpret_cast<float*>(
            logitsTensor->data +
            batch * (logitsTensor->dims[1] * logitsTensor->dims[2] * sizeof(float)) +  // offset by batch
            (lastNonZeroIndex * logitsTensor->dims[2] * sizeof(float)));               // offset to get last element of second dimension

        // argmax
        DEBUG_MSG("argmax batch " << batch);
        float* result = std::max_element(logits, logits + logitsTensor->dims[2]);
        int64_t token = std::distance(logits, result);
        previousTokens.push_back(token);

        // detokenize
        DEBUG_MSG("detokenizing token batch " << batch);
        auto text = model->detokenize(previousTokens, maxBufferLength);
        DEBUG_MSG("detokenized token: (" << token << ") to: (" << text << ") for batch " << batch);
        results.emplace_back(std::move(text));
    }

    DEBUG_MSG("getting max string length");
    size_t maxStringLength = 0;
    for (const auto& str : results) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    size_t width = maxStringLength + 1;

    DEBUG_MSG("prepraing output tensor");
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        return 1;
    }

    // Outputs allocation
    CustomNodeTensor& output = (*outputs)[0];
    output.name = OUTPUT_NAME_TEXTS;
    output.dataBytes = width * results.size();
    output.data = (uint8_t*)malloc(output.dataBytes);
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = results.size();
    output.dims[1] = width;
    output.precision = U8;

    DEBUG_MSG("writing output");
    for (size_t i = 0; i < results.size(); i++) {
        std::memcpy(output.data + i * width, results[i].data(), results[i].size());
        output.data[i * width + results[i].size()] = 0;
    }
    DEBUG_MSG("execute() end");
    auto end = std::chrono::steady_clock::now();
    DEBUG_MSG("execute() end; took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << " ms");
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 3;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = INPUT_NAME_LOGITS;
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].dims[2] = -1;
    (*info)[0].precision = FP32;

    (*info)[1].name = INPUT_NAME_PREVIOUS_TOKENS;
    (*info)[1].dimsCount = 2;
    (*info)[1].dims = (uint64_t*)malloc((*info)[1].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = -1;
    (*info)[1].dims[1] = -1;
    (*info)[1].precision = I64;

    (*info)[2].name = INPUT_NAME_PREVIOUS_ATTENTION;
    (*info)[2].dimsCount = 2;
    (*info)[2].dims = (uint64_t*)malloc((*info)[2].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[2].dims[0] = -1;
    (*info)[2].dims[1] = -1;
    (*info)[2].precision = I64;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = OUTPUT_NAME_TEXTS;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = U8;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
