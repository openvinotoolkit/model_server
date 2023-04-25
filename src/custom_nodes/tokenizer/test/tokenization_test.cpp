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
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "custom_node_interface.h"  // NOLINT
#include "model.hpp"

#define TEST_MODEL_FILE_PATH "./gpt2.bin"

#define INPUT_NAME_TEXTS "texts"

#define OUTPUT_NAME_TOKENS "input_ids"
#define OUTPUT_NAME_ATTENTION "attention_mask"

using namespace custom_nodes::tokenizer;

TEST(TokenizerTest, Run) {
    BlingFireModel model(TEST_MODEL_FILE_PATH);
    int maxIdArrLen = 1024;
    auto result = model.tokenize("こんにちは", maxIdArrLen);
    std::vector<int64_t> expected = {23294, 241, 22174, 28618, 2515, 94, 31676};
    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], expected[i]) << "expected: " << expected[i] << "; actual: " << result[i];
    }
}

TEST(TokenizerTest, Run_TooSmallBuffer) {
    BlingFireModel model(TEST_MODEL_FILE_PATH);
    int maxIdArrLen = 4;
    auto result = model.tokenize("こんにちは", maxIdArrLen);
    std::vector<int64_t> expected = {23294, 241, 22174, 28618};
    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], expected[i]) << "expected: " << expected[i] << "; actual: " << result[i];
    }
}

TEST(TokenizerTest, init_deinit) {
    void* model = nullptr;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = TEST_MODEL_FILE_PATH;
    int ret = initialize(&model, params, 1);
    ASSERT_EQ(ret, 0);
    ASSERT_NE(model, nullptr);

    ret = deinitialize(model);
    ASSERT_EQ(ret, 0);

    model = nullptr;
    params[0].value = "../invalid.bin";
    ret = initialize(&model, params, 1);
    ASSERT_NE(ret, 0);
    ASSERT_EQ(model, nullptr);

    ret = deinitialize(model);
    ASSERT_EQ(ret, 0);
}

TEST(TokenizerTest, inputs_info) {
    struct CustomNodeTensorInfo* info = nullptr;
    int infoCount = 0;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = TEST_MODEL_FILE_PATH;

    BlingFireModel model(params[0].value);

    int ret = getInputsInfo(&info, &infoCount, params, 1, (void*)&model);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(infoCount, 1);
    ASSERT_EQ(std::strcmp(info[0].name, INPUT_NAME_TEXTS), 0);
    ASSERT_EQ(info[0].dimsCount, 2);
    ASSERT_EQ(info[0].dims[0], -1);
    ASSERT_EQ(info[0].dims[1], -1);
    ASSERT_EQ(info[0].precision, U8);
    ret = release(info, (void*)&model);
    ASSERT_EQ(ret, 0);
}

TEST(TokenizerTest, outputs_info) {
    struct CustomNodeTensorInfo* info = nullptr;
    int infoCount = 0;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = TEST_MODEL_FILE_PATH;

    BlingFireModel model(params[0].value);

    int ret = getOutputsInfo(&info, &infoCount, params, 1, (void*)&model);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(infoCount, 2);

    ASSERT_EQ(std::strcmp(info[0].name, OUTPUT_NAME_TOKENS), 0);
    ASSERT_EQ(info[0].dimsCount, 2);
    ASSERT_EQ(info[0].dims[0], -1);
    ASSERT_EQ(info[0].dims[1], -1);
    ASSERT_EQ(info[0].precision, I64);

    ASSERT_EQ(std::strcmp(info[1].name, OUTPUT_NAME_ATTENTION), 0);
    ASSERT_EQ(info[1].dimsCount, 2);
    ASSERT_EQ(info[1].dims[0], -1);
    ASSERT_EQ(info[1].dims[1], -1);
    ASSERT_EQ(info[1].precision, I64);

    ret = release(info, (void*)&model);
    ASSERT_EQ(ret, 0);
}

static void putStringsToTensor(std::vector<std::string> strings, struct CustomNodeTensor& tensor) {
    size_t maxStringLength = 0;
    for (auto& str : strings) {
        maxStringLength = std::max(str.size(), maxStringLength);
    }
    size_t width = maxStringLength + 1;

    tensor.dataBytes = strings.size() * width * sizeof(uint8_t);
    tensor.data = (uint8_t*)malloc(tensor.dataBytes);

    int i = 0;
    for (auto& str : strings) {
        std::memcpy(tensor.data + i * width, str.c_str(), str.size());
        tensor.data[i * width + str.size()] = 0;
        i++;
    }

    tensor.dimsCount = 2;
    tensor.dims = (uint64_t*)malloc(2 * sizeof(uint64_t));
    tensor.dims[0] = strings.size();
    tensor.dims[1] = width;

    tensor.precision = U8;
    tensor.name = INPUT_NAME_TEXTS;
}

class TokenizerFixtureTest : public ::testing::Test {
protected:
    struct output {
        std::vector<int64_t> tokens;
        std::vector<int64_t> attention;
    };
    void run(std::vector<std::string> in, std::vector<output>& out) {
        struct CustomNodeTensor inputs[1];
        struct CustomNodeTensor* outputs = nullptr;
        int outputsCount = 0;
        putStringsToTensor(in, inputs[0]);
        int ret = execute(inputs, 1, &outputs, &outputsCount, params, 3, model);
        free(inputs[0].data);
        free(inputs[0].dims);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(outputsCount, 2);
        std::vector<output> result;
        result.resize(outputs->dims[0]);
        for (int i = 0; i < outputsCount; i++) {
            if (std::strcmp(outputs[i].name, OUTPUT_NAME_ATTENTION) == 0) {
                for (int j = 0; j < outputs[i].dims[0]; j++) {
                    result[j].attention = std::vector<int64_t>(
                        (int64_t*)outputs[i].data + j * outputs[i].dims[1],
                        (int64_t*)outputs[i].data + j * outputs[i].dims[1] + outputs[i].dims[1]);
                }
            } else if (std::strcmp(outputs[i].name, OUTPUT_NAME_TOKENS) == 0) {
                for (int j = 0; j < outputs[i].dims[0]; j++) {
                    result[j].tokens = std::vector<int64_t>(
                        (int64_t*)outputs[i].data + j * outputs[i].dims[1],
                        (int64_t*)outputs[i].data + j * outputs[i].dims[1] + outputs[i].dims[1]);
                }
            } else {
                FAIL() << "Unknown output name: " << outputs[i].name;
            }
        }
        out = result;
        ASSERT_EQ(release(outputs, model), 0);
    }
    void SetUp() override {
        params[0].key = "model_path";
        params[0].value = TEST_MODEL_FILE_PATH;
        params[1].key = "max_ids_arr_length";
        params[1].value = "1024";
        params[2].key = "debug";
        params[2].value = "true";
        int ret = initialize(&model, params, 3);
        ASSERT_EQ(ret, 0);
        ASSERT_NE(model, nullptr);
    }
    void TearDown() override {
        int ret = deinitialize(model);
        ASSERT_EQ(ret, 0);
    }
    struct CustomNodeParam params[3];
    void* model = nullptr;
};

TEST_F(TokenizerFixtureTest, execute) {
    std::vector<output> outputs;
    run({"", "Hello world!", "こんにちは"}, outputs);
    ASSERT_EQ(outputs.size(), 3);

    // ""
    ASSERT_EQ(outputs[0].tokens.size(), 7);
    ASSERT_EQ(outputs[0].attention.size(), 7);
    ASSERT_EQ(std::memcmp(outputs[0].attention.data(), std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0}.data(), 7 * sizeof(int64_t)), 0);

    // "Hello world!"
    ASSERT_EQ(outputs[1].tokens.size(), 7);
    ASSERT_EQ(outputs[1].attention.size(), 7);
    ASSERT_EQ(std::memcmp(outputs[1].tokens.data(), std::vector<int64_t>{18435, 995, 0}.data(), 3 * sizeof(int64_t)), 0);
    ASSERT_EQ(std::memcmp(outputs[1].attention.data(), std::vector<int64_t>{1, 1, 1, 0, 0, 0, 0}.data(), 7 * sizeof(int64_t)), 0);

    // "こんにちは"
    ASSERT_EQ(outputs[1].tokens.size(), 7);
    ASSERT_EQ(outputs[1].attention.size(), 7);
    ASSERT_EQ(std::memcmp(outputs[2].tokens.data(), std::vector<int64_t>{23294, 241, 22174, 28618, 2515, 94, 31676}.data(), 7 * sizeof(int64_t)), 0);
    ASSERT_EQ(std::memcmp(outputs[2].attention.data(), std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1}.data(), 7 * sizeof(int64_t)), 0);
}
