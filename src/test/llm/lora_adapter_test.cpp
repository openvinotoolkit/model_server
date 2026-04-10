//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/genai/lora_adapter.hpp>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../llm/servable.hpp"
#include "../../llm/servable_initializer.hpp"
#include "../../status.hpp"
#include "../test_with_temp_dir.hpp"

using namespace ovms;

class LoraAdapterInitTest : public TestWithTempDir {
protected:
    std::shared_ptr<GenAiServableProperties> properties;
    mediapipe::LLMCalculatorOptions nodeOptions;
    std::string loraDir;
    std::string loraFilePath;

    // Creates a minimal valid safetensors file that ov::genai::Adapter can load
    static void createMinimalSafetensorsFile(const std::string& dir) {
        std::filesystem::create_directories(dir);
        std::string path = dir + "/adapter_model.safetensors";

        std::string header =
            R"({"lora_A.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},)"
            R"("lora_B.weight":{"dtype":"F32","shape":[2,1],"data_offsets":[8,16]}})";
        // Pad header to 8-byte alignment
        while (header.size() % 8 != 0)
            header += ' ';

        uint64_t headerLen = header.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&headerLen), sizeof(headerLen));
        f.write(header.data(), headerLen);
        // 16 bytes of zero tensor data (4 floats of zeros)
        const std::vector<char> zeros(16, 0);
        f.write(zeros.data(), zeros.size());
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        properties = std::make_shared<GenAiServableProperties>();
        loraDir = directoryPath + "/lora_adapter";
        createMinimalSafetensorsFile(loraDir);
        loraFilePath = loraDir + "/adapter_model.safetensors";
    }
};

// --- Protobuf parsing tests ---

TEST_F(LoraAdapterInitTest, ProtobufLoraAdapterFieldsParsedCorrectly) {
    std::string pbtxt = R"(
        models_path: "/some/model"
        lora_adapter { model_path: "/path/to/lora1" alpha: 0.5 }
        lora_adapter { model_path: "/path/to/lora2" }
    )";
    mediapipe::LLMCalculatorOptions opts;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(pbtxt, &opts));
    ASSERT_EQ(opts.lora_adapter_size(), 2);
    EXPECT_EQ(opts.lora_adapter(0).model_path(), "/path/to/lora1");
    EXPECT_FLOAT_EQ(opts.lora_adapter(0).alpha(), 0.5f);
    EXPECT_EQ(opts.lora_adapter(1).model_path(), "/path/to/lora2");
    EXPECT_FLOAT_EQ(opts.lora_adapter(1).alpha(), 1.0f);  // default
}

// --- No adapters ---

TEST_F(LoraAdapterInitTest, NoAdaptersReturnsOk) {
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "/some/path", properties), StatusCode::OK);
    EXPECT_TRUE(properties->pluginConfig.empty());
}

// --- Invalid path ---

TEST_F(LoraAdapterInitTest, NonExistentPathFails) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(directoryPath + "/nonexistent_lora");
    adapter->set_alpha(0.5f);
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}

// --- Alpha validation ---

TEST_F(LoraAdapterInitTest, AlphaZeroFails) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    adapter->set_alpha(0.0f);
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}

TEST_F(LoraAdapterInitTest, AlphaNegativeFails) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    adapter->set_alpha(-0.5f);
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}

TEST_F(LoraAdapterInitTest, AlphaAboveOneFails) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    adapter->set_alpha(1.5f);
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}

// --- Happy paths ---

TEST_F(LoraAdapterInitTest, ValidAdapterWithAlpha) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    adapter->set_alpha(0.5f);
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "", properties), StatusCode::OK);
    EXPECT_FALSE(properties->pluginConfig.empty());
    EXPECT_EQ(properties->pluginConfig.count("adapters"), 1);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 1);
}

TEST_F(LoraAdapterInitTest, DefaultAlphaSucceeds) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    // alpha defaults to 1.0 in proto
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "", properties), StatusCode::OK);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 1);
}

TEST_F(LoraAdapterInitTest, AlphaExactlyOneSucceeds) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);
    adapter->set_alpha(1.0f);
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "", properties), StatusCode::OK);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 1);
}

TEST_F(LoraAdapterInitTest, MultipleAdaptersRegistered) {
    auto* a1 = nodeOptions.add_lora_adapter();
    a1->set_model_path(loraFilePath);
    a1->set_alpha(0.3f);
    auto* a2 = nodeOptions.add_lora_adapter();
    a2->set_model_path(loraFilePath);
    a2->set_alpha(0.7f);
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "", properties), StatusCode::OK);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 2);
}

// --- Path resolution ---

TEST_F(LoraAdapterInitTest, RelativePathResolvedAgainstGraphPath) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path("lora_adapter/adapter_model.safetensors");
    adapter->set_alpha(0.5f);
    // graphPath = directoryPath, so relative "lora_adapter" resolves to directoryPath/lora_adapter
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, directoryPath, properties), StatusCode::OK);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 1);
}

TEST_F(LoraAdapterInitTest, AbsolutePathIgnoresGraphPath) {
    auto* adapter = nodeOptions.add_lora_adapter();
    adapter->set_model_path(loraFilePath);  // absolute path
    adapter->set_alpha(0.5f);
    ASSERT_EQ(initializeLoraAdapters(nodeOptions, "/wrong/graph/path", properties), StatusCode::OK);
    EXPECT_EQ(properties->adapterConfig.get_adapters().size(), 1);
}

// --- Mixed valid/invalid ---

TEST_F(LoraAdapterInitTest, SecondAdapterInvalidAlphaFailsAll) {
    auto* a1 = nodeOptions.add_lora_adapter();
    a1->set_model_path(loraFilePath);
    a1->set_alpha(0.5f);
    auto* a2 = nodeOptions.add_lora_adapter();
    a2->set_model_path(loraFilePath);
    a2->set_alpha(0.0f);  // invalid
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}

TEST_F(LoraAdapterInitTest, SecondAdapterInvalidPathFailsAll) {
    auto* a1 = nodeOptions.add_lora_adapter();
    a1->set_model_path(loraFilePath);
    a1->set_alpha(0.5f);
    auto* a2 = nodeOptions.add_lora_adapter();
    a2->set_model_path(directoryPath + "/no_such_adapter.safetensors");
    a2->set_alpha(0.5f);
    EXPECT_EQ(initializeLoraAdapters(nodeOptions, "", properties),
        StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}
