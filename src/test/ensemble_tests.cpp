//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../modelconfig.hpp"
#include "../pipeline.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

class EnsembleFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Prepare manager
        setenv("NIREQ", "2", 1);
        ModelConfig config = DUMMY_MODEL_CONFIG;
        managerWithDummyModel.reloadModelWithVersions(config);

        // Prepare request
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(10);
    }

    ConstructorEnabledModelManager managerWithDummyModel;

    PredictRequest request;
    PredictResponse response;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    std::string dummyInputName = "b";
    std::string dummyOutputName = "a";
    std::string customPipelineInputName = "custom_dummy_input";
    std::string customPipelineOutputName = "custom_dummy_output";

    std::vector<float> requestData{-5.0, 3.0, 0.0, -12.0, 9.0, -100.0, 102.0, 92.0, -1.0, 12.0};
};

TEST_F(EnsembleFlowTest, DummyModel) {
    // Most basic configuration, just process single dummy model request

    // input   dummy    output
    //  O------->O------->O

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    Node* input_ptr = input.get();
    Node* model_ptr = model.get();
    Node* output_ptr = output.get();

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    // Event loop simulation
    BlobMap blobs;
    input_ptr->execute();
    blobs.clear();
    input_ptr->fetchResults(blobs);

    model_ptr->setInputs(*input_ptr, blobs);
    blobs.clear();
    model_ptr->execute();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Simulate event loop wait for inference to happen
    model_ptr->fetchResults(blobs);

    output_ptr->setInputs(*model_ptr, blobs);
    blobs.clear();
    output_ptr->execute();
    output_ptr->fetchResults(blobs);

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);

    auto responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)response.outputs().at(customPipelineOutputName).tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, 10 * sizeof(float)));
}

TEST_F(EnsembleFlowTest, SeriesOfDummyModels) {
    // Most basic configuration, just process single dummy model request

    const int N = 10;

    // input      dummy x N      output
    //  O------->O->O...O->O------->O

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto output = std::make_unique<ExitNode>(&response);

    std::unique_ptr<DLNode> dummy_nodes[N];
    for (int i = 0; i < N; i++) {
        dummy_nodes[i] = std::make_unique<DLNode>("dummy_node_" + std::to_string(i), dummyModelName, requestedModelVersion, managerWithDummyModel);
    }

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *(dummy_nodes[0]), {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*(dummy_nodes[N - 1]), *output, {{dummyOutputName, customPipelineOutputName}});
    for (int i = 0; i < N - 1; i++) {
        pipeline.connect(*(dummy_nodes[i]), *(dummy_nodes[i + 1]), {{dummyOutputName, dummyInputName}});
    }

    Node* input_ptr = input.get();
    Node* output_ptr = output.get();
    Node* dummy_node_ptr[N];
    for (int i = 0; i < N; i++) {
        dummy_node_ptr[i] = dummy_nodes[i].get();
    }

    pipeline.push(std::move(input));
    pipeline.push(std::move(output));
    for (auto& dummy_node : dummy_nodes) {
        pipeline.push(std::move(dummy_node));
    }

    // Event loop simulation
    BlobMap blobs;
    input_ptr->execute();
    blobs.clear();
    input_ptr->fetchResults(blobs);

    dummy_node_ptr[0]->setInputs(*input_ptr, blobs);
    blobs.clear();
    dummy_node_ptr[0]->execute();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Simulate event loop wait for inference to happen
    dummy_node_ptr[0]->fetchResults(blobs);

    for (int i = 1; i < N; i++) {
        dummy_node_ptr[i]->setInputs(*(dummy_node_ptr[i - 1]), blobs);
        blobs.clear();
        dummy_node_ptr[i]->execute();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Simulate event loop wait for inference to happen
        dummy_node_ptr[i]->fetchResults(blobs);
    }

    output_ptr->setInputs(*(dummy_node_ptr[N - 1]), blobs);
    blobs.clear();
    output_ptr->execute();
    output_ptr->fetchResults(blobs);

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);

    auto responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0 * N; });

    float* actual_output = (float*)response.outputs().at(customPipelineOutputName).tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, sizeof(expected_output)));
}
