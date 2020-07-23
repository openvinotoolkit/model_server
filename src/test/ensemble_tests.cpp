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
#define DEBUG
#include "../timer.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

class EnsembleFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Prepare manager
        setenv("NIREQ", "200", 1);
        // TODO above should be set to eg 2-3-4 when problem with parallel execution on the same
        // model will be resolved in model ensemble

        // Prepare request
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(10);
    }

    PredictRequest request;
    PredictResponse response;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    std::string dummyInputName = "b";
    std::string dummyOutputName = "a";
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";

    std::vector<float> requestData{-5.0, 3.0, 0.0, -12.0, 9.0, -100.0, 102.0, 92.0, -1.0, 12.0};
};

TEST_F(EnsembleFlowTest, DummyModel) {
    // Most basic configuration, just process single dummy model request

    // input   dummy    output
    //  O------->O------->O

    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    pipeline.execute();

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), 1 * 10 * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), 1);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), 10);

    auto responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, 10 * sizeof(float)));
}

TEST_F(EnsembleFlowTest, SeriesOfDummyModels) {
    // Most basic configuration, just process single dummy model request

    Timer timer;
    timer.start("prepare pipeline");

    const int N = 100;
    // input      dummy x N      output
    //  O------->O->O...O->O------->O

    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

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

    pipeline.push(std::move(input));
    pipeline.push(std::move(output));
    for (auto& dummy_node : dummy_nodes) {
        pipeline.push(std::move(dummy_node));
    }

    timer.stop("prepare pipeline");
    timer.start("pipeline::execute");
    pipeline.execute();
    timer.stop("pipeline::execute");

    timer.start("compare results");
    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), 1 * 10 * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), 1);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), 10);

    auto responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0 * N; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, sizeof(expected_output)));
    timer.stop("compare results");


    std::cout << "prepare pipeline: " << timer.elapsed<std::chrono::microseconds>("prepare pipeline") / 1000 << "ms\n";
    std::cout << "pipeline::execute: " << timer.elapsed<std::chrono::microseconds>("pipeline::execute") / 1000 << "ms\n";
    std::cout << "compare results: " << timer.elapsed<std::chrono::microseconds>("compare results") / 1000 << "ms\n";
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithDynamicBatchSize) {
    // Scenario

    // input(3x10)   dummy(1x10), change batch size    output(3x10)
    //  O-------------------------->O----------------------->O

    // input 3x10
    // dummy is 1x10, perform model batch size change to 3x10
    // process dummy
    // check if output is 3x10

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(3);
    std::vector<float> requestData = {
        -5, -4, -3, -2, -1, 1, 2, 3, 4, 5,  // batch 1
        -15, -14, -13, -12, -11, 11, 12, 13, 14, 15,  // batch 2
        -25, -24, -23, -22, -21, 21, 22, 23, 24, 25,  // batch 3
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    ModelConfig dynamicBatchConfig = DUMMY_MODEL_CONFIG;
    dynamicBatchConfig.setBatchingParams("auto");
    ConstructorEnabledModelManager managerWithDynamicBatchDummyModel;
    managerWithDynamicBatchDummyModel.reloadModelWithVersions(dynamicBatchConfig);

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    pipeline.execute();

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), 3 * 10 * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), 3);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), 10);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, 3 * 10 * sizeof(float)));
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithDynamicShape) {
    // Scenario

    // input(1x5)      dummy(1x10), reshape            output(1x5)
    //  O---------------------->O--------------------------->O

    // input 1x5
    // dummy is 1x10, perform model reshape to 1x5
    // process dummy
    // check if output is 1x5

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(5);
    std::vector<float> requestData = {
        -5, -4, -3, -2, -1,  // batch 1
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    ModelConfig dynamicShapeConfig = DUMMY_MODEL_CONFIG;
    dynamicShapeConfig.setBatchSize(0);  // = not specified in --batch_size parameter
    dynamicShapeConfig.parseShapeParameter("auto");
    ConstructorEnabledModelManager managerWithDynamicShapeDummyModel;
    managerWithDynamicShapeDummyModel.reloadModelWithVersions(dynamicShapeConfig);

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    pipeline.execute();

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), 1 * 5 * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), 1);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), 5);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, 1 * 5 * sizeof(float)));
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithDynamicBatchAndShape) {
    // Scenario

    // input(3x500)   dummy(1x10), reshape, change batch size    output(3x500)
    //  O------------------------------>O----------------------------->O

    // input 3x500
    // dummy is 1x10, perform model batch size change to 3x500
    // process dummy
    // check if output is 3x500

    const int BATCH = 3;
    const int WIDTH = 500;

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(BATCH);
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(WIDTH);
    std::vector<float> requestData;
    for (int i = 0; i < BATCH; i++) {  // batch size
        for (int j = 0; j < WIDTH; j++) {  // width
            requestData.push_back((i+1) * (j+1));
            /*
            1.0, 2.0, 3.0, ..., 500.0,
            2.0, 4.0, 6.0, ..., 1000.0,
            3.0, 6.0, 9.0, ..., 1500.0
            */
        }
    }
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(0);  // simulate --batch_size parameter not set
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    pipeline.execute();

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), BATCH * WIDTH * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), BATCH);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), WIDTH);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, BATCH * WIDTH * sizeof(float)));
}

TEST_F(EnsembleFlowTest, ParallelDummyModels) {
    // Most basic configuration, just process single dummy model request
    // TODO For now will fail if NIREQ < N * threads
    const int N = 200;
    /* input      dummy x N      output
        O---------->O------------->O
        ...        ...            /\
        L---------->O-------------_|
    */
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto output = std::make_unique<ExitNode>(&response);
    Pipeline pipeline(*input, *output);
    std::unique_ptr<DLNode> dummy_nodes[N];

    for (int i = 0; i < N; i++) {
        dummy_nodes[i] = std::make_unique<DLNode>("dummy_node_" + std::to_string(i), dummyModelName, requestedModelVersion, managerWithDummyModel);
        pipeline.connect(*input, *(dummy_nodes[i]), {{customPipelineInputName + std::to_string(i), dummyInputName}});
        pipeline.connect(*(dummy_nodes[i]), *output, {{dummyOutputName, customPipelineOutputName + std::to_string(i)}});
        pipeline.push(std::move(dummy_nodes[i]));
    }
    pipeline.push(std::move(input));
    pipeline.push(std::move(output));

    // Prepare request
    std::vector<float> requestDataT(N * DUMMY_MODEL_INPUT_SIZE);
    for (int i = 0; i < N; ++i) {
        std::transform(requestData.begin(),
            requestData.end(),
            requestDataT.begin() + DUMMY_MODEL_INPUT_SIZE * i,
            [i](int x) { return x + i; });
    }
    for (int i = 0; i < N; i++) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName + std::to_string(i)];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)(requestDataT.data() + i * DUMMY_MODEL_INPUT_SIZE),
                                               DUMMY_MODEL_INPUT_SIZE * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(10);
    }
    ASSERT_EQ(pipeline.execute(), ovms::StatusCode::OK);
    for (int i = 0; i < N; i++) {
        ASSERT_EQ(response.outputs().count(customPipelineOutputName + std::to_string(i)), 1);
    }
    auto responseData = requestDataT;
    std::transform(requestDataT.begin(), requestDataT.end(), requestDataT.begin(), [](float& v) { return v + 1.0; });

    float* expected_output = requestDataT.data();
    for (int i = 0; i < N; i++) {
        float* actual_output = (float*)response.outputs().at(customPipelineOutputName + std::to_string(i)).tensor_content().data();
        EXPECT_EQ(0, std::memcmp(actual_output, expected_output + i * DUMMY_MODEL_OUTPUT_SIZE, DUMMY_MODEL_OUTPUT_SIZE * sizeof(float)));
    }
}

TEST_F(EnsembleFlowTest, FailInDLNodeSetInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in setInputs)    output
    //  O------->O------->O
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}, {"NON_EXISTING_INPUT", "REQUIRED_IN_THEORY_OUTPUT"}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    EXPECT_EQ(pipeline.execute(), ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(EnsembleFlowTest, FailInDLNodeExecuteInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in execute)    output
    //  O------->O------->O
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName + "_NON_EXISTING_INPUT_NAME_IN_MODEL"}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    EXPECT_EQ(pipeline.execute(), ovms::StatusCode::INVALID_MISSING_INPUT);
}

class DLNodeFailInFetch : public DLNode {
public:
    DLNodeFailInFetch(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager = ModelManager::getInstance()) :
        DLNode(nodeName, modelName, modelVersion, modelManager) {}
    ovms::Status fetchResults(BlobMap&) override {
        return StatusCode::UNKNOWN_ERROR;
    }
};

TEST_F(EnsembleFlowTest, FailInDLNodeFetchResults) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in fetch)    output
    //  O------->O------->O
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNodeFailInFetch>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {{customPipelineInputName, dummyInputName}});
    pipeline.connect(*model, *output, {{dummyOutputName, customPipelineOutputName}});

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::UNKNOWN_ERROR) << status.string();
}
