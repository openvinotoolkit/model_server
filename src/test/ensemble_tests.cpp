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
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../modelconfig.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#define DEBUG
#include <cstdio>

#include <stdlib.h>

#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

const uint NIREQ = 2;

class EnsembleFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Prepare manager
        config = DUMMY_MODEL_CONFIG;
        config.setNireq(NIREQ);

        // Prepare request
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        requestData = bs1requestData;
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);
    }

    void checkResponse(int seriesLength, int batchSize = 1) {
        ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
        const auto& output_proto = response.outputs().at(customPipelineOutputName);

        ASSERT_EQ(output_proto.tensor_content().size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
        ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
        ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), batchSize);
        ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), DUMMY_MODEL_OUTPUT_SIZE);

        auto responseData = requestData;
        std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

        float* actual_output = (float*)output_proto.tensor_content().data();
        float* expected_output = responseData.data();
        const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
        EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
            << readableError(expected_output, actual_output, dataLengthToCheck);
    }

    std::string readableError(const float* expected_output, const float* actual_output, const size_t size) {
        std::stringstream ss;
        for (int i = 0; i < size; ++i) {
            if (actual_output[i] != expected_output[i]) {
                ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
                break;
            }
        }
        return ss.str();
    }

    void performWrongPipelineConfigTest(const char* configFileContent) {
        std::string fileToReload = "/tmp/ovms_config_file1.json";
        createConfigFileWithContent(configFileContent, fileToReload);
        ConstructorEnabledModelManager managerWithDummyModel;
        setenv("NIREQ", "1", 1);
        managerWithDummyModel.startFromFile(fileToReload);
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        std::unique_ptr<Pipeline> pipeline;
        auto status = managerWithDummyModel.createPipeline(pipeline,
            "pipeline1Dummy",
            &request,
            &response);
        ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NAME_MISSING) << status.string();
        managerWithDummyModel.join();
    }

    ModelConfig config;

    PredictRequest request;
    PredictResponse response;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";

    std::vector<float> requestData;
    const std::vector<float> bs1requestData{-5.0, 3.0, 0.0, -12.0, 9.0, -100.0, 102.0, 92.0, -1.0, 12.0};
};

TEST_F(EnsembleFlowTest, DummyModel) {
    // Most basic configuration, just process single dummy model request
    // input   dummy    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int dummySeriallyConnectedCount = 1;
    checkResponse(dummySeriallyConnectedCount);
}

TEST_F(EnsembleFlowTest, DummyModelDirectAndPipelineInference) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setNireq(1);
    managerWithDummyModel.reloadModelWithVersions(config);

    // Get dummy model instance
    std::shared_ptr<ovms::ModelInstance> model;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    auto status = ovms::getModelInstance(managerWithDummyModel, dummyModelName, 0, model, unload_guard);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // Prepare request for dummy model directly
    tensorflow::serving::PredictRequest simpleModelRequest = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    std::vector<float> requestData{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto& input = (*simpleModelRequest.mutable_inputs())[DUMMY_MODEL_INPUT_NAME];
    input.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    tensorflow::serving::PredictResponse simpleModelResponse;
    // Do the inference directly on dummy model before inference on pipeline
    ASSERT_EQ(inference(*model, &simpleModelRequest, &simpleModelResponse, unload_guard), ovms::StatusCode::OK);

    ASSERT_EQ(simpleModelResponse.outputs().count(DUMMY_MODEL_OUTPUT_NAME), 1);
    auto& output_tensor = (*simpleModelResponse.mutable_outputs())[DUMMY_MODEL_OUTPUT_NAME];
    ASSERT_EQ(output_tensor.tensor_shape().dim_size(), 2);
    EXPECT_EQ(output_tensor.tensor_shape().dim(0).size(), 1);
    EXPECT_EQ(output_tensor.tensor_shape().dim(1).size(), 10);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_tensor.tensor_content().data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int dummySeriallyConnectedCount = 1;
    checkResponse(dummySeriallyConnectedCount);

    // Do the inference directly on dummy model after inference on pipeline
    ASSERT_EQ(inference(*model, &simpleModelRequest, &simpleModelResponse, unload_guard), ovms::StatusCode::OK);

    ASSERT_EQ(simpleModelResponse.outputs().count(DUMMY_MODEL_OUTPUT_NAME), 1);
    output_tensor = (*simpleModelResponse.mutable_outputs())[DUMMY_MODEL_OUTPUT_NAME];
    ASSERT_EQ(output_tensor.tensor_shape().dim_size(), 2);
    EXPECT_EQ(output_tensor.tensor_shape().dim(0).size(), 1);
    EXPECT_EQ(output_tensor.tensor_shape().dim(1).size(), 10);

    actual_output = (float*)output_tensor.tensor_content().data();
    expected_output = responseData.data();
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck);
}

TEST_F(EnsembleFlowTest, SeriesOfDummyModels) {
    // Most basic configuration, just process single dummy model request

    Timer timer;
    timer.start("prepare pipeline");

    const int N = 100;
    // input      dummy x N      output
    //  O------->O->O...O->O------->O

    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);

    std::unique_ptr<DLNode> dummy_nodes[N];
    for (int i = 0; i < N; i++) {
        dummy_nodes[i] = std::make_unique<DLNode>("dummy_node_" + std::to_string(i), dummyModelName, requestedModelVersion, managerWithDummyModel);
    }

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *(dummy_nodes[0]), {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*(dummy_nodes[N - 1]), *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});
    for (int i = 0; i < N - 1; i++) {
        pipeline.connect(*(dummy_nodes[i]), *(dummy_nodes[i + 1]), {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
    }

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));
    for (auto& dummy_node : dummy_nodes) {
        pipeline.push(std::move(dummy_node));
    }

    timer.stop("prepare pipeline");
    timer.start("pipeline::execute");
    pipeline.execute();
    timer.stop("pipeline::execute");

    timer.start("compare results");
    checkResponse(N);
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
    const int batchSize = 3;
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(batchSize);
    requestData = {
        // std::vector<float> requestData = {
        -5, -4, -3, -2, -1, 1, 2, 3, 4, 5,            // batch 1
        -15, -14, -13, -12, -11, 11, 12, 13, 14, 15,  // batch 2
        -25, -24, -23, -22, -21, 21, 22, 23, 24, 25,  // batch 3
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchingParams("auto");
    ConstructorEnabledModelManager managerWithDynamicBatchDummyModel;
    managerWithDynamicBatchDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int seriallyConnectedDummyModels = 1;
    checkResponse(seriallyConnectedDummyModels, batchSize);
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

    config.setBatchSize(0);  // = not specified in --batch_size parameter
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager managerWithDynamicShapeDummyModel;
    managerWithDynamicShapeDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

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

    const int BATCH_SIZE = 3;
    const int WIDTH = 500;

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(BATCH_SIZE);
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(WIDTH);
    std::vector<float> requestData;
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < WIDTH; j++) {
            requestData.push_back((i + 1) * (j + 1));
            /*
            1.0, 2.0, 3.0, ..., 500.0,
            2.0, 4.0, 6.0, ..., 1000.0,
            3.0, 6.0, 9.0, ..., 1500.0
            */
        }
    }
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchSize(0);  // simulate --batch_size parameter not set
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), ovms::StatusCode::OK);

    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    const auto& output_proto = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto.tensor_content().size(), BATCH_SIZE * WIDTH * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), BATCH_SIZE);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), WIDTH);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();

    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, BATCH_SIZE * WIDTH * sizeof(float)));
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithDynamicShape_RequestHasDifferentDim0) {
    // Scenario
    // Shape is set to auto but only first dimension differs - change batch size via reshape

    // input(20x10)   dummy(1x10), reshape    output(20x10)
    //  O------------------------------>O----------------------------->O

    // input 20x10
    // dummy is 1x10, perform model reshape to 20x10
    // process dummy
    // check if output is 20x10

    const int BATCH_SIZE = 20;
    const int WIDTH = 10;

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(BATCH_SIZE);
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(WIDTH);
    requestData.clear();
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < WIDTH; j++) {
            requestData.push_back((i + 1) * (j + 1));
            /*
            1.0, 2.0, 3.0, ..., 10.0,
            2.0, 4.0, 6.0, ..., 20.0,
            3.0, 6.0, 9.0, ..., 30.0,
            ...
            20.0, 40.0, ..., 200.0
            */
        }
    }
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchSize(0);  // simulate --batch_size parameter not set
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), ovms::StatusCode::OK);

    const int seriallyConnectedDummyModels = 1;
    checkResponse(seriallyConnectedDummyModels, BATCH_SIZE);
}

TEST_F(EnsembleFlowTest, ParallelDummyModels) {
    // Most basic configuration, just process single dummy model request
    const int N = 200;
    /* input      dummy x N      output
        O---------->O------------->O
        ...        ...            /\
        L---------->O-------------_|
    */
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);
    Pipeline pipeline(*input_node, *output_node);
    std::unique_ptr<DLNode> dummy_nodes[N];

    for (int i = 0; i < N; i++) {
        dummy_nodes[i] = std::make_unique<DLNode>("dummy_node_" + std::to_string(i), dummyModelName, requestedModelVersion, managerWithDummyModel);
        pipeline.connect(*input_node, *(dummy_nodes[i]), {{customPipelineInputName + std::to_string(i), DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*(dummy_nodes[i]), *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + std::to_string(i)}});
        pipeline.push(std::move(dummy_nodes[i]));
    }
    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));

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
        const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);
        const float* expected_output_address_to_check = expected_output + i * DUMMY_MODEL_OUTPUT_SIZE;
        EXPECT_EQ(0, std::memcmp(actual_output, expected_output_address_to_check, dataLengthToCheck))
            << "Comparison on node:" << i << " output failed" << std::endl
            << readableError(expected_output_address_to_check, actual_output, DUMMY_MODEL_OUTPUT_SIZE);
    }
}

TEST_F(EnsembleFlowTest, FailInDLNodeSetInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in setInputs)    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}, {"NON_EXISTING_INPUT", "REQUIRED_IN_THEORY_OUTPUT"}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    EXPECT_EQ(pipeline.execute(), ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(EnsembleFlowTest, FailInDLNodeExecuteInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in execute)    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, std::string(DUMMY_MODEL_INPUT_NAME) + "_NON_EXISTING_INPUT_NAME_IN_MODEL"}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    EXPECT_EQ(pipeline.execute(), ovms::StatusCode::INVALID_MISSING_INPUT);
}

class DLNodeFailInFetch : public DLNode {
public:
    DLNodeFailInFetch(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager = ModelManager::getInstance()) :
        DLNode(nodeName, modelName, modelVersion, modelManager, {}) {}
    ovms::Status fetchResults(BlobMap&) override {
        return StatusCode::UNKNOWN_ERROR;
    }
};

TEST_F(EnsembleFlowTest, FailInDLNodeFetchResults) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in fetch)    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto model_node = std::make_unique<DLNodeFailInFetch>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::UNKNOWN_ERROR) << status.string();
}

TEST_F(EnsembleFlowTest, CorrectPipelineDefinitionValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::OK);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithModelBatchingModeAutoValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setBatchingMode(AUTO);
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithModelShapeModeAutoValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    shapes_map_t shapes;
    ShapeInfo shape;
    shape.shapeMode = AUTO;
    std::pair<std::string, ShapeInfo> shapeInfo("shape", shape);
    shapes.insert(shapeInfo);
    config.setShapes(shapes);
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithMissingNodeModelValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node1", "dummy"},
        {NodeKind::DL, "dummy_node2", "missing"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::MODEL_NAME_MISSING);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithMissingConnectionNodeValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // missingNode (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"missingNode", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::MODEL_NAME_MISSING);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithNodeOutputMissingValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{"Missing", customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionWithNodeInputMissingValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, "Missing"}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_INPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(managerWithDummyModel), StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(EnsembleFlowTest, SimplePipelineFactoryCreation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Nodes
    // entry   dummy_node    exit
    //  O--------->O---------->O
    //           dummy
    //          default
    // Models/Versions

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    // entry (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (customPipelineOutputName)
    connections["exit"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    ASSERT_EQ(factory.createDefinition("my_new_pipeline", info, connections, managerWithDummyModel), StatusCode::OK);

    std::unique_ptr<Pipeline> pipeline;

    // Create pipeline out of created definition
    ASSERT_EQ(factory.create(pipeline, "my_new_pipeline", &request, &response, managerWithDummyModel), StatusCode::OK);

    // Execute pipeline
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    const int dummySeriallyConnectedCount = 1;
    checkResponse(dummySeriallyConnectedCount);
}

TEST_F(EnsembleFlowTest, ParallelPipelineFactoryUsage) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    //                 Nodes
    //              dummy_node_N
    //         .-------->O----------v
    //  entry O--------->O---------->O exit
    //         *-------->O----------^
    //                dummy
    //               default
    //           Models/Versions

    const int PARALLEL_DUMMY_NODES = 3;
    const int PARALLEL_SIMULATED_REQUEST_COUNT = 30;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::EXIT, "exit"},
    };

    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        info.emplace_back(NodeKind::DL, "dummy_node_" + std::to_string(i), "dummy");
    }

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        // entry (customPipelineInputName) O--------->O dummy_node_N (DUMMY_MODEL_INPUT_NAME)
        connections["dummy_node_" + std::to_string(i)] = {
            {"entry", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    }

    // dummy_node_0 (DUMMY_MODEL_OUTPUT_NAME) O---------v
    // dummy_node_1 (DUMMY_MODEL_OUTPUT_NAME) O--------->O exit (output_0, output_1, output_N)
    // dummy_node_N (DUMMY_MODEL_OUTPUT_NAME) O---------^
    auto& exitConnections = connections["exit"];
    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        exitConnections["dummy_node_" + std::to_string(i)] = {{DUMMY_MODEL_OUTPUT_NAME, "output_" + std::to_string(i)}};
    }

    // Create pipeline definition
    ASSERT_EQ(factory.createDefinition("my_new_pipeline", info, connections, managerWithDummyModel), StatusCode::OK);

    auto run = [&]() {
        std::unique_ptr<Pipeline> pipeline;
        PredictResponse response_local;

        // Create pipeline out of created definition
        ASSERT_EQ(factory.create(pipeline, "my_new_pipeline", &request, &response_local, managerWithDummyModel), StatusCode::OK);

        // Execute pipeline
        ASSERT_EQ(pipeline->execute(), StatusCode::OK);

        // Validate response
        ASSERT_EQ(response_local.outputs_size(), PARALLEL_DUMMY_NODES);

        auto responseData = requestData;
        std::for_each(responseData.begin(), responseData.end(), [](float& v) { v += 1.0; });

        size_t expectedContentSize = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);

        for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
            std::string outputName = "output_" + std::to_string(i);
            ASSERT_EQ(response_local.outputs().count(outputName), 1);
            const auto& tensor = response_local.outputs().at(outputName);
            ASSERT_EQ(tensor.tensor_content().size(), expectedContentSize);
            float* actual_output = (float*)tensor.tensor_content().data();
            float* expected_output = responseData.data();

            EXPECT_EQ(0, std::memcmp(actual_output, expected_output, expectedContentSize));
        }
    };

    std::vector<std::promise<void>> promises(PARALLEL_SIMULATED_REQUEST_COUNT);
    std::vector<std::thread> threads;

    for (int n = 0; n < PARALLEL_SIMULATED_REQUEST_COUNT; n++) {
        threads.emplace_back(std::thread([&promises, n, &run]() {
            promises[n].get_future().get();
            run();
        }));
    }

    // Sleep to allow all threads to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (auto& promise : promises) {
        promise.set_value();
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_MultipleEntryNodes) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry1"},
        {NodeKind::ENTRY, "entry2"},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::OK);
    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_MultipleExitNodes) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::EXIT, "exit1"},
        {NodeKind::EXIT, "exit2"},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::OK);
    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_MULTIPLE_EXIT_NODES);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_ExitMissing) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::OK);
    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_EntryMissing) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::EXIT, "exit"},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::OK);
    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_DefinitionMissing) {
    PipelineFactory factory;

    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_DEFINITION_NAME_MISSING);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_NodeNameDuplicate) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "entry"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "exit"},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::OK);
    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, ovms::ModelManager::getInstance()), StatusCode::PIPELINE_NODE_NAME_DUPLICATE);
}

const char* pipelineOneDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithInputOutputsMappings) {
    std::string fileToReload = "/tmp/ovms_config_file1.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager managerWithDummyModel;
    setenv("NIREQ", "1", 1);
    managerWithDummyModel.startFromFile(fileToReload);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    std::unique_ptr<Pipeline> pipeline;
    auto status = managerWithDummyModel.createPipeline(pipeline,
        "pipeline1Dummy",
        &request,
        &response);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    const int dummySeriallyConnectedCount = 1;
    checkResponse(dummySeriallyConnectedCount);
    managerWithDummyModel.join();
}

const char* pipelineOneDummyConfig2ParallelDummy = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                },
                {
                    "name": "dummyNode2",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output2"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                },
                {"custom_dummy_output2": {"SourceNodeName": "dummyNode2",
                                         "SourceNodeOutputName": "new_dummy_output2"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithInputOutputsMappings2ParallelDummy) {
    std::string fileToReload = "/tmp/ovms_config_file1.json";
    createConfigFileWithContent(pipelineOneDummyConfig2ParallelDummy, fileToReload);
    ConstructorEnabledModelManager managerWithDummyModel;
    setenv("NIREQ", "2", 1);
    managerWithDummyModel.startFromFile(fileToReload);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    std::unique_ptr<Pipeline> pipeline;
    auto status = managerWithDummyModel.createPipeline(pipeline,
        "pipeline1Dummy",
        &request,
        &response);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().count(customPipelineOutputName), 1);
    ASSERT_EQ(response.outputs().count(std::string(customPipelineOutputName) + "2"), 1);
    // check 1st output
    const auto& output_proto = response.outputs().at(customPipelineOutputName);
    const int batchSize = 1;
    const int seriesLength = 1;
    ASSERT_EQ(output_proto.tensor_content().size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), batchSize);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), DUMMY_MODEL_OUTPUT_SIZE);

    auto responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck);

    // check 2nd output
    const auto& output_proto2 = response.outputs().at(customPipelineOutputName);

    ASSERT_EQ(output_proto2.tensor_content().size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto2.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto2.tensor_shape().dim(0).size(), batchSize);
    ASSERT_EQ(output_proto2.tensor_shape().dim(1).size(), DUMMY_MODEL_OUTPUT_SIZE);

    actual_output = (float*)output_proto2.tensor_content().data();
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck);
    managerWithDummyModel.join();
}

const char* pipelineOneDummyConfigWrongNodeKind = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL modeloze Wrong kind",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithWrongNodeKind) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWrongNodeKind);
}

const char* pipelineOneDummyConfigMissingNodeModelName = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeModelName) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeModelName);
}

const char* pipelineOneDummyConfigMissingNodeName = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeName) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeName);
}

const char* pipelineOneDummyConfigMissingNodeInputs = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeInputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeInputs);
}

const char* pipelineOneDummyConfigWithMissingNodeOutputs = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeOutputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWithMissingNodeOutputs);
}

const char* pipelineOneDummyConfigWithMissingPipelineOutputs = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingPipelineOutputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWithMissingPipelineOutputs);
}

const char* pipelineOneDummyConfigWithMissingPipelineInputs = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"SourceNodeName": "entry",
                               "SourceNodeOutputName": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"ModelOutputName": "a",
                         "OutputName": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"SourceNodeName": "dummyNode",
                                         "SourceNodeOutputName": "new_dummy_output"}
                }
            ]
        }
    ]
})";

// There is no need at the moment to have inputs declared in pipeline inputs since those are taken anyway from input/output
// mapping which is defined in following nodes (nodes that use anything directly from PredictRequest)
TEST_F(EnsembleFlowTest, DISABLED_PipelineFactoryCreationWithMissingPipelineInputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWithMissingPipelineInputs);
}

TEST_F(EnsembleFlowTest, ErrorHandlingSkipsDeferredNodesExecutionIfExecutionFailed) {
    // This test creates specific scenario where 3 parallel nodes are getting executed
    // with nireq=1. The second node gets stream id ticket for inference and is deferred
    // for execution later. Meanwhile error occurs in third parallel node (shape validation error).

    // Expected result - have pipeline cancelled with proper error code

    // Manger with dummy model and nireq=1
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setNireq(1);
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);

    auto dummy_node_1 = std::make_unique<DLNode>("dummy_node_1", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto dummy_node_2 = std::make_unique<DLNode>("dummy_node_2", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto dummy_node_3 = std::make_unique<DLNode>("dummy_node_3", dummyModelName, requestedModelVersion, managerWithDummyModel);

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *dummy_node_1, {{"proto_input_1x10", DUMMY_MODEL_INPUT_NAME}});  // this node will start execution, reserve stream id
    pipeline.connect(*input_node, *dummy_node_2, {{"proto_input_1x10", DUMMY_MODEL_INPUT_NAME}});  // this node will start execution, get future object for stream id, defer to queue
    pipeline.connect(*input_node, *dummy_node_3, {{"proto_input_1x5", DUMMY_MODEL_INPUT_NAME}});   // this node will fail at validation time
    pipeline.connect(*dummy_node_1, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, "proto_output_1x10_A"}});
    pipeline.connect(*dummy_node_2, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, "proto_output_1x10_B"}});
    pipeline.connect(*dummy_node_3, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, "proto_output_1x5"}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));
    pipeline.push(std::move(dummy_node_1));
    pipeline.push(std::move(dummy_node_2));
    pipeline.push(std::move(dummy_node_3));

    request.Clear();

    auto& proto_input_1x5 = (*request.mutable_inputs())["proto_input_1x5"];
    auto& proto_input_1x10 = (*request.mutable_inputs())["proto_input_1x10"];

    proto_input_1x5.set_dtype(tensorflow::DataType::DT_FLOAT);
    proto_input_1x10.set_dtype(tensorflow::DataType::DT_FLOAT);

    std::vector<float> data_1x5(5);
    std::vector<float> data_1x10(10);
    std::iota(data_1x5.begin(), data_1x5.end(), 0);    // 0, 1, 2, 3, 4
    std::iota(data_1x10.begin(), data_1x10.end(), 5);  // 5, 6, ..., 14

    proto_input_1x5.mutable_tensor_content()->assign((char*)data_1x5.data(), data_1x5.size() * sizeof(float));
    proto_input_1x5.mutable_tensor_shape()->add_dim()->set_size(1);
    proto_input_1x5.mutable_tensor_shape()->add_dim()->set_size(data_1x5.size());

    proto_input_1x10.mutable_tensor_content()->assign((char*)data_1x10.data(), data_1x10.size() * sizeof(float));
    proto_input_1x10.mutable_tensor_shape()->add_dim()->set_size(1);
    proto_input_1x10.mutable_tensor_shape()->add_dim()->set_size(data_1x10.size());

    EXPECT_EQ(pipeline.execute(), StatusCode::INVALID_SHAPE);
}
