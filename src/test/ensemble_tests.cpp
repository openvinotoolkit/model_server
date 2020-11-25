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

using testing::_;
using testing::Return;

const uint NIREQ = 2;

class EnsembleFlowTest : public TestWithTempDir {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
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
        for (size_t i = 0; i < size; ++i) {
            if (actual_output[i] != expected_output[i]) {
                ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
                break;
            }
        }
        return ss.str();
    }

    void performWrongPipelineConfigTest(const char* configFileContent) {
        std::string fileToReload = directoryPath + "/ovms_config_file1.json";
        createConfigFileWithContent(configFileContent, fileToReload);
        ConstructorEnabledModelManager managerWithDummyModel;
        managerWithDummyModel.startFromFile(fileToReload);
        waitForOVMSConfigReload(managerWithDummyModel);
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

TEST_F(EnsembleFlowTest, CorrectPipelineDefinitionNodesValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::OK);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithModelBatchingModeAutoValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setBatchingMode(AUTO);
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithModelShapeModeAutoValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.parseShapeParameter("auto");
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithMissingNodeModelValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node1", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node2", "missing", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node 1 (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node1"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // request (customPipelineInputName) O--------->O dummy node 2 (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node2"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node1", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_1"}}},
        {"dummy_node2", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_2"}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithMissingConnectionNodeValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // missingNode (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {"missingNode", {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_NODE);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithNodeOutputMissingValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{"MISSING", customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodesWithNodeModelInputMissingValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    //                                           /\--------|
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_OUTPUT_NAME}}},
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "MISSING"}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_CONNECTION_TO_MISSING_MODEL_INPUT);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionNodeNotAllInputsConnectedValidation) {
    ConstructorEnabledModelManager manager;
    ModelConfig sumModelConfig = SUM_MODEL_CONFIG;
    manager.reloadModelWithVersions(sumModelConfig);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "sum_node", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // Missing connection for SUM_MODEL_INPUT_NAME_2
    connections["sum_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, SUM_MODEL_INPUT_NAME_1}}}};

    connections[EXIT_NODE_NAME] = {
        {"sum_node", {{SUM_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(manager), StatusCode::PIPELINE_NOT_ALL_INPUTS_CONNECTED);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionShapesNotMatchBetweenDLModelTensorsValidation) {
    ConstructorEnabledModelManager manager;
    ModelConfig dummy_1x10 = config;
    ModelConfig dummy_1x20 = config;
    dummy_1x10.setName("dummy_1x10");
    dummy_1x20.setName("dummy_1x20");
    dummy_1x10.setBatchSize(0);
    dummy_1x20.setBatchSize(0);
    ASSERT_EQ(dummy_1x10.parseShapeParameter("(1,10)"), StatusCode::OK);
    ASSERT_EQ(dummy_1x20.parseShapeParameter("(1,20)"), StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_1x10), StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_1x20), StatusCode::OK);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node_1x10", "dummy_1x10", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_1x20", "dummy_1x20", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_1x10"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_1x20"] = {
        {"dummy_node_1x10", {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_1x20", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(manager), StatusCode::INVALID_SHAPE);
}

// Disabled until CVS-36446 is done.
TEST_F(EnsembleFlowTest, DISABLED_PipelineDefinitionPrecisionsNotMatchBetweenDLModelTensorsValidation) {
    ConstructorEnabledModelManager manager;
    ModelConfig dummy_fp32 = config;
    ModelConfig dummy_u8 = config;
    dummy_fp32.setName("dummy_fp32");
    dummy_u8.setName("dummy_u8");
    // Set precision of dummy_FP32 to FP32
    // Set precision of dummy_U8 to U8
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_fp32), StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_u8), StatusCode::OK);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node_fp32", "dummy_fp32", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_u8", "dummy_u8", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_fp32"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_u8"] = {
        {"dummy_node_fp32", {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_u8", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(manager), StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionMultipleConnectionsToModelInputValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME},
                              {customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_MODEL_INPUT_CONNECTED_TO_MULTIPLE_DATA_SOURCES);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionExitNodeIsDependencyErrorValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {EXIT_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

    // Create pipeline definition
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validateNodes(managerWithDummyModel), StatusCode::PIPELINE_EXIT_USED_AS_NODE_DEPENDENCY);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionComplexGraphWithNoCycleValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::DL, "dummy_node1", "output"},
        {NodeKind::DL, "dummy_node2", "output"},
        {NodeKind::DL, "dummy_node3", "output"},
        {NodeKind::DL, "dummy_node4", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node O--------->O dummy node 1
    connections["dummy_node1"] = {
        {"dummy_node", {{"output", "input"}}}};

    // dummy node 1 O--------->O dummy node 2
    connections["dummy_node2"] = {
        {"dummy_node1", {{"output", "input"}}}};

    // dummy node 2 O-------->\/
    // dummy node 4 O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node2", {{"output", "input"}}},
        {"dummy_node4", {{"output", "input"}}}};

    // request O--------->O dummy node 3
    connections["dummy_node3"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node 3 O-------->\/
    // dummy node 2 O--------->O dummy node 4
    connections["dummy_node4"] = {
        {"dummy_node3", {{"output", "input"}}},
        {"dummy_node2", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::OK);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionComplexGrapgWithCycleValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::DL, "dummy_node1", "output"},
        {NodeKind::DL, "dummy_node2", "output"},
        {NodeKind::DL, "dummy_node3", "output"},
        {NodeKind::DL, "dummy_node4", "output"},
        {NodeKind::DL, "dummy_node5", "output"},
        {NodeKind::DL, "dummy_node6", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node O--------->O dummy node 1
    connections["dummy_node1"] = {
        {"dummy_node", {{"output", "input"}}}};

    // dummy node 1 O--------->O dummy node 2
    connections["dummy_node2"] = {
        {"dummy_node1", {{"output", "input"}}}};

    // dummy node 2 O-------->\/
    // dummy node 6 O--------->O dummy node 3
    connections["dummy_node3"] = {
        {"dummy_node2", {{"output", "input"}}},
        {"dummy_node6", {{"output", "input"}}}};

    // dummy node 3 O-------->\/
    // dummy node 6 O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node3", {{"output", "input"}}},
        {"dummy_node6", {{"output", "input"}}}};

    // request O--------->O dummy node 4
    connections["dummy_node4"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node 3 O-------->\/
    // dummy node 4 O--------->O dummy node 5
    connections["dummy_node5"] = {
        {"dummy_node4", {{"output", "input"}}},
        {"dummy_node3", {{"output", "input"}}}};

    // dummy node 5 O--------->O dummy node 6
    connections["dummy_node6"] = {
        {"dummy_node5", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::PIPELINE_CYCLE_FOUND);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionContainingCycleValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::DL, "dummy_node1", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // response O--------->O dummy node
    connections["dummy_node"] = {
        {EXIT_NODE_NAME, {{"output", "input"}}}};

    // dummy node 1 O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node1", {{"output", "input"}}}};

    // dummy node O--------->O dummy node 1
    connections["dummy_node1"] = {
        {"dummy_node", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::PIPELINE_CYCLE_FOUND);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionContainingNodeConnectedToItselfValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node ----|
    //                            /\-----|
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}},
        {"dummy_node", {{"output", "input"}}}};

    // dummy node 1 O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::PIPELINE_CYCLE_FOUND);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionContainingTwoCyclesValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::DL, "dummy_node1", "output"},
        {NodeKind::DL, "dummy_node2", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node O--------->O dummy node
    connections["dummy_node"] = {
        {EXIT_NODE_NAME, {{"output", "input"}}}};

    // dummy node 1 O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node1", {{"output", "input"}}}};

    // dummy node   O---------------\/
    // dummy node 2 O--------->dummy node 1
    connections["dummy_node1"] = {
        {"dummy_node", {{"output", "input"}}},
        {"dummy_node2", {{"output", "input"}}}};

    // dummy node 1 O--------->O dummy node 2
    connections["dummy_node2"] = {
        {"dummy_node1", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::PIPELINE_CYCLE_FOUND);
}

TEST_F(EnsembleFlowTest, PipelineDefinitionContainingUnconnectedNodeValidation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "output"},
        {NodeKind::DL, "dummy_node1", "output"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request O--------->O dummy node
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"output", "input"}}}};

    // dummy node O--------->O response
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{"output", "input"}}}};

    // Create pipeline definition
    PipelineDefinition pipelineDefinition("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition.validateForCycles(), StatusCode::PIPELINE_CONTAINS_UNCONNECTED_NODES);
}

TEST_F(EnsembleFlowTest, SimplePipelineFactoryCreation) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    // Nodes
    // request   dummy_node    response
    //  O--------->O---------->O
    //           dummy
    //          default
    // Models/Versions

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (customPipelineInputName) O--------->O dummy node (DUMMY_MODEL_INPUT_NAME)
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    // dummy node (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (customPipelineOutputName)
    connections[EXIT_NODE_NAME] = {
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
    //  request O--------->O---------->O response
    //         *-------->O----------^
    //                dummy
    //               default
    //           Models/Versions

    const int PARALLEL_DUMMY_NODES = 3;
    const int PARALLEL_SIMULATED_REQUEST_COUNT = 30;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        info.emplace_back(std::move(NodeInfo(
            NodeKind::DL,
            "dummy_node_" + std::to_string(i),
            "dummy",
            std::nullopt,
            {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}})));
    }

    pipeline_connections_t connections;

    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        // request (customPipelineInputName) O--------->O dummy_node_N (DUMMY_MODEL_INPUT_NAME)
        connections["dummy_node_" + std::to_string(i)] = {
            {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    }

    // dummy_node_0 (DUMMY_MODEL_OUTPUT_NAME) O---------v
    // dummy_node_1 (DUMMY_MODEL_OUTPUT_NAME) O--------->O response (output_0, output_1, output_N)
    // dummy_node_N (DUMMY_MODEL_OUTPUT_NAME) O---------^
    auto& responseConnections = connections[EXIT_NODE_NAME];
    for (int i = 0; i < PARALLEL_DUMMY_NODES; i++) {
        responseConnections["dummy_node_" + std::to_string(i)] = {{DUMMY_MODEL_OUTPUT_NAME, "output_" + std::to_string(i)}};
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
        {NodeKind::ENTRY, "request1"},
        {NodeKind::ENTRY, "request2"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_MultipleExitNodes) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::EXIT, "response1"},
        {NodeKind::EXIT, "response2"},
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::PIPELINE_MULTIPLE_EXIT_NODES);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_ExitMissing) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
    };

    EXPECT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT);
}

TEST_F(EnsembleFlowTest, PipelineFactoryWrongConfiguration_EntryMissing) {
    // Prepare manager
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    EXPECT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT);
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
        {NodeKind::ENTRY, ENTRY_NODE_NAME},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    ASSERT_EQ(factory.createDefinition("pipeline", info, {}, managerWithDummyModel), StatusCode::PIPELINE_NODE_NAME_DUPLICATE);
}

const std::string PIPELINE_1_DUMMY_NAME = "pipeline1Dummy";

static const char* pipelineOneDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

static const char* pipelineOneDynamicParamDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1,
                "shape": "auto"
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithInputOutputsMappings) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.startFromFile(fileToReload);
    waitForOVMSConfigReload(managerWithDummyModel);
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

static const char* pipelineOneDummyConfig2ParallelDummy = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 2
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                },
                {
                    "name": "dummyNode2",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output2"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                },
                {"custom_dummy_output2": {"node_name": "dummyNode2",
                                         "data_item": "new_dummy_output2"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithInputOutputsMappings2ParallelDummy) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig2ParallelDummy, fileToReload);
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.startFromFile(fileToReload);
    waitForOVMSConfigReload(managerWithDummyModel);
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

static const char* pipelineOneDummyConfigWrongNodeKind = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithWrongNodeKind) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWrongNodeKind);
}

static const char* pipelineOneDummyConfigMissingNodeModelName = R"(
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeModelName) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeModelName);
}

static const char* pipelineOneDummyConfigMissingNodeName = R"(
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeName) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeName);
}

static const char* pipelineOneDummyConfigMissingNodeInputs = R"(
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
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeInputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigMissingNodeInputs);
}

static const char* pipelineOneDummyConfigWithMissingNodeOutputs = R"(
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingNodeOutputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWithMissingNodeOutputs);
}

static const char* pipelineOneDummyConfigWithMissingPipelineOutputs = R"(
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingPipelineOutputs) {
    performWrongPipelineConfigTest(pipelineOneDummyConfigWithMissingPipelineOutputs);
}

static const char* pipelineOneDummyConfigWithMissingPipelineInputs = R"(
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithMissingPipelineInputs) {
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

TEST_F(EnsembleFlowTest, ReloadPipelineDefinitionWithNewModelNameShouldPass) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok());

    config.setName("newDummy");
    status = managerWithDummyModel.reloadModelWithVersions(config);
    ASSERT_TRUE(status.ok()) << status.string();
    std::vector<NodeInfo> infoNew{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "newDummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    status = pd.reload(managerWithDummyModel, std::move(infoNew), std::move(connections));
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(EnsembleFlowTest, ReloadPipelineDefinitionWithNewNonExistingModelNameShouldFail) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok());

    ASSERT_TRUE(status.ok()) << status.string();
    std::vector<NodeInfo> infoNew{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "newDummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    status = pd.reload(managerWithDummyModel, std::move(infoNew), std::move(connections));
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL) << status.string();
}

TEST_F(EnsembleFlowTest, ReloadPipelineDefinitionWithAllModelVersionsRetiredShouldFail) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok()) << status.string();
    managerWithDummyModel.findModelByName("dummy")->retireAllVersions();

    status = pd.reload(managerWithDummyModel, std::move(info), std::move(connections));
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL) << status.string();
}

TEST_F(EnsembleFlowTest, RevalidatePipelineDefinitionWhen1ModelVersionBecomesAvailableShouldPass) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok()) << status.string();
    managerWithDummyModel.findModelByName("dummy")->retireAllVersions();

    status = pd.validate(managerWithDummyModel);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL) << status.string();

    status = managerWithDummyModel.reloadModelWithVersions(config);
    ASSERT_TRUE(status.ok()) << status.string();
    status = pd.validate(managerWithDummyModel);
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(EnsembleFlowTest, DISABLED_RetirePipelineDefinitionExecuteShouldFail) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok());
    pd.retire(managerWithDummyModel);
    std::unique_ptr<Pipeline> pipeline;
    status = pd.create(pipeline, &request, &response, managerWithDummyModel);
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE);
}

TEST_F(EnsembleFlowTest, ExecuteOnPipelineCreatedBeforeRetireShouldPass) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    pipeline_connections_t connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    PipelineDefinition pd(pipelineName, info, connections);
    auto status = pd.validate(managerWithDummyModel);
    ASSERT_TRUE(status.ok());
    std::unique_ptr<Pipeline> pipelineBeforeRetire;
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_TRUE(status.ok());
    pd.retire(managerWithDummyModel);
    pipelineBeforeRetire->execute();
    uint dummySeriallyConnectedCount = 1;
    checkResponse(dummySeriallyConnectedCount);
}

class MockedPipelineDefinitionWithHandlingStatus : public PipelineDefinition {
public:
    MockedPipelineDefinitionWithHandlingStatus(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections) :
        PipelineDefinition(pipelineName, nodeInfos, connections) {}
    PipelineDefinitionStatus& getControlableStatus() {
        return status;
    }
};

TEST_F(EnsembleFlowTest, WaitForLoadingPipelineDefinitionFromBeginStatus) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    const std::string pipelineName = "originalName";
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };
    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    MockedPipelineDefinitionWithHandlingStatus pd(pipelineName, info, connections);
    std::unique_ptr<Pipeline> pipelineBeforeRetire;
    std::thread t([&managerWithDummyModel, &pd]() {
        std::this_thread::sleep_for(std::chrono::microseconds(PipelineDefinition::WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS / 4));
        auto status = pd.validate(managerWithDummyModel);
        ASSERT_TRUE(status.ok());
        SPDLOG_ERROR("Made pd validated");
    });
    auto status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_TRUE(status.ok());
    pd.getControlableStatus().handle(ValidationFailedEvent());
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
    pd.getControlableStatus().handle(UsedModelChangedEvent());
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
    std::thread t2([&managerWithDummyModel, &pd]() {
        std::this_thread::sleep_for(std::chrono::microseconds(PipelineDefinition::WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS / 4));
        auto status = pd.validate(managerWithDummyModel);
        ASSERT_TRUE(status.ok());
        SPDLOG_ERROR("Made pd validated");
    });
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_TRUE(status.ok());
    uint dummySeriallyConnectedCount = 1;
    pipelineBeforeRetire->execute();
    checkResponse(dummySeriallyConnectedCount);
    t.join();
    t2.join();
}

static const char* configJsonWithNoPipeline = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ]
})";

TEST_F(EnsembleFlowTest, RetireAllPipelinesAfterLoading) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(configJsonWithNoPipeline, fileToReload);
    waitForOVMSConfigReload(manager);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);
}
static const char* pipelineOneDummyConfigWithChangedInputName = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["NEW_INPUT_NAME"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "NEW_INPUT_NAME"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";
const std::string NEW_INPUT_NAME = "NEW_INPUT_NAME";

TEST_F(EnsembleFlowTest, ReloadPipelineAfterLoadingSuccesfullyChangedInputName) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    tensor_map_t inputsInfoBefore;
    auto pdPtr = manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME);
    status = pdPtr->getInputsInfo(inputsInfoBefore, manager);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(inputsInfoBefore.count(NEW_INPUT_NAME), 0);

    // now reload
    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(pipelineOneDummyConfigWithChangedInputName, fileToReload);
    waitForOVMSConfigReload(manager);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    tensor_map_t inputsInfoAfter;
    status = pdPtr->getInputsInfo(inputsInfoAfter, manager);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(inputsInfoAfter.count(NEW_INPUT_NAME), 1);
}
static const char* pipelineOneDummyConfigWithMissingModel = R"(
{
    "model_config_list": [
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
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";
TEST_F(EnsembleFlowTest, ReloadPipelineAfterLoadingFailDueToMissingModel) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(pipelineOneDummyConfigWithMissingModel, fileToReload);
    waitForOVMSConfigReload(manager);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}
static const char* pipelineTwoDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipelineToRetire",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        },
        {
            "name": "pipelineToReload",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";
static const char* pipelineTwoDummyConfigAfterChanges = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipelineToAdd",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        },
        {
            "name": "pipelineToReload",
            "inputs": ["NEW_INPUT_NAME"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "NEW_INPUT_NAME"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

const std::string PIPELINE_TO_RETIRE{"pipelineToRetire"};
const std::string PIPELINE_TO_RELOAD{"pipelineToReload"};
const std::string PIPELINE_TO_ADD{"pipelineToAdd"};

TEST_F(EnsembleFlowTest, RetireReloadAddPipelineAtTheSameTime) {
    // First add 2 pipelines with different names
    // Then change config in a way:
    //  * remove 1 pipeline
    //  * change connection name between 2 nodes
    //  * add new pipeline (just with different name)
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineTwoDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RETIRE)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_ADD), nullptr);

    tensor_map_t inputsInfoBefore;
    auto pipelineToReloadPtr = manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD);
    status = pipelineToReloadPtr->getInputsInfo(inputsInfoBefore, manager);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(inputsInfoBefore.count(NEW_INPUT_NAME), 0);

    // now reload
    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(pipelineTwoDummyConfigAfterChanges, fileToReload);
    waitForOVMSConfigReload(manager);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RETIRE)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_ADD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    tensor_map_t inputsInfoAfter;
    status = pipelineToReloadPtr->getInputsInfo(inputsInfoAfter, manager);
    ASSERT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(inputsInfoAfter.count(NEW_INPUT_NAME), 1);
}

TEST_F(EnsembleFlowTest, EnablingDynamicParametersForModelUsedInPipeline) {
    /*
        This test modifies config.json to enable dynamic parameters for model used in pipeline.
        Test ensures such change will not invalidate pipeline.
        Test ensures model have no dynamic parameters applied.
    */
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(pipelineOneDynamicParamDummyConfig, fileToReload);
    waitForOVMSConfigReload(manager);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    auto instance = manager.findModelInstance("dummy");
    ASSERT_NE(instance, nullptr);
    ASSERT_FALSE(instance->getModelConfig().isDynamicParameterEnabled());
}

static const char* dummyWithDynamicParamConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1,
                "shape": "auto"
            }
        }
    ]
})";

TEST_F(EnsembleFlowTest, EnablingDynamicParametersAndRemovingPipeline) {
    /*
        This test modifies config.json to enable dynamic parameters for model used in pipeline.
        In the same time, we remove pipeline from config file.
        Test ensures such change is valid and model will be reloaded and dynamic parmeters will be applied.
        Test ensures pipeline gets retired.
    */
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    waitForOVMSConfigReload(manager);
    createConfigFileWithContent(dummyWithDynamicParamConfig, fileToReload);
    waitForOVMSConfigReload(manager);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);

    auto instance = manager.findModelInstance("dummy");
    ASSERT_NE(instance, nullptr);
    ASSERT_TRUE(instance->getModelConfig().isDynamicParameterEnabled());
}
