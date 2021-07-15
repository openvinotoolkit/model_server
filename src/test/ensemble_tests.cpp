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
#include <cstdio>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../binaryutils.hpp"
#include "../dl_node.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "../pipelinedefinition.hpp"
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
        prepareRequest(bs1requestData, request, customPipelineInputName);
        requestData = bs1requestData;
        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
    }

    void prepareRequest(const std::vector<float>& requestData, PredictRequest& request, const std::string& customPipelineInputName) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);
    }

    void prepareRequest(const std::vector<float>& requestData, PredictRequest& request, const std::string& customPipelineInputName, const std::vector<size_t>& shape) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        for (size_t i = 0; i < shape.size(); i++) {
            proto.mutable_tensor_shape()->add_dim()->set_size(shape[i]);
        }
    }

    void prepareBinaryRequest(const std::string& jpegPath, PredictRequest& request, const std::string& customPipelineInputName, int batchSize = 1) {
        std::ifstream DataFile;
        DataFile.open(jpegPath, std::ios::binary);
        DataFile.seekg(0, std::ios::end);
        size_t filesize = DataFile.tellg();
        DataFile.seekg(0);
        std::unique_ptr<char[]> image_bytes(new char[filesize]);
        DataFile.read(image_bytes.get(), filesize);

        tensorflow::TensorProto& inputProto = (*request.mutable_inputs())[customPipelineInputName];
        inputProto.set_dtype(tensorflow::DataType::DT_STRING);
        for (int i = 0; i < batchSize; i++) {
            inputProto.add_string_val(image_bytes.get(), filesize);
        }
        inputProto.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    }

    void checkDummyResponse(int seriesLength, int batchSize = 1) {
        ::checkDummyResponse(customPipelineOutputName, requestData, request, response, seriesLength, batchSize);
    }

    void performWrongPipelineConfigTest(const char* configFileContent) {
        std::string fileToReload = directoryPath + "/ovms_config_file1.json";
        createConfigFileWithContent(configFileContent, fileToReload);
        ConstructorEnabledModelManager managerWithDummyModel;
        managerWithDummyModel.loadConfig(fileToReload);
        std::unique_ptr<Pipeline> pipeline;
        auto status = managerWithDummyModel.createPipeline(pipeline,
            "pipeline1Dummy",
            &request,
            &response);
        ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NAME_MISSING) << status.string();
    }

    ModelConfig config;

    PredictRequest request;
    PredictResponse response;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";
    std::shared_ptr<ovms::TensorInfo> dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        InferenceEngine::Precision::FP32,
        DUMMY_MODEL_SHAPE,
        InferenceEngine::Layout::NC);
    std::shared_ptr<ovms::TensorInfo> dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        InferenceEngine::Precision::FP32,
        DUMMY_MODEL_SHAPE,
        InferenceEngine::Layout::NC);

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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);
    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int dummySeriallyConnectedCount = 1;
    checkDummyResponse(dummySeriallyConnectedCount);
}

TEST_F(EnsembleFlowTest, DummyModelDirectAndPipelineInference) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setNireq(1);
    managerWithDummyModel.reloadModelWithVersions(config);

    // Get dummy model instance
    std::shared_ptr<ovms::ModelInstance> model;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    auto status = managerWithDummyModel.getModelInstance(dummyModelName, 0, model, unload_guard);
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
    ASSERT_EQ(model->infer(&simpleModelRequest, &simpleModelResponse, unload_guard), ovms::StatusCode::OK);

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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int dummySeriallyConnectedCount = 1;
    checkDummyResponse(dummySeriallyConnectedCount);

    // Do the inference directly on dummy model after inference on pipeline
    ASSERT_EQ(model->infer(&simpleModelRequest, &simpleModelResponse, unload_guard), ovms::StatusCode::OK);

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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
    checkDummyResponse(N);
    timer.stop("compare results");

    std::cout << "prepare pipeline: " << timer.elapsed<std::chrono::microseconds>("prepare pipeline") / 1000 << "ms\n";
    std::cout << "pipeline::execute: " << timer.elapsed<std::chrono::microseconds>("pipeline::execute") / 1000 << "ms\n";
    std::cout << "compare results: " << timer.elapsed<std::chrono::microseconds>("compare results") / 1000 << "ms\n";
}

// Disabled with deserialization unification. For this use case to work we would have to additionally rely on "isPipeline" in getFinalShapedTensorInfo() to not use shape from tensor info but to rely on tensorProto
TEST_F(EnsembleFlowTest, DISABLED_ExecutePipelineWithDynamicBatchSize) {
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
        -5, -4, -3, -2, -1, 1, 2, 3, 4, 5,            // batch 1
        -15, -14, -13, -12, -11, 11, 12, 13, 14, 15,  // batch 2
        -25, -24, -23, -22, -21, 21, 22, 23, 24, 25,  // batch 3
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchingParams("auto");
    ConstructorEnabledModelManager managerWithDynamicBatchDummyModel;
    managerWithDynamicBatchDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
    auto outputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{3, DUMMY_MODEL_OUTPUT_SIZE},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{customPipelineOutputName, outputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    pipeline.execute();
    const int seriallyConnectedDummyModels = 1;
    checkDummyResponse(seriallyConnectedDummyModels, batchSize);
}

// Disabled with deserialization unification. For this use case to work we would have to additionally rely on "isPipeline" in getFinalShapedTensorInfo() to not use shape from tensor info but to rely on tensorProto
TEST_F(EnsembleFlowTest, DISABLED_ExecutePipelineWithDynamicShape) {
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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 5},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        InferenceEngine::Precision::FP32,
        shape_t{3, 500},
        InferenceEngine::Layout::NC);
    const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{3, 500},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        InferenceEngine::Precision::FP32,
        shape_t{BATCH_SIZE, WIDTH},
        InferenceEngine::Layout::NC);
    const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{BATCH_SIZE, WIDTH},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), ovms::StatusCode::OK);

    const int seriallyConnectedDummyModels = 1;
    checkDummyResponse(seriallyConnectedDummyModels, BATCH_SIZE);
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
    tensor_map_t inputsInfoTmp;
    for (int i = 0; i < N; i++) {
        const std::string inputName = customPipelineInputName + std::to_string(i);
        inputsInfoTmp[inputName] = std::make_shared<ovms::TensorInfo>(inputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
    }
    const tensor_map_t inputsInfo = inputsInfoTmp;
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    tensor_map_t outputsInfo;
    for (size_t i = 0; i < N; ++i) {
        const std::string outputName = customPipelineOutputName + std::to_string(i);
        outputsInfo.emplace(outputName,
            std::make_shared<ovms::TensorInfo>(outputName,
                InferenceEngine::Precision::FP32,
                DUMMY_MODEL_SHAPE,
                InferenceEngine::Layout::NC));
    }
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);
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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
    ovms::Status fetchResults(NodeSession& nodeSession, SessionResults& sessionResults) override {
        // no release is called as in dl_node.cpp when on error path
        DLNode::fetchResults(nodeSession, sessionResults);
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
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto failInFetchNode = std::make_unique<DLNodeFailInFetch>("failInFetch_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *failInFetchNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*failInFetchNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(failInFetchNode));
    pipeline.push(std::move(output_node));

    auto status = pipeline.execute();
    EXPECT_EQ(status, ovms::StatusCode::UNKNOWN_ERROR) << status.string();
}

TEST_F(EnsembleFlowTest, FailInDLNodeFetchResultsStreamIdReleasedForDeferredNode) {
    // input   dummy(fail in fetch)    output
    //  O------->O------->O
    // input   dummy    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setNireq(1);
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto failInFetchNode = std::make_unique<DLNodeFailInFetch>("failInFetch_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto modelNode = std::make_unique<DLNodeFailInFetch>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);

    pipeline.connect(*input_node, *failInFetchNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*input_node, *modelNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*failInFetchNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});
    pipeline.connect(*modelNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_NOT_IMPORTANT"}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(failInFetchNode));
    pipeline.push(std::move(modelNode));
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

    ASSERT_EQ(manager.reloadModelWithVersions(dummy_1x10), StatusCode::OK_RELOADED);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_1x20), StatusCode::OK_RELOADED);

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
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_fp32), StatusCode::OK_RELOADED);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_u8), StatusCode::OK_RELOADED);

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
    const std::string pipelineName = "my_new_pipeline";
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
    ASSERT_EQ(factory.createDefinition(pipelineName, info, connections, managerWithDummyModel), StatusCode::OK);

    std::unique_ptr<Pipeline> pipeline;

    // Create pipeline out of created definition
    ASSERT_EQ(factory.create(pipeline, pipelineName, &request, &response, managerWithDummyModel), StatusCode::OK);

    // Execute pipeline
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    const int dummySeriallyConnectedCount = 1;
    checkDummyResponse(dummySeriallyConnectedCount);
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

static const std::string PIPELINE_1_DUMMY_NAME = "pipeline1Dummy";

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

TEST_F(EnsembleFlowTest, PipelineFactoryCreationWithInputOutputsMappings) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.loadConfig(fileToReload);
    std::unique_ptr<Pipeline> pipeline;
    auto status = managerWithDummyModel.createPipeline(pipeline,
        "pipeline1Dummy",
        &request,
        &response);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    const int dummySeriallyConnectedCount = 1;
    checkDummyResponse(dummySeriallyConnectedCount);
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
    managerWithDummyModel.loadConfig(fileToReload);
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
    const tensor_map_t inputsInfo{{"proto_input_1x10",
                                      std::make_shared<ovms::TensorInfo>("proto_input_1x10",
                                          InferenceEngine::Precision::FP32,
                                          DUMMY_MODEL_SHAPE,
                                          InferenceEngine::Layout::NC)},
        {"proto_input_1x5",
            std::make_shared<ovms::TensorInfo>("proto_input_1x5",
                InferenceEngine::Precision::FP32,
                shape_t{1, 5},
                InferenceEngine::Layout::NC)}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

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
const std::string notifierDetails{"UnusedNotifierDetails"};

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
    pd.makeSubscriptions(managerWithDummyModel);
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

TEST_F(EnsembleFlowTest, RetirePipelineDefinitionExecuteShouldFail) {
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
    checkDummyResponse(dummySeriallyConnectedCount);
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
    std::unordered_map<std::string, std::unordered_map<std::string, Aliases>> connections;
    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};
    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};
    MockedPipelineDefinitionWithHandlingStatus pd(pipelineName, info, connections);
    pd.makeSubscriptions(managerWithDummyModel);
    std::unique_ptr<Pipeline> pipelineBeforeRetire;
    std::thread t([&managerWithDummyModel, &pd]() {
        std::this_thread::sleep_for(std::chrono::microseconds(PipelineDefinition::WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS / 4));
        auto status = pd.validate(managerWithDummyModel);
        ASSERT_TRUE(status.ok());
        SPDLOG_INFO("Made pd validated");
    });
    auto status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_TRUE(status.ok()) << status.string();
    pd.getControlableStatus().handle(UsedModelChangedEvent(notifierDetails));
    pd.getControlableStatus().handle(ValidationFailedEvent());
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET) << status.string();
    pd.getControlableStatus().handle(UsedModelChangedEvent(notifierDetails));
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_EQ(status, ovms::StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET) << status.string();
    std::thread t2([&managerWithDummyModel, &pd]() {
        std::this_thread::sleep_for(std::chrono::microseconds(PipelineDefinition::WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS / 4));
        auto status = pd.validate(managerWithDummyModel);
        ASSERT_TRUE(status.ok()) << status.string();
        SPDLOG_INFO("Made pd validated");
    });
    status = pd.create(pipelineBeforeRetire, &request, &response, managerWithDummyModel);
    ASSERT_TRUE(status.ok()) << status.string();
    uint dummySeriallyConnectedCount = 1;
    pipelineBeforeRetire->execute();
    checkDummyResponse(dummySeriallyConnectedCount);
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
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    createConfigFileWithContent(configJsonWithNoPipeline, fileToReload);
    manager.loadConfig(fileToReload);
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

TEST_F(EnsembleFlowTest, ReloadPipelineAfterLoadingSuccessfullyChangedInputName) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    auto pdPtr = manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME);
    auto inputsInfoBefore = pdPtr->getInputsInfo();
    ASSERT_EQ(inputsInfoBefore.count(NEW_INPUT_NAME), 0);

    // now reload
    createConfigFileWithContent(pipelineOneDummyConfigWithChangedInputName, fileToReload);
    manager.loadConfig(fileToReload);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    auto inputsInfoAfter = pdPtr->getInputsInfo();
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
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    createConfigFileWithContent(pipelineOneDummyConfigWithMissingModel, fileToReload);
    manager.loadConfig(fileToReload);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}
static const char* pipelineOneDummyConfigWithCorruptedModel = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy-wrong-path-to-model",
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
TEST_F(EnsembleFlowTest, ReloadPipelineAfterLoadingFailDueToCorruptedModel) {
    std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(pipelineOneDummyConfigWithCorruptedModel, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::PATH_INVALID);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    manager.loadConfig(fileToReload);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
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
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RETIRE)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_ADD), nullptr);

    auto pipelineToReloadPtr = manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD);
    auto inputsInfoBefore = pipelineToReloadPtr->getInputsInfo();
    ASSERT_EQ(inputsInfoBefore.count(NEW_INPUT_NAME), 0);

    // now reload
    createConfigFileWithContent(pipelineTwoDummyConfigAfterChanges, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RETIRE)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_RELOAD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_TO_ADD)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    auto inputsInfoAfter = pipelineToReloadPtr->getInputsInfo();
    EXPECT_EQ(inputsInfoAfter.count(NEW_INPUT_NAME), 1);
}

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

TEST_F(EnsembleFlowTest, EnablingDynamicParametersForModelUsedInPipeline) {
    /*
        This test modifies config.json to enable dynamic parameters for model used in pipeline.
        Test ensures such change will not invalidate pipeline.
        Test ensures model have no dynamic parameters applied.
    */
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(pipelineOneDynamicParamDummyConfig, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    auto instance = manager.findModelInstance("dummy");
    ASSERT_NE(instance, nullptr);
    ASSERT_FALSE(instance->getModelConfig().isDynamicParameterEnabled());
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
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

static const char* dummyWithStatefulModelType = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1,
                "stateful": true,
                "low_latency_transformation": true,
                "max_sequence_number": 1000,
                "shape": {"b": "(1,10) "}
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
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(dummyWithDynamicParamConfig, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);

    auto instance = manager.findModelInstance("dummy");
    ASSERT_NE(instance, nullptr);
    ASSERT_TRUE(instance->getModelConfig().isDynamicParameterEnabled());
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(EnsembleFlowTest, EnablingStatefulParamteresForModelUsedInPipeline) {
    /*
        This test modifies config.json to enable stateful model used in pipeline.
        In the same time, we remove pipeline from config file.
        Test ensures such change is valid and model will be reloaded and stateful model will be loaded.
        Test ensures pipeline gets retired.
    */
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(dummyWithStatefulModelType, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);

    auto instance = manager.findModelInstance("dummy");
    ASSERT_NE(instance, nullptr);
    ASSERT_FALSE(instance->getModelConfig().isStateful());  // Switching model type is not valid
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
}

static const char* pipelineModelSameNameConfig = R"(
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
        },
        {
        "config": {
                "name": "pipeline1Dummy",
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

static const char* pipelineModelSameNameConfigNoPipeline = R"(
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
        },
        {
        "config": {
                "name": "pipeline1Dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1,
                "shape": "auto"
            }
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineConfigModelWithSameName) {
    // Expected result - model added, adding pipeline failed
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineModelSameNameConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::PIPELINE_NAME_OCCUPIED);

    ASSERT_FALSE(manager.getPipelineFactory().definitionExists(PIPELINE_1_DUMMY_NAME));

    auto instance = manager.findModelInstance(PIPELINE_1_DUMMY_NAME);
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(EnsembleFlowTest, ModelLoadedAddPipelineWithSameName) {
    // Expected result - adding pipeline failed
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineModelSameNameConfigNoPipeline, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    auto instance = manager.findModelInstance(PIPELINE_1_DUMMY_NAME);
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);

    createConfigFileWithContent(pipelineModelSameNameConfig, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_FALSE(manager.getPipelineFactory().definitionExists(PIPELINE_1_DUMMY_NAME));

    instance = manager.findModelInstance(PIPELINE_1_DUMMY_NAME);
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
}

TEST_F(EnsembleFlowTest, PipelineLoadedAddModelWithSameName) {
    // Expected result - adding model failed
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(pipelineModelSameNameConfig, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_TRUE(manager.getPipelineFactory().definitionExists(PIPELINE_1_DUMMY_NAME));
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    auto instance = manager.findModelInstance(PIPELINE_1_DUMMY_NAME);
    ASSERT_EQ(instance, nullptr);
}

TEST_F(EnsembleFlowTest, PipelineRetiredAddModelWithSameName) {
    // Expected result - adding model failed
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(pipelineModelSameNameConfigNoPipeline, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_TRUE(manager.getPipelineFactory().definitionExists(PIPELINE_1_DUMMY_NAME));
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::RETIRED);

    auto instance = manager.findModelInstance(PIPELINE_1_DUMMY_NAME);
    ASSERT_EQ(instance, nullptr);
}

static const char* pipelinePipelineSameNameConfig = R"(
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
        },
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
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

TEST_F(EnsembleFlowTest, PipelineAddSecondPipelineWithSameName) {
    // Expected result - adding second pipeline fails
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status.ok()) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);

    createConfigFileWithContent(pipelinePipelineSameNameConfig, fileToReload);
    status = manager.loadConfig(fileToReload);

    ASSERT_TRUE(manager.getPipelineFactory().definitionExists(PIPELINE_1_DUMMY_NAME));
    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
    auto& nodeInfos = manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getNodeInfos();
    ASSERT_FALSE(std::find_if(nodeInfos.begin(), nodeInfos.end(), [](auto nodeInfo) { return nodeInfo.nodeName == "dummyNode"; }) == nodeInfos.end());
    ASSERT_TRUE(std::find_if(nodeInfos.begin(), nodeInfos.end(), [](auto nodeInfo) { return nodeInfo.nodeName == "dummyNode2"; }) == nodeInfos.end());
}

static const char* pipelineDemultiplexerBatchSize = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "batch_size": 2,
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
                    ],
                    "demultiply_count": 2
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

TEST_F(EnsembleFlowTest, DemultiplexerMultipleBatchSizeNotAllowed) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineDemultiplexerBatchSize, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::PIPELINE_DEMULTIPLEXER_MULTIPLE_BATCH_SIZE);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

static const char* pipelineDemultiplexerShape = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(3, 10) ",
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
                    ],
                    "demultiply_count": 2
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

TEST_F(EnsembleFlowTest, DemultiplexerMultipleBatchSizeWithShapeNotAllowed) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineDemultiplexerShape, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::PIPELINE_DEMULTIPLEXER_MULTIPLE_BATCH_SIZE);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

static const char* pipelineSingleIncrement4DimInputNHWC = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, ExecuteSingleIncrement4DimInputNHWC) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimInputNHWC, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, request, "pipeline_input", {1, 1, 2, 3});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {2.0, 5.0, 3.0, 6.0, 4.0, 7.0}, request, response, {1, 3, 1, 2});
}

static const char* pipelineSingleIncrement4DimInputNHWCDynamicBatch = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, ExecuteSingleIncrement4DimInputNHWCDynamicBatch) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimInputNHWCDynamicBatch, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
        request, "pipeline_input", {2, 1, 1, 2, 3});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 11.0, 41.0, 21.0, 51.0, 31.0, 61.0}, request, response, {2, 1, 3, 1, 2});
}

static const char* pipelineSingleIncrement4DimOutputNHWC = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"output": "nhwc"},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, ExecuteSingleIncrement4DimOutputNHWC) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, request, "pipeline_input", {1, 3, 1, 2});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {2.0, 4.0, 6.0, 3.0, 5.0, 7.0}, request, response, {1, 1, 2, 3});
}

static const char* pipelineSingleIncrement4DimOutputNHWCDynamicBatch = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"output": "nhwc"},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, ExecuteSingleIncrement4DimOutputNHWCDynamicBatch) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCDynamicBatch, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                       10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
        request, "pipeline_input", {2, 1, 3, 1, 2});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 11.0, 31.0, 51, 21.0, 41.0, 61.0}, request, response, {2, 1, 1, 2, 3});
}

static const char* pipelineAmbiguousInputMeta = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment_nhwc",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"input": "nhwc"},
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "increment_nchw",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"input": "nchw"},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node_nhwc",
                    "model_name": "increment_nhwc",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                },
                {
                    "name": "increment_node_nchw",
                    "model_name": "increment_nchw",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output_nhwc": {"node_name": "increment_node_nhwc",
                                          "data_item": "out"}
                },
                {"pipeline_output_nchw": {"node_name": "increment_node_nchw",
                                          "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, PipelineAmbiguousInputMetaFailsToLoad) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineAmbiguousInputMeta, fileToReload);
    ConstructorEnabledModelManager manager;
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::PIPELINE_INPUTS_AMBIGUOUS_METADATA);
}

static const char* pipelineInnerConnectedNhwc = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment_nchw_in_nhwc_out",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"output": "nhwc"},
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "increment_nhwc_in_nchw_out",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,2) ",
                "layout": {"input": "nhwc"},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node_nchw_in_nhwc_out",
                    "model_name": "increment_nchw_in_nhwc_out",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                },
                {
                    "name": "increment_node_nhwc_in_nchw_out",
                    "model_name": "increment_nhwc_in_nchw_out",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "increment_node_nchw_in_nhwc_out",
                                   "data_item": "out"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node_nhwc_in_nchw_out",
                                          "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, ExecutePipelineWithInnerNhwcConnection) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineInnerConnectedNhwc, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, request, "pipeline_input", {1, 3, 1, 2});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, request, response, {1, 3, 1, 2});
}

class EnsembleFlowTestBinaryInput : public EnsembleFlowTest {
public:
    const std::string imagePath = "/ovms/src/test/binaryutils/rgb.jpg";
    const std::string graycaleImagePath = "/ovms/src/test/binaryutils/grayscale.jpg";
};

static const char* pipelineSingleIncrement4DimOutputNHWC1x1 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,1) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, BatchSize1) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(imagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {37.0, 28.0, 238.0}, request, response, {1, 3, 1, 1});
}

static const char* pipelineSingleIncrement4DimOutputNCHW1x1 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,1) ",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, NchwEntryNotSupported) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNCHW1x1, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(imagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::UNSUPPORTED_LAYOUT);
}

static const char* pipelineSingleIncrement4DimOutputNHWC1x1Grayscale = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1,1,1) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, GrayscaleImage) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1Grayscale, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(graycaleImagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {1.0}, request, response, {1, 1, 1, 1});
}

static const char* pipelineSingleIncrement4DimOutputNHWC1x1BS5 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(5,3,1,1) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, BatchSize5) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1BS5, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, request, response, {5, 3, 1, 1});
}

static const char* pipelineSingleIncrement4DimOutputNHWC2x2 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,2,2) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, ResizeBatch1) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC2x2, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(imagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0}, request, response, {1, 3, 2, 2});
}

static const char* pipelineSingleIncrement4DimOutputNHWC2x2BS5 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(5,3,2,2) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, ResizeBatch5) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC2x2BS5, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0},
        request, response, {5, 3, 2, 2});
}

static const char* pipelineSingleIncrement4DimOutputNHWC1Channel = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1,1,1) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, ColorChannelsDiffer) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1Channel, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(imagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(EnsembleFlowTestBinaryInput, InvalidData) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    tensorflow::TensorProto& inputProto = (*request.mutable_inputs())["pipeline_input"];
    inputProto.set_dtype(tensorflow::DataType::DT_STRING);
    inputProto.add_string_val("INVALID_IMAGE");
    inputProto.mutable_tensor_shape()->add_dim()->set_size(1);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::IMAGE_PARSING_FAILED);
}

static const char* pipelineSingleIncrement4DimOutputNHWC1x1EntryDemultiplexer = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,3,1,1) ",
                "layout": "nhwc",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output",
                         "alias": "out"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "increment_node",
                                     "data_item": "out"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, EntryDemultiplexer) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1EntryDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    checkIncrement4DimResponse("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, request, response, {5, 1, 3, 1, 1});
}
