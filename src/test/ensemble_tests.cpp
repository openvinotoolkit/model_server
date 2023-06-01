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
#include <memory>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../dags/dl_node.hpp"
#include "../dags/entry_node.hpp"
#include "../dags/exit_node.hpp"
#include "../dags/nodestreamidguard.hpp"
#include "../dags/pipeline.hpp"
#include "../dags/pipeline_factory.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../metric_registry.hpp"
#include "../model_metric_reporter.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "../tensor_conversion.hpp"
#include "../timer.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

using ::testing::ElementsAre;

const uint NIREQ = 2;

template <typename Pair,
    typename RequestType = typename Pair::first_type,
    typename ResponseType = typename Pair::second_type>
class EnsembleFlowBothApiTest : public TestWithTempDir {
public:
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        config = DUMMY_MODEL_CONFIG;
        config.setNireq(NIREQ);

        reporter = std::make_unique<ModelMetricReporter>(&this->metricConfig, &this->registry, "example_pipeline_name", 1);

        // Prepare request
        prepareRequest(bs1requestData, request, customPipelineInputName);
        requestData = bs1requestData;
        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
            ovms::Precision::FP32,
            DUMMY_MODEL_SHAPE_META,
            Layout{"NC"});
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
            ovms::Precision::FP32,
            DUMMY_MODEL_SHAPE_META,
            Layout{"NC"});
    }

    void prepareRequest(const std::vector<float>& requestData, TFSRequestType& request, const std::string& customPipelineInputName, const ovms::signed_shape_t& shape = {1, DUMMY_MODEL_INPUT_SIZE}) {
        request.Clear();
        preparePredictRequest(request, inputs_info_t{{customPipelineInputName, {shape, ovms::Precision::FP32}}}, requestData);
    }

    void prepareRequest(const std::vector<float>& requestData, KFSRequest& request, const std::string& customPipelineInputName, const ovms::signed_shape_t& shape = {1, DUMMY_MODEL_INPUT_SIZE}) {
        request.Clear();
        prepareKFSInferInputTensor(request, customPipelineInputName, std::make_tuple(shape, ovmsPrecisionToKFSPrecision(ovms::Precision::FP32)), requestData);
    }

    void checkDummyResponse(int seriesLength, int batchSize = 1, const std::string& servableName = "") {
        ::checkDummyResponse(customPipelineOutputName, requestData, request, response, seriesLength, batchSize, servableName);
    }

    ModelConfig config;
    RequestType request;
    ResponseType response;
    MetricRegistry registry;
    MetricConfig metricConfig;
    std::unique_ptr<ModelMetricReporter> reporter;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";
    std::shared_ptr<const ovms::TensorInfo> dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});
    std::shared_ptr<const ovms::TensorInfo> dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});

    std::vector<float> requestData;
    const std::vector<float> bs1requestData{-5.0, 3.0, 0.0, -12.0, 9.0, -100.0, 102.0, 92.0, -1.0, 12.0};
};

// New test suite. Tests both - TFS and KFS.
// Currently only tests which contain gather in exit node.
using MyTypes = ::testing::Types<TFSInterface, KFSInterface>;
TYPED_TEST_SUITE(EnsembleFlowBothApiTest, MyTypes);

// Old test suite. Tests only TFS API.
class EnsembleFlowTest : public TestWithTempDir {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        config = DUMMY_MODEL_CONFIG;
        config.setNireq(NIREQ);

        reporter = std::make_unique<ModelMetricReporter>(&this->metricConfig, &this->registry, "example_pipeline_name", 1);

        // Prepare request
        prepareRequest(bs1requestData, request, customPipelineInputName);
        requestData = bs1requestData;
        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
            ovms::Precision::FP32,
            DUMMY_MODEL_SHAPE_META,
            Layout{"NC"});
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
            ovms::Precision::FP32,
            DUMMY_MODEL_SHAPE_META,
            Layout{"NC"});
    }

    void prepareRequest(const std::vector<float>& requestData, PredictRequest& request, const std::string& customPipelineInputName) {
        request.Clear();
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);
    }

    void prepareRequest(const std::vector<float>& requestData, PredictRequest& request, const std::string& customPipelineInputName, const std::vector<size_t>& shape) {
        request.Clear();
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        for (size_t i = 0; i < shape.size(); i++) {
            proto.mutable_tensor_shape()->add_dim()->set_size(shape[i]);
        }
    }

    void prepareBinaryRequest(const std::string& jpegPath, PredictRequest& request, const std::string& customPipelineInputName, int batchSize = 1) {
        size_t filesize;
        std::unique_ptr<char[]> image_bytes;
        readImage(jpegPath, filesize, image_bytes);

        request.Clear();
        tensorflow::TensorProto& inputProto = (*request.mutable_inputs())[customPipelineInputName];
        inputProto.set_dtype(tensorflow::DataType::DT_STRING);
        for (int i = 0; i < batchSize; i++) {
            inputProto.add_string_val(image_bytes.get(), filesize);
        }
        inputProto.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    }

    void prepareMisalignedBinaryImageRequest(const std::string& image1, const std::string& image2, PredictRequest& request, const std::string& customPipelineInputName) {
        request.Clear();
        tensorflow::TensorProto& inputProto = (*request.mutable_inputs())[customPipelineInputName];
        inputProto.set_dtype(tensorflow::DataType::DT_STRING);

        size_t filesize;
        std::unique_ptr<char[]> image_bytes;
        readImage(image1, filesize, image_bytes);
        inputProto.add_string_val(image_bytes.get(), filesize);

        readImage(image2, filesize, image_bytes);
        inputProto.add_string_val(image_bytes.get(), filesize);

        inputProto.mutable_tensor_shape()->add_dim()->set_size(2);
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
    MetricRegistry registry;
    MetricConfig metricConfig;
    std::unique_ptr<ModelMetricReporter> reporter;

    std::string dummyModelName = "dummy";
    std::optional<model_version_t> requestedModelVersion{std::nullopt};
    const std::string customPipelineInputName = "custom_dummy_input";
    const std::string customPipelineOutputName = "custom_dummy_output";
    std::shared_ptr<const ovms::TensorInfo> dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});
    std::shared_ptr<const ovms::TensorInfo> dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});

    std::vector<float> requestData;
    const std::vector<float> bs1requestData{-5.0, 3.0, 0.0, -12.0, 9.0, -100.0, 102.0, 92.0, -1.0, 12.0};
};

TYPED_TEST(EnsembleFlowBothApiTest, DummyModel) {
    // Most basic configuration, just process single dummy model request
    // input   dummy    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(this->config);

    // Configure pipeline
    const tensor_map_t inputsInfo{{this->customPipelineInputName, this->dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<typename TypeParam::first_type>>(&this->request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", this->dummyModelName, this->requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{this->customPipelineOutputName, this->dagDummyModelOutputTensorInfo}};
    std::set<std::string> gatherFromNode = {};
    std::string pipelineName = "test_pipeline";
    auto output_node = std::make_unique<ExitNode<typename TypeParam::second_type>>(&this->response, outputsInfo, gatherFromNode, true, pipelineName);
    Pipeline pipeline(*input_node, *output_node, *this->reporter);
    pipeline.connect(*input_node, *model_node, {{this->customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, this->customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    const int dummySeriallyConnectedCount = 1;
    this->checkDummyResponse(dummySeriallyConnectedCount, 1, pipelineName);
}

TYPED_TEST(EnsembleFlowBothApiTest, TwoInnerNodesConnectedShapeRangePartiallyMatching) {
    ConstructorEnabledModelManager managerWithDummyModel;

    this->config = DUMMY_MODEL_CONFIG;
    this->config.setName("dummy_A");
    this->config.setBatchSize(std::nullopt);
    this->config.parseShapeParameter("(-1,1:3)");
    managerWithDummyModel.reloadModelWithVersions(this->config);

    this->config = DUMMY_MODEL_CONFIG;
    this->config.setName("dummy_B");
    this->config.setBatchSize(std::nullopt);
    this->config.parseShapeParameter("(-1,2:4)");
    managerWithDummyModel.reloadModelWithVersions(this->config);

    // Configure pipeline
    this->dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(this->customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{Dimension::any(), {1, 3}},
        Layout{"NC"});
    this->dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(this->customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{Dimension::any(), {2, 4}},
        Layout{"NC"});

    // 2x2 passing
    {
        this->prepareRequest(std::vector<float>{5.0, 6.0, 15.0, 16.0}, this->request, this->customPipelineInputName, {2, 2});
        this->response.Clear();

        const tensor_map_t inputsInfo{{this->customPipelineInputName, this->dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<typename TypeParam::first_type>>(&this->request, inputsInfo);
        auto model_node_A = std::make_unique<DLNode>("dummy_node_A", "dummy_A", this->requestedModelVersion, managerWithDummyModel);
        auto model_node_B = std::make_unique<DLNode>("dummy_node_B", "dummy_B", this->requestedModelVersion, managerWithDummyModel);
        const tensor_map_t outputsInfo{{this->customPipelineOutputName, this->dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<typename TypeParam::second_type>>(&this->response, outputsInfo);
        Pipeline pipeline(*input_node, *output_node, *this->reporter);
        pipeline.connect(*input_node, *model_node_A, {{this->customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_A, *model_node_B, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_B, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, this->customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node_A));
        pipeline.push(std::move(model_node_B));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
        checkIncrement4DimResponse<float>(this->customPipelineOutputName, std::vector<float>{7.0, 8.0, 17.0, 18.0}, this->response, {2, 2});
    }

    // 2x4 not passing due to not matched dummy_A (but matching dummy_B)
    {
        this->prepareRequest(std::vector<float>{5.0, 6.0, 15.0, 16.0, 5.0, 6.0, 15.0, 16.0}, this->request, this->customPipelineInputName, {2, 4});
        this->response.Clear();

        const tensor_map_t inputsInfo{{this->customPipelineInputName, this->dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<typename TypeParam::first_type>>(&this->request, inputsInfo);
        auto model_node_A = std::make_unique<DLNode>("dummy_node_A", "dummy_A", this->requestedModelVersion, managerWithDummyModel);
        auto model_node_B = std::make_unique<DLNode>("dummy_node_B", "dummy_B", this->requestedModelVersion, managerWithDummyModel);
        const tensor_map_t outputsInfo{{this->customPipelineOutputName, this->dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<typename TypeParam::second_type>>(&this->response, outputsInfo);
        Pipeline pipeline(*input_node, *output_node, *this->reporter);
        pipeline.connect(*input_node, *model_node_A, {{this->customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_A, *model_node_B, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_B, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, this->customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node_A));
        pipeline.push(std::move(model_node_B));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
    }

    // 2x1 not passing due to not matched dummy_B (but matching dummy_A)
    {
        this->prepareRequest(std::vector<float>{5.0, 6.0}, this->request, this->customPipelineInputName, {2, 1});
        this->response.Clear();

        const tensor_map_t inputsInfo{{this->customPipelineInputName, this->dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<typename TypeParam::first_type>>(&this->request, inputsInfo);
        auto model_node_A = std::make_unique<DLNode>("dummy_node_A", "dummy_A", this->requestedModelVersion, managerWithDummyModel);
        auto model_node_B = std::make_unique<DLNode>("dummy_node_B", "dummy_B", this->requestedModelVersion, managerWithDummyModel);
        const tensor_map_t outputsInfo{{this->customPipelineOutputName, this->dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<typename TypeParam::second_type>>(&this->response, outputsInfo);
        Pipeline pipeline(*input_node, *output_node, *this->reporter);
        pipeline.connect(*input_node, *model_node_A, {{this->customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_A, *model_node_B, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node_B, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, this->customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node_A));
        pipeline.push(std::move(model_node_B));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
    }
}

// This test is only theoretical scenario, since pipeline validation should not allow such pipelines.
TEST_F(EnsembleFlowTest, TwoInnerNodesConnectedShapeRangeNotMatching) {
    ConstructorEnabledModelManager managerWithDummyModel;

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_A");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(-1,1:3)");
    managerWithDummyModel.reloadModelWithVersions(config);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_B");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(-1,4:6)");
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{Dimension::any(), {1, 3}},
        Layout{"NC"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{Dimension::any(), {4, 6}},
        Layout{"NC"});

    // 2x2 not matching dummy_B at execution time
    prepareRequest(std::vector<float>{5.0, 6.0, 15.0, 16.0}, request, customPipelineInputName, {2, 2});
    response.Clear();

    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node_A = std::make_unique<DLNode>("dummy_node_A", "dummy_A", requestedModelVersion, managerWithDummyModel);
    auto model_node_B = std::make_unique<DLNode>("dummy_node_B", "dummy_B", requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);
    Pipeline pipeline(*input_node, *output_node, *this->reporter);
    pipeline.connect(*input_node, *model_node_A, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_A, *model_node_B, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_B, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node_A));
    pipeline.push(std::move(model_node_B));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
}

class EnsembleFlowValidationTest : public EnsembleFlowTest {
public:
    std::unique_ptr<Pipeline> createDummyPipeline(ConstructorEnabledModelManager& managerWithDummyModel) {
        const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
        const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);
        auto pipeline = std::make_unique<Pipeline>(*input_node, *output_node, *this->reporter);
        pipeline->connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline->connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline->push(std::move(input_node));
        pipeline->push(std::move(model_node));
        pipeline->push(std::move(output_node));
        return pipeline;
    }
};

TEST_F(EnsembleFlowValidationTest, DummyModelValid) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorNumberOfInputs) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())["input1"];
    auto& proto2 = (*request.mutable_inputs())["input2"];

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_INPUTS);
    proto1.Clear();
    proto2.Clear();
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorMissingInput) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())["input1"];

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_MISSING_INPUT);
    proto1.Clear();
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorShapeValueNegative) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(-10);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorBinaryInputWrongNumberOfShapeDimensions) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.set_dtype(tensorflow::DataType::DT_STRING);
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);

    // enforce the endpoint to be 4d to not fall into string handling
    this->dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(this->customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{1, 224, 224, 3},
        ovms::Layout{"NHWC"});
    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorBinaryInputBatchSizeMismatch) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.set_dtype(tensorflow::DataType::DT_STRING);
    proto1.mutable_tensor_shape()->add_dim()->set_size(2);

    // enforce the endpoint to be 4d to not fall into string handling
    this->dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(this->customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{1, 224, 224, 3},
        ovms::Layout{"NHWC"});
    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorPrecisionMismatch) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(10);
    proto1.set_dtype(tensorflow::DataType::DT_INT32);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorInvalidNumberOfShapeDimensions) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(10);
    proto1.mutable_tensor_shape()->add_dim()->set_size(3);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorInvalidBatchSize) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(2);
    proto1.mutable_tensor_shape()->add_dim()->set_size(10);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorInvalidShape) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(11);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleFlowValidationTest, DummyModelProtoValidationErrorInvalidTensorContentSize) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.mutable_tensor_shape()->add_dim()->set_size(10);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);
    const std::vector<float> data{1.0f};
    proto1.mutable_tensor_content()->assign((char*)data.data(), data.size() * sizeof(float));

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_CONTENT_SIZE);
}

class EnsembleFlowValidationShapeRangeTest : public EnsembleFlowValidationTest {
protected:
    void SetUp() {
        EnsembleFlowValidationTest::SetUp();

        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
            ovms::Precision::FP32,
            ovms::Shape{{1, 10}, {2, 11}},
            Layout{"NC"});
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
            ovms::Precision::FP32,
            ovms::Shape{{1, 10}, {2, 11}},
            Layout{"NC"});

        config = DUMMY_MODEL_CONFIG;
        config.setBatchingParams("0");
        config.parseShapeParameter("(1:10,2:11)");
    }
};

TEST_F(EnsembleFlowValidationShapeRangeTest, DummyModelValid) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
}

TEST_F(EnsembleFlowValidationShapeRangeTest, DummyModelProtoValidationErrorInvalidBatchSize) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(11);
    proto1.mutable_tensor_shape()->add_dim()->set_size(10);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(EnsembleFlowValidationShapeRangeTest, DummyModelProtoValidationErrorInvalidShape) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    request.Clear();
    auto& proto1 = (*request.mutable_inputs())[customPipelineInputName];
    proto1.mutable_tensor_shape()->add_dim()->set_size(6);
    proto1.mutable_tensor_shape()->add_dim()->set_size(1);
    proto1.set_dtype(tensorflow::DataType::DT_FLOAT);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
}

class EnsembleFlowValidationShapeAnyTest : public EnsembleFlowValidationTest {
protected:
    void SetUp() {
        EnsembleFlowValidationTest::SetUp();

        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
            ovms::Precision::FP32,
            ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()},
            Layout{"NC"});
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
            ovms::Precision::FP32,
            ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()},
            Layout{"NC"});

        config = DUMMY_MODEL_CONFIG;
        config.setBatchingParams("0");
        config.parseShapeParameter("(-1,-1)");
    }
};

TEST_F(EnsembleFlowValidationShapeAnyTest, DummyModelValid) {
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    auto pipeline = createDummyPipeline(managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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
    tensorflow::serving::PredictRequest simpleModelRequest;
    preparePredictRequest(simpleModelRequest,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
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
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);
    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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

    enum : unsigned int {
        PREPARE,
        EXECUTE,
        COMPARE,
        TIMER_END
    };
    Timer<TIMER_END> timer;
    timer.start(PREPARE);

    const int N = 100;
    // input      dummy x N      output
    //  O------->O->O...O->O------->O

    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    std::unique_ptr<DLNode> dummy_nodes[N];
    for (int i = 0; i < N; i++) {
        dummy_nodes[i] = std::make_unique<DLNode>("dummy_node_" + std::to_string(i), dummyModelName, requestedModelVersion, managerWithDummyModel);
    }

    Pipeline pipeline(*input_node, *output_node, *this->reporter);
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

    timer.stop(PREPARE);
    timer.start(EXECUTE);
    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    timer.stop(EXECUTE);

    timer.start(COMPARE);
    checkDummyResponse(N);
    timer.stop(COMPARE);

    std::cout << "prepare pipeline: " << timer.elapsed<std::chrono::microseconds>(PREPARE) / 1000 << "ms\n";
    std::cout << "pipeline::execute: " << timer.elapsed<std::chrono::microseconds>(EXECUTE) / 1000 << "ms\n";
    std::cout << "compare results: " << timer.elapsed<std::chrono::microseconds>(COMPARE) / 1000 << "ms\n";
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithBatchSizeAny) {
    // Scenario

    // input(3x10)   dummy(1x10), change batch size to any    output(3x10)
    //  O-------------------------->O----------------------------->O

    // input 3x10
    // dummy is natively 1x10, batch size change to -1 (any)
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

    config.setBatchingParams("-1");
    ConstructorEnabledModelManager managerWithDynamicBatchDummyModel;
    managerWithDynamicBatchDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName, ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), 10}, Layout{"NC"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), 10}, Layout{"NC"});
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    const int seriallyConnectedDummyModels = 1;
    checkDummyResponse(seriallyConnectedDummyModels, batchSize);
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithBatchSizeRange) {
    // Scenario

    // input(3x10)   dummy(1x10), change batch size to (1:5x10)    output(3x10)
    //  O-------------------------->O------------------------------->O

    // input 3x10
    // dummy is natively 1x10, batch size change to 1:5 (range)
    // process dummy
    // check if output is 3x10
    // check if execution fails for batch higher than 5

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    int batchSize = 3;
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(batchSize);
    requestData = {
        -5, -4, -3, -2, -1, 1, 2, 3, 4, 5,            // batch 1
        -15, -14, -13, -12, -11, 11, 12, 13, 14, 15,  // batch 2
        -25, -24, -23, -22, -21, 21, 22, 23, 24, 25,  // batch 3
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchingParams("1:5");
    ConstructorEnabledModelManager managerWithDynamicBatchDummyModel;
    managerWithDynamicBatchDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName, ovms::Precision::FP32, ovms::Shape{{1, 5}, 10}, Layout{"NC"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, ovms::Precision::FP32, ovms::Shape{{1, 5}, 10}, Layout{"NC"});
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
        const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
        const int seriallyConnectedDummyModels = 1;
        checkDummyResponse(seriallyConnectedDummyModels, batchSize);
    }
    // Prepare invalid data
    batchSize = 6;
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(batchSize);
    requestData = std::vector<float>(batchSize * 10, 1.234);
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
        const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_BATCH_SIZE);
    }
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithShapeAny) {
    // Scenario

    // input(1x5)      dummy(1x10) second dimension set to any             output(1x5)
    //  O---------------------->O----------------------------------------------->O

    // input 1x5
    // dummy is natively 1x10, but second dimension set to any (-1)
    // process dummy
    // check if output is 1x5

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(5);
    std::vector<float> requestData = {
        -5, -4, -3, -2, -1,  // batch 1
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchSize(std::nullopt);  // = not specified in --batch_size parameter
    config.parseShapeParameter("(1,-1)");
    ConstructorEnabledModelManager managerWithDynamicShapeDummyModel;
    managerWithDynamicShapeDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName, ovms::Precision::FP32, ovms::Shape{1, ovms::Dimension::any()}, Layout{"NC"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, ovms::Precision::FP32, ovms::Shape{1, ovms::Dimension::any()}, Layout{"NC"});
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);

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

TEST_F(EnsembleFlowTest, ExecutePipelineWithShapeRange) {
    // Scenario

    // input(1x5)      dummy(1x10) second dimension set to range (1:5)             output(1x5)
    //  O---------------------->O----------------------------------------------->O

    // input 1x5
    // dummy is natively 1x10, but second dimension set to (1:5) range
    // process dummy
    // check if output is 1x5
    // check if there is an error for input dimension higher than 5

    tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName];
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(5);
    std::vector<float> requestData = {
        -5, -4, -3, -2, -1,  // batch 1
    };
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));

    config.setBatchSize(std::nullopt);  // = not specified in --batch_size parameter
    config.parseShapeParameter("(1,1:5)");
    ConstructorEnabledModelManager managerWithDynamicShapeDummyModel;
    managerWithDynamicShapeDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName, ovms::Precision::FP32, ovms::Shape{1, {1, 5}}, Layout{"NC"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName, ovms::Precision::FP32, ovms::Shape{1, {1, 5}}, Layout{"NC"});
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
        const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);

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
    // Prepare invalid data
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(6);
    requestData = std::vector<float>(6, 1.234);
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
        const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
    }
}

TEST_F(EnsembleFlowTest, ExecutePipelineWithBatchAndShapeSetToAny) {
    // Scenario

    // input(3x500)   dummy(1x10), all dimensions set to any    output(3x500)
    //  O------------------------------>O----------------------------->O

    // input 3x500
    // dummy is natively 1x10, but all dimensions set to any
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

    config.setBatchSize(std::nullopt);  // simulate --batch_size parameter not set
    config.parseShapeParameter("(-1,-1)");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()},
        Layout{"NC"});
    const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()},
        Layout{"NC"});
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::OK);

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

TEST_F(EnsembleFlowTest, ExecutePipelineWithBatchAndShapeSetToRange) {
    // Scenario

    // input(3x500)   dummy(1x10), all dimensions set to range (1:1000)    output(3x500)
    //  O------------------------------>O----------------------------->O

    // input 3x500
    // dummy is natively 1x10, but all dimensions are reset to support range (1:1000)
    // process dummy
    // check if output is 3x500
    // check if for input dimension higher than 1000 there is an error

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

    config.setBatchSize(std::nullopt);  // simulate --batch_size parameter not set
    config.parseShapeParameter("(1:1000,1:1000)");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{{1, 1000}, {1, 1000}},
        Layout{"NC"});
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{{1, 1000}, {1, 1000}},
        Layout{"NC"});
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);

        const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::OK);

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
    // Prepare invalid data
    proto.mutable_tensor_shape()->mutable_dim(1)->set_size(1001);
    requestData = std::vector<float>(BATCH_SIZE * 1001, 1.234);
    proto.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
    {
        const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
        auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
        auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);

        const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
        auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

        Pipeline pipeline(*input_node, *output_node, *this->reporter);

        pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

        pipeline.push(std::move(input_node));
        pipeline.push(std::move(model_node));
        pipeline.push(std::move(output_node));

        ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::INVALID_SHAPE);
    }
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
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicBatchDummyModel);
    auto outputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{3, DUMMY_MODEL_OUTPUT_SIZE},
        Layout{"NC"});
    const tensor_map_t outputsInfo{{customPipelineOutputName, outputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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

    config.setBatchSize(std::nullopt);  // = not specified in --batch_size parameter
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager managerWithDynamicShapeDummyModel;
    managerWithDynamicShapeDummyModel.reloadModelWithVersions(config);

    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDynamicShapeDummyModel);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{1, 5},
        Layout{"NC"});
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);

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

TEST_F(EnsembleFlowTest, DISABLED_ExecutePipelineWithDynamicBatchAndShape) {
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

    config.setBatchSize(std::nullopt);  // simulate --batch_size parameter not set
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{3, 500},
        Layout{"NC"});
    const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{3, 500},
        Layout{"NC"});
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::OK);

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

TEST_F(EnsembleFlowTest, DISABLED_ExecutePipelineWithDynamicShape_RequestHasDifferentDim0) {
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

    config.setBatchSize(std::nullopt);  // simulate --batch_size parameter not set
    config.parseShapeParameter("auto");
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);

    // Configure pipeline
    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        ovms::Shape{BATCH_SIZE, WIDTH},
        Layout{"NC"});
    const tensor_map_t inputsInfo{{customPipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, manager);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        ovms::Shape{BATCH_SIZE, WIDTH},
        Layout{"NC"});
    const tensor_map_t outputsInfo{{customPipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::OK);

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
            ovms::Precision::FP32,
            DUMMY_MODEL_SHAPE_META,
            Layout{"NC"});
    }
    const tensor_map_t inputsInfo = inputsInfoTmp;
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    tensor_map_t outputsInfo;
    for (size_t i = 0; i < N; ++i) {
        const std::string outputName = customPipelineOutputName + std::to_string(i);
        outputsInfo.emplace(outputName,
            std::make_shared<ovms::TensorInfo>(outputName,
                ovms::Precision::FP32,
                DUMMY_MODEL_SHAPE_META,
                Layout{"NC"}));
    }
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);
    Pipeline pipeline(*input_node, *output_node, *this->reporter);
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
    request.Clear();
    for (int i = 0; i < N; i++) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[customPipelineInputName + std::to_string(i)];
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        proto.mutable_tensor_content()->assign((char*)(requestDataT.data() + i * DUMMY_MODEL_INPUT_SIZE),
            DUMMY_MODEL_INPUT_SIZE * sizeof(float));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(10);
    }
    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::OK);
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

class DLNodeFirst : public DLNode {
    std::vector<int>& order;

public:
    DLNodeFirst(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager, std::vector<int>& order, std::unordered_map<std::string, std::string> nodeOutputNameAlias = {},
        std::optional<int32_t> demultiplyCount = std::nullopt, std::set<std::string> gatherFromNode = {}) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount, gatherFromNode),
        order(order) {}
    ovms::Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override {
        auto status = DLNode::execute(sessionId, notifyEndQueue);
        order.push_back(1);
        return status;
    }
};

class DLNodeDeferred : public DLNode {
    std::vector<int>& order;

public:
    DLNodeDeferred(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager, std::vector<int>& order, std::unordered_map<std::string, std::string> nodeOutputNameAlias = {},
        std::optional<int32_t> demultiplyCount = std::nullopt, std::set<std::string> gatherFromNode = {}) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount, gatherFromNode),
        order(order) {}
    ovms::Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override {
        auto status = DLNode::execute(sessionId, notifyEndQueue);
        order.push_back(2);
        return status;
    }
};

class DLNodeNext : public DLNode {
    std::vector<int>& order;

public:
    DLNodeNext(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager, std::vector<int>& order, std::unordered_map<std::string, std::string> nodeOutputNameAlias = {},
        std::optional<int32_t> demultiplyCount = std::nullopt, std::set<std::string> gatherFromNode = {}) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount, gatherFromNode),
        order(order) {}
    ovms::Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override {
        auto status = DLNode::execute(sessionId, notifyEndQueue);
        order.push_back(3);
        return status;
    }
};

TEST_F(EnsembleFlowTest, OrderOfScheduling) {
    ConstructorEnabledModelManager managerWithDummyModel;
    config.setNireq(1);
    managerWithDummyModel.reloadModelWithVersions(config);

    tensor_map_t inputsInfoTmp;
    inputsInfoTmp[customPipelineInputName] = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfoTmp);

    tensor_map_t outputsInfoTmp;
    outputsInfoTmp[customPipelineOutputName + "_1"] = std::make_shared<ovms::TensorInfo>(customPipelineOutputName + "_1",
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});
    outputsInfoTmp[customPipelineOutputName + "_2"] = std::make_shared<ovms::TensorInfo>(customPipelineOutputName + "_2",
        ovms::Precision::FP32,
        DUMMY_MODEL_SHAPE_META,
        Layout{"NC"});
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfoTmp);

    // DL Nodes
    std::vector<int> order;
    auto node_1 = std::make_unique<DLNodeFirst>("dummy_node_1", dummyModelName, requestedModelVersion, managerWithDummyModel, order);
    auto node_2 = std::make_unique<DLNodeDeferred>("dummy_node_2", dummyModelName, requestedModelVersion, managerWithDummyModel, order);
    auto node_3 = std::make_unique<DLNodeNext>("dummy_node_3", dummyModelName, requestedModelVersion, managerWithDummyModel, order);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *node_1, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});

    pipeline.connect(*node_1, *node_3, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*input_node, *node_2, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});

    pipeline.connect(*node_2, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_1"}});
    pipeline.connect(*node_3, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_2"}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));

    pipeline.push(std::move(node_1));
    pipeline.push(std::move(node_2));
    pipeline.push(std::move(node_3));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    std::vector<int> expectedOrder = {
        1,   // try to schedule node_1 with success
        2,   // try to schedule node_2, defer (with order ticket #1)
        3,   // after node_1 ends, try to run next node (node_3), defer with order ticket #2
        2,   // also try to schedule previously deferred nodes, node_2 gets scheduled with success
        3};  // node_2 ends, try to schedule previously deferred node_3 with success
    int expectedOrderIt = 0;
    int lastValue = 0;
    for (int orderElement : order) {
        if (orderElement != lastValue) {
            EXPECT_EQ(orderElement, expectedOrder[expectedOrderIt]);
            expectedOrderIt++;
        }
        lastValue = orderElement;
    }
    // This fragment above is implemented that way because amount of scheduling retries may differ between different machines
    // depending on the inference time of the dummy model
    /*
         -----O1-----O3----
    O---<                  >----O
         -----O2-----------
    */
}

TEST_F(EnsembleFlowTest, FailInDLNodeSetInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in setInputs)    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}, {"NON_EXISTING_INPUT", "REQUIRED_IN_THEORY_OUTPUT"}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    EXPECT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(EnsembleFlowTest, FailInDLNodeExecuteInputsMissingInput) {
    // Most basic configuration, just process single dummy model request

    // input   dummy(fail in execute)    output
    //  O------->O------->O
    ConstructorEnabledModelManager managerWithDummyModel;
    managerWithDummyModel.reloadModelWithVersions(config);
    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node = std::make_unique<DLNode>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *model_node, {{customPipelineInputName, std::string(DUMMY_MODEL_INPUT_NAME) + "_NON_EXISTING_INPUT_NAME_IN_MODEL"}});
    pipeline.connect(*model_node, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    EXPECT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), ovms::StatusCode::INVALID_MISSING_INPUT);
}

class DLNodeFailInFetch : public DLNode {
public:
    DLNodeFailInFetch(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager) :
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
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto failInFetchNode = std::make_unique<DLNodeFailInFetch>("failInFetch_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *failInFetchNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*failInFetchNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(failInFetchNode));
    pipeline.push(std::move(output_node));

    auto status = pipeline.execute(DEFAULT_TEST_CONTEXT);
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
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto failInFetchNode = std::make_unique<DLNodeFailInFetch>("failInFetch_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto modelNode = std::make_unique<DLNodeFailInFetch>("dummy_node", dummyModelName, requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);

    pipeline.connect(*input_node, *failInFetchNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*input_node, *modelNode, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*failInFetchNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});
    pipeline.connect(*modelNode, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName + "_NOT_IMPORTANT"}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(failInFetchNode));
    pipeline.push(std::move(modelNode));
    pipeline.push(std::move(output_node));

    auto status = pipeline.execute(DEFAULT_TEST_CONTEXT);
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
    dummy_1x10.setBatchSize(std::nullopt);
    dummy_1x20.setBatchSize(std::nullopt);
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

TEST_F(EnsembleFlowTest, PipelineDefinitionPrecisionsNotMatchBetweenDLModelTensorsValidation) {
    ConstructorEnabledModelManager manager;
    ModelConfig dummy_fp32 = config;
    ModelConfig dummy_fp64 = DUMMY_FP64_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_fp32), StatusCode::OK_RELOADED);
    ASSERT_EQ(manager.reloadModelWithVersions(dummy_fp64), StatusCode::OK_RELOADED);

    PipelineFactory factory;

    // Simulate reading from pipeline_config.json
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{customPipelineInputName, customPipelineInputName}}},
        {NodeKind::DL, "dummy_node_fp32", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_fp64", "dummy_fp64", std::nullopt, {{DUMMY_FP64_MODEL_OUTPUT_NAME, DUMMY_FP64_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_fp32"] = {
        {ENTRY_NODE_NAME, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_fp64"] = {
        {"dummy_node_fp32", {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_FP64_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_fp64", {{DUMMY_FP64_MODEL_OUTPUT_NAME, customPipelineOutputName}}}};

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
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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
        ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);

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
    ConstructorEnabledModelManager manager;
    PipelineFactory factory;

    PredictRequest request;
    PredictResponse response;
    std::unique_ptr<Pipeline> pipeline;
    EXPECT_EQ(factory.create(pipeline, "pipeline", &request, &response, manager), StatusCode::PIPELINE_DEFINITION_NAME_MISSING);
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
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
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
                                          ovms::Precision::FP32,
                                          DUMMY_MODEL_SHAPE_META,
                                          Layout{"NC"})},
        {"proto_input_1x5",
            std::make_shared<ovms::TensorInfo>("proto_input_1x5",
                ovms::Precision::FP32,
                ovms::Shape{1, 5},
                Layout{"NC"})}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);

    auto dummy_node_1 = std::make_unique<DLNode>("dummy_node_1", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto dummy_node_2 = std::make_unique<DLNode>("dummy_node_2", dummyModelName, requestedModelVersion, managerWithDummyModel);
    auto dummy_node_3 = std::make_unique<DLNode>("dummy_node_3", dummyModelName, requestedModelVersion, managerWithDummyModel);

    Pipeline pipeline(*input_node, *output_node, *this->reporter);
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

    EXPECT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
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
    pipelineBeforeRetire->execute(DEFAULT_TEST_CONTEXT);
    uint dummySeriallyConnectedCount = 1;
    checkDummyResponse(dummySeriallyConnectedCount);
}

TEST_F(EnsembleFlowTest, RuntimeWrongBatchSizeArbitraryPosition) {
    ConstructorEnabledModelManager managerWithDummyModel;

    ModelConfig configCN = DUMMY_MODEL_CONFIG;
    configCN.setName("dummy_C1_N10");
    configCN.setBatchingParams("0");
    configCN.parseShapeParameter("(1,10)");
    ASSERT_EQ(configCN.parseLayoutParameter("cn"), StatusCode::OK);
    managerWithDummyModel.reloadModelWithVersions(configCN);

    configCN = DUMMY_MODEL_CONFIG;
    configCN.setName("dummy_C1_N15");
    configCN.setBatchingParams("0");
    configCN.parseShapeParameter("(1,15)");
    ASSERT_EQ(configCN.parseLayoutParameter("cn"), StatusCode::OK);
    managerWithDummyModel.reloadModelWithVersions(configCN);

    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        Shape{1, 15},
        Layout{"CN"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        Shape{1, 10},
        Layout{"CN"});

    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node_1 = std::make_unique<DLNode>("dummy_node_1", "dummy_C1_N10", requestedModelVersion, managerWithDummyModel);
    auto model_node_2 = std::make_unique<DLNode>("dummy_node_2", "dummy_C1_N15", requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);
    Pipeline pipeline(*input_node, *output_node, *this->reporter);
    pipeline.connect(*input_node, *model_node_1, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_1, *model_node_2, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_2, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node_1));
    pipeline.push(std::move(model_node_2));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(EnsembleFlowTest, RuntimeWrongShapeArbitraryBatchPosition) {
    ConstructorEnabledModelManager managerWithDummyModel;

    ModelConfig configCN = DUMMY_MODEL_CONFIG;
    configCN.setName("dummy_C1_N10");
    configCN.setBatchingParams("0");
    configCN.parseShapeParameter("(1,10)");
    ASSERT_EQ(configCN.parseLayoutParameter("cn"), StatusCode::OK);
    managerWithDummyModel.reloadModelWithVersions(configCN);

    configCN = DUMMY_MODEL_CONFIG;
    configCN.setName("dummy_C2_N10");
    configCN.setBatchingParams("0");
    configCN.parseShapeParameter("(2,10)");
    ASSERT_EQ(configCN.parseLayoutParameter("cn"), StatusCode::OK);
    managerWithDummyModel.reloadModelWithVersions(configCN);

    dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineOutputName,
        ovms::Precision::FP32,
        Shape{2, 10},
        Layout{"CN"});
    dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(customPipelineInputName,
        ovms::Precision::FP32,
        Shape{1, 10},
        Layout{"CN"});

    // Configure pipeline
    const tensor_map_t inputsInfo{{customPipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode<PredictRequest>>(&request, inputsInfo);
    auto model_node_1 = std::make_unique<DLNode>("dummy_node_1", "dummy_C1_N10", requestedModelVersion, managerWithDummyModel);
    auto model_node_2 = std::make_unique<DLNode>("dummy_node_2", "dummy_C2_N10", requestedModelVersion, managerWithDummyModel);
    const tensor_map_t outputsInfo{{customPipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode<PredictResponse>>(&response, outputsInfo);
    Pipeline pipeline(*input_node, *output_node, *this->reporter);
    pipeline.connect(*input_node, *model_node_1, {{customPipelineInputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_1, *model_node_2, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node_2, *output_node, {{DUMMY_MODEL_OUTPUT_NAME, customPipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(model_node_1));
    pipeline.push(std::move(model_node_2));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
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
    pipelineBeforeRetire->execute(DEFAULT_TEST_CONTEXT);
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

TEST_F(EnsembleFlowTest, EnablingStatefulParametersForModelUsedInPipeline) {
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

static const char* pipelineDemultiplexerShapeNotEqualToDemultiplyCount = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(3, 2, 10) ",
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

TYPED_TEST(EnsembleFlowBothApiTest, DemultiplexerMultipleBatchSizeWithShapeNotEqualToDemultiplyCountNotAllowed) {
    std::string fileToReload = this->directoryPath + "/config.json";
    createConfigFileWithContent(pipelineDemultiplexerShapeNotEqualToDemultiplyCount, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_TENSOR_SHARD_COUNT);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

static const char* pipelineInnerNodeConnectionShapeRangeNotMatch = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy_A",
                "base_path": "/ovms/src/test/dummy",
                "shape": "(-1,30:40) ",
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "dummy_B",
                "base_path": "/ovms/src/test/dummy",
                "shape": "(-1,41:60) ",
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
                    "name": "dummyNode_A",
                    "model_name": "dummy_A",
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
                    "name": "dummyNode_B",
                    "model_name": "dummy_B",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "dummyNode_A",
                               "data_item": "new_dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode_B",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, InnerNodeConnectionShapeRangeNotMatch) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineInnerNodeConnectionShapeRangeNotMatch, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::INVALID_SHAPE) << status.string();

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

static const char* pipelineInnerNodeConnectionShapeRangePartiallyMatch = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy_A",
                "base_path": "/ovms/src/test/dummy",
                "shape": "(-1,30:40) ",
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "dummy_B",
                "base_path": "/ovms/src/test/dummy",
                "shape": "(-1,40:60) ",
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
                    "name": "dummyNode_A",
                    "model_name": "dummy_A",
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
                    "name": "dummyNode_B",
                    "model_name": "dummy_B",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "dummyNode_A",
                               "data_item": "new_dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode_B",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTest, InnerNodeConnectionShapeRangePartiallyMatch) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineInnerNodeConnectionShapeRangePartiallyMatch, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
}

static const char* pipelineDemultiplexerShapeEqualToDemultiplyCount = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(2, 2, 10) ",
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

TYPED_TEST(EnsembleFlowBothApiTest, DemultiplexerMultipleBatchSizeWithShapeEqualToDemultiplyCountAllowed) {
    std::string fileToReload = this->directoryPath + "/config.json";
    createConfigFileWithContent(pipelineDemultiplexerShapeEqualToDemultiplyCount, fileToReload);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);

    ASSERT_EQ(manager.getPipelineFactory().findDefinitionByName(PIPELINE_1_DUMMY_NAME)->getStateCode(),
        PipelineDefinitionStateCode::AVAILABLE);
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
                "shape": "(1,1,2,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {2.0, 5.0, 3.0, 6.0, 4.0, 7.0}, response, {1, 3, 1, 2});
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
                "shape": "(1,1,2,3) ",
                "layout": "nhwc:nchw",
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

TYPED_TEST(EnsembleFlowBothApiTest, ExecuteSingleIncrement4DimInputNHWCDynamicBatch) {
    std::string fileToReload = this->directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimInputNHWCDynamicBatch, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    this->prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
        this->request, "pipeline_input", {2, 1, 1, 2, 3});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &this->request, &this->response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>(std::string{"pipeline_output"}, {2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 11.0, 41.0, 21.0, 51.0, 31.0, 61.0}, this->response, {2, 1, 3, 1, 2});
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
                "layout": {"output": "nhwc:nchw"},
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {2.0, 4.0, 6.0, 3.0, 5.0, 7.0}, response, {1, 1, 2, 3});
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
                "layout": {"output": "nhwc:nchw"},
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

TYPED_TEST(EnsembleFlowBothApiTest, ExecuteSingleIncrement4DimOutputNHWCDynamicBatch) {
    std::string fileToReload = this->directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCDynamicBatch, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    this->prepareRequest({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                             10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
        this->request, "pipeline_input", {2, 1, 3, 1, 2});

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &this->request, &this->response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 11.0, 31.0, 51, 21.0, 41.0, 61.0}, this->response, {2, 1, 1, 2, 3});
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
                "shape": "(1,1,2,3) ",
                "layout": {"input": "nhwc:nchw"},
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
                "layout": {"output": "nhwc:nchw"},
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "increment_nhwc_in_nchw_out",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1,2,3) ",
                "layout": {"input": "nhwc:nchw"},
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, response, {1, 3, 1, 2});
}

class EnsembleFlowTestBinaryInput : public EnsembleFlowTest {
public:
    const std::string imagePath = "/ovms/src/test/binaryutils/rgb.jpg";
    const std::string imagePath2x2 = "/ovms/src/test/binaryutils/rgb2x2.jpg";
    const std::string imagePath4x4 = "/ovms/src/test/binaryutils/rgb4x4.jpg";
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
                "shape": "(1,1,1,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 28.0, 238.0}, response, {1, 3, 1, 1});
}

static const char* pipelineWith4DimDummyFP64 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/dummy_fp64",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1,1,3) ",
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
                        {"input:0": {"node_name": "request",
                                   "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output:0",
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

TEST_F(EnsembleFlowTestBinaryInput, DoublePrecision) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWith4DimDummyFP64, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareBinaryRequest(imagePath, request, "pipeline_input");

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<double>("pipeline_output", {37.0, 28.0, 238.0}, response, {1, 1, 1, 3});
}

static const char* pipelineSingleIncrement4DimOutputNHWC1x1BatchAny = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(-1,1,1,3) ",
                "layout": "nhwc:nchw",
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

TEST_F(EnsembleFlowTestBinaryInput, BatchSizeAny) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1BatchAny, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    const size_t batchSize = 100;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimShape("pipeline_output", response, {batchSize, 3, 1, 1});
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
    auto status = pipeline->execute(DEFAULT_TEST_CONTEXT);
    ASSERT_EQ(status, StatusCode::INVALID_NO_OF_CHANNELS) << status.string();
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
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {1.0}, response, {1, 1, 1, 1});
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
                "shape": "(5,1,1,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, response, {5, 3, 1, 1});
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
                "shape": "(1,2,2,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0}, response, {1, 3, 2, 2});
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
                "shape": "(5,2,2,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0, 37.0, 37.0, 37.0, 37.0, 28.0, 28.0, 28.0, 28.0, 238.0, 238.0, 238.0, 238.0}, response, {5, 3, 2, 2});
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
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(EnsembleFlowTestBinaryInput, InvalidData) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWC1x1, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    request.Clear();
    tensorflow::TensorProto& inputProto = (*request.mutable_inputs())["pipeline_input"];
    inputProto.set_dtype(tensorflow::DataType::DT_STRING);
    inputProto.add_string_val("INVALID_IMAGE");
    inputProto.mutable_tensor_shape()->add_dim()->set_size(1);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::IMAGE_PARSING_FAILED);
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
                "shape": "(1,1,1,3) ",
                "layout": "nhwc:nchw",
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

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, response, {5, 1, 3, 1, 1});
}

static const char* pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryStaticDemultiplexer = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1:3,1:3,3) ",
                "layout": "nhwc:nchw",
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "increment_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 5,
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

TEST_F(EnsembleFlowTestBinaryInput, EntryStaticDemultiplexerResolutionMatches) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryStaticDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, response, {5, 1, 3, 1, 1});
}

TEST_F(EnsembleFlowTestBinaryInput, EntryStaticDemultiplexerResolutionAutoAlign) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryStaticDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath4x4, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimShape("pipeline_output", response, {5, 1, 3, 3, 3});
}

static const char* pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryDynamicDemultiplexer = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1:3,1:3,3) ",
                "layout": "nhwc:nchw",
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

TEST_F(EnsembleFlowTestBinaryInput, EntryDynamicDemultiplexerResolutionMatches) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryDynamicDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0, 37.0, 28.0, 238.0}, response, {5, 1, 3, 1, 1});
}

TEST_F(EnsembleFlowTestBinaryInput, EntryDynamicDemultiplexerResolutionResolutionMismatch) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineSingleIncrement4DimOutputNHWCRangeResolutionEntryDynamicDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 5;
    prepareBinaryRequest(imagePath4x4, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "increment_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);
}

static const char* pipelineWithOnlyDynamicCustomNode = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_image",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_image.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_image",
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

// This test ensure binary inputs work for pipelines with layout ANY.
// Such pipelines have only custom nodes as entry nodes.
// In this case we do not reject the request but create NHWC content out of that.
TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANY_RequestBS1) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNode, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 1;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {44.0, 35.0, 245.0}, response, {1, 1, 1, 3});
}

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANY_RequestBS2) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNode, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 2;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {44.0, 35.0, 245.0, 44.0, 35.0, 245.0}, response, {2, 1, 1, 3});
}

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANY_RequestMisaligned) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNode, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareMisalignedBinaryImageRequest(imagePath, imagePath2x2, request, "pipeline_input");
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH);
}

TEST_F(EnsembleFlowTest, TensorContentInputWithPipelineInputLayoutANY_RequestNhwc) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNode, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareRequest({1.0, 2.0, 3.0, 4.0}, request, "pipeline_input", {1, 4, 1});  // should be [1, 4, 1, 1]
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

static const char* pipelineWithOnlyDynamicCustomNodeAndDemultiplexer = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_image",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_image.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_image",
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANYAndDemultiplexer_RequestBS1) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNodeAndDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 1;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {44.0, 35.0, 245.0}, response, {1, 1, 1, 1, 3});
}

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANYAndDemultiplexer_RequestBS2) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNodeAndDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 2;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {44.0, 35.0, 245.0, 44.0, 35.0, 245.0}, response, {2, 1, 1, 1, 3});
}

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANYAndDemultiplexer_RequestMisaligned) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNodeAndDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    prepareMisalignedBinaryImageRequest(imagePath2x2, imagePath, request, "pipeline_input");
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH);
}

TYPED_TEST(EnsembleFlowBothApiTest, TensorContentInputWithPipelineInputLayoutANYAndDemultiplexer_RequestNhwc) {
    std::string fileToReload = this->directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithOnlyDynamicCustomNodeAndDemultiplexer, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    this->prepareRequest({1.0, 2.0, 3.0, 4.0}, this->request, "pipeline_input", {1, 1, 4, 1});  // should be [1, 1, 4, 1, 1]
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &this->request, &this->response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
}

static const char* pipelineWithDynamicCustomNodeDemultiplexerAndDynamicResolutionModel = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,-1,-1,3) ",
                "layout": "nhwc:nchw",
                "nireq": 1
            }
        }
    ],
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_image",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_image.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_image",
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "custom_node",
                                   "data_item": "custom_node_output"}}
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

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANYCustomNodeDemultiplexerAndDynamicResolutionModel) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithDynamicCustomNodeDemultiplexerAndDynamicResolutionModel, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    int batchSize = 1;
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {45.0, 36.0, 246.0}, response, {1, 1, 3, 1, 1});
}

static const char* pipelineWithDynamicCustomNodeDemultiplexerAndRangeOfResolutionModel = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "increment",
                "base_path": "/ovms/src/test/increment_1x3x4x5",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "shape": "(1,1:2,1:2,3) ",
                "layout": "nhwc:nchw",
                "nireq": 1
            }
        }
    ],
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_image",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_image.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_image",
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "increment_node",
                    "model_name": "increment",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "custom_node",
                                   "data_item": "custom_node_output"}}
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

TEST_F(EnsembleFlowTestBinaryInput, BinaryInputWithPipelineInputLayoutANYCustomNodeDemultiplexerAndRangeOfResolutionModel) {
    std::string fileToReload = directoryPath + "/config.json";
    createConfigFileWithContent(pipelineWithDynamicCustomNodeDemultiplexerAndRangeOfResolutionModel, fileToReload);
    ConstructorEnabledModelManager manager;
    std::unique_ptr<Pipeline> pipeline;

    // Try with resolution out of shape range, expect INVALID_SHAPE
    int batchSize = 1;
    prepareBinaryRequest(imagePath4x4, request, "pipeline_input", batchSize);

    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::INVALID_SHAPE);

    request.Clear();
    response.Clear();

    // Try with resolution matching the shape, expect OK
    ASSERT_EQ(manager.getPipelineFactory().create(pipeline, "my_pipeline", &request, &response, manager), StatusCode::OK);
    prepareBinaryRequest(imagePath, request, "pipeline_input", batchSize);

    ASSERT_EQ(pipeline->execute(DEFAULT_TEST_CONTEXT), StatusCode::OK);
    checkIncrement4DimResponse<float>("pipeline_output", {45.0, 36.0, 246.0}, response, {1, 1, 3, 1, 1});
}
