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
#include <filesystem>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../modelconfig.hpp"
#include "../modelmanager.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "test_utils.hpp"

using namespace ovms;

using ::testing::ElementsAre;

/*
 * Scenario with pipelines with underlying DL models loaded with mapping_config.json.
 * This set of tests ensures model input/output mapping is respected by pipeline execution.
 */

class PipelineWithInputOutputNameMappedModel : public ::testing::Test {
protected:
    void SetUp() override {
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

        const std::string directoryName = std::string(test_info->test_suite_name());
        directoryPath = "/tmp" + directoryName;
        configPath = directoryPath + "/config.json";
        modelPath = directoryPath + "/dummy";
        mappingConfigPath = modelPath + "/1/mapping_config.json";

        // Copy dummy model to temporary destination
        std::filesystem::remove_all(directoryPath);
        std::filesystem::create_directories(directoryPath);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    }

    void TearDown() override {
        // Clean up temporary destination
        std::filesystem::remove_all(directoryPath);
    }

    std::string directoryPath;
    std::string configPath;
    std::string modelPath;
    std::string mappingConfigPath;

    ConstructorEnabledModelManager managerWithDummyModel;
};

TEST_F(PipelineWithInputOutputNameMappedModel, SuccessfullyReferToMappedNamesAndExecute) {
    // Create mapping config for model
    createConfigFileWithContent(R"({
        "inputs": {"b": "input_tensor"},
        "outputs": {"a": "output_tensor"}
    })",
        mappingConfigPath);

    // Load models
    auto modelConfig = DUMMY_MODEL_CONFIG;
    modelConfig.setBasePath(modelPath);
    ASSERT_EQ(managerWithDummyModel.reloadModelWithVersions(modelConfig), StatusCode::OK);

    // Create pipeline definition
    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "dummyA", "dummy"},
        {NodeKind::DL, "dummyB", "dummy"},
        {NodeKind::EXIT, "response"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    connections["dummyA"] = {
        {"request", {{"vector", "input_tensor"}}}};
    connections["dummyB"] = {
        {"dummyA", {{"output_tensor", "input_tensor"}}}};
    connections["response"] = {
        {"dummyB", {{"output_tensor", "response_tensor_name"}}}};

    // Ensure definition created without errors
    ASSERT_EQ(factory.createDefinition("pipeline", info, connections, managerWithDummyModel), StatusCode::OK);

    // Prepare request
    std::unique_ptr<Pipeline> pipeline;
    tensorflow::serving::PredictRequest request;
    tensorflow::serving::PredictResponse response;
    auto& input_proto = (*request.mutable_inputs())["vector"];
    input_proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    input_proto.mutable_tensor_shape()->add_dim()->set_size(1);
    input_proto.mutable_tensor_shape()->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);

    std::vector<float> input_data(DUMMY_MODEL_INPUT_SIZE);
    std::iota(input_data.begin(), input_data.end(), 1);  // 1, 2, 3, ..., 10
    input_proto.mutable_tensor_content()->assign((char*)input_data.data(), sizeof(float) * DUMMY_MODEL_INPUT_SIZE);

    // Execute pipeline
    factory.create(pipeline, "pipeline", &request, &response, managerWithDummyModel);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    // Compare response
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.outputs().count("response_tensor_name"), 1);
    const auto& output_proto = response.outputs().at("response_tensor_name");
    ASSERT_EQ(output_proto.dtype(), tensorflow::DataType::DT_FLOAT);
    ASSERT_THAT(asVector(output_proto.tensor_shape()), ElementsAre(1, 10));

    std::vector<float> output_data(DUMMY_MODEL_INPUT_SIZE);
    std::transform(input_data.begin(), input_data.end(), output_data.begin(), [](float& v) { return v + 2.0; });  // 3, 4, ... 12
    ASSERT_EQ(asVector<float>(output_proto.tensor_content()), output_data);
}

TEST_F(PipelineWithInputOutputNameMappedModel, ReferingToOriginalInputNameFailsCreation) {
    // Create mapping config for model
    createConfigFileWithContent(R"({
        "inputs": {"b": "input_tensor"},
        "outputs": {"a": "output_tensor"}
    })",
        mappingConfigPath);

    // Load models
    auto modelConfig = DUMMY_MODEL_CONFIG;
    modelConfig.setBasePath(modelPath);
    ASSERT_EQ(managerWithDummyModel.reloadModelWithVersions(modelConfig), StatusCode::OK);

    // Create pipeline definition
    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "dummyA", "dummy"},
        {NodeKind::DL, "dummyB", "dummy"},
        {NodeKind::EXIT, "response"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    connections["dummyA"] = {
        {"request", {{"vector", "b"}}}};
    connections["dummyB"] = {
        {"dummyA", {{"output_tensor", "b"}}}};
    connections["response"] = {
        {"dummyB", {{"output_tensor", "response_tensor_name"}}}};

    EXPECT_EQ(factory.createDefinition("pipeline", info, connections, managerWithDummyModel), StatusCode::INVALID_MISSING_INPUT);
}

TEST_F(PipelineWithInputOutputNameMappedModel, ReferingToOriginalOutputNameFailsCreation) {
    // Create mapping config for model
    createConfigFileWithContent(R"({
        "inputs": {"b": "input_tensor"},
        "outputs": {"a": "output_tensor"}
    })",
        mappingConfigPath);

    // Load models
    auto modelConfig = DUMMY_MODEL_CONFIG;
    modelConfig.setBasePath(modelPath);
    ASSERT_EQ(managerWithDummyModel.reloadModelWithVersions(modelConfig), StatusCode::OK);

    // Create pipeline definition
    PipelineFactory factory;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "dummyA", "dummy"},
        {NodeKind::DL, "dummyB", "dummy"},
        {NodeKind::EXIT, "response"},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

    connections["dummyA"] = {
        {"request", {{"vector", "input_tensor"}}}};
    connections["dummyB"] = {
        {"dummyA", {{"a", "input_tensor"}}}};
    connections["response"] = {
        {"dummyB", {{"a", "response_tensor_name"}}}};

    EXPECT_EQ(factory.createDefinition("pipeline", info, connections, managerWithDummyModel), StatusCode::INVALID_MISSING_OUTPUT);
}
