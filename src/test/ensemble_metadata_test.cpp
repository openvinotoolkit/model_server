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

#include "../pipeline_factory.hpp"
#include "test_utils.hpp"

using namespace ovms;

TEST(EnsembleMetadata, OneNode) {
    /*
        This test creates pipeline definition with one DL model node.
        Test ensures we receive correct metadata - one input for the DL model node.
    */

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "response"},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {"request", {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["response"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), shape_t({1, DUMMY_MODEL_INPUT_SIZE}));
    EXPECT_EQ(input->getPrecision(), InferenceEngine::Precision::FP32);
}

TEST(EnsembleMetadata, MultipleNodesOnDifferentLevelsUsingTheSamePipelineInputs) {
    /*
        This test creates pipeline definition with multiple connections refering to entry node.
        This way we use the same input tensors in nodes on different levels of depth.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo does not return duplicated information.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig increment_model_config = DUMMY_MODEL_CONFIG;
    increment_model_config.setName("increment");
    ASSERT_EQ(manager.reloadModelWithVersions(increment_model_config), StatusCode::OK);

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "N1", "increment"},
        {NodeKind::DL, "N2", "sum"},
        {NodeKind::EXIT, "response"},
    };

    pipeline_connections_t connections;

    const std::string& INCREMENT_MODEL_INPUT_NAME = DUMMY_MODEL_INPUT_NAME;
    const std::string& INCREMENT_MODEL_OUTPUT_NAME = DUMMY_MODEL_OUTPUT_NAME;
    const int INCREMENT_MODEL_INPUT_SIZE = DUMMY_MODEL_INPUT_SIZE;

    connections["N1"] = {
        {"request", {{"request_input_for_N1", INCREMENT_MODEL_INPUT_NAME}}}};

    connections["N2"] = {
        {"request", {{"request_input_for_N2_and_exit", SUM_MODEL_INPUT_NAME_1}}},
        {"N1", {{INCREMENT_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_2}}}};

    connections["response"] = {
        {"N1", {{INCREMENT_MODEL_OUTPUT_NAME, "intermediate_result_from_increment"}}},
        {"N2", {{SUM_MODEL_OUTPUT_NAME, "intermediate_result_from_sum"}}},
        {"request", {{"request_input_for_N2_and_exit", "original_input_for_N2"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_for_N1"), inputs.end());
    ASSERT_NE(inputs.find("request_input_for_N2_and_exit"), inputs.end());

    const auto& request_input_for_N1 = inputs.at("request_input_for_N1");
    EXPECT_EQ(request_input_for_N1->getShape(), shape_t({1, INCREMENT_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N1->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& request_input_for_N2_and_exit = inputs.at("request_input_for_N2_and_exit");
    EXPECT_EQ(request_input_for_N2_and_exit->getShape(), shape_t({1, SUM_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N2_and_exit->getPrecision(), InferenceEngine::Precision::FP32);
}

TEST(EnsembleMetadata, EmptyPipelineReturnsCorrectInputInfo) {
    /*
        This test creates pipeline definition with no DL model nodes.
        The only thing that pipeline does is passing request data into response with no inferences in between.
        Test ensures such pipeline will return named input information but with undefined shape and precision.
    */
    ConstructorEnabledModelManager manager;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::EXIT, "response"},
    };

    pipeline_connections_t connections;

    connections["response"] = {
        {"request", {{"name_from_entry", "name_for_response"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_NE(inputs.find("name_from_entry"), inputs.end());

    const auto& name_from_entry = inputs.at("name_from_entry");
    EXPECT_EQ(name_from_entry->getShape(), shape_t({}));
    EXPECT_EQ(name_from_entry->getPrecision(), InferenceEngine::Precision::UNSPECIFIED);
}

TEST(EnsembleMetadata, ParallelDLModelNodesReferingToManyPipelineInputs) {
    /*
        This test creates pipeline definition with 4 parallel DL model nodes, all refering to entry node.
        All nodes require 2 inputs, all from different sources, so in total we need 8 different pipeline inputs.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo contains all 8 required inputs.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "sum_node_quarter_1", "sum"},
        {NodeKind::DL, "sum_node_quarter_2", "sum"},
        {NodeKind::DL, "sum_node_quarter_3", "sum"},
        {NodeKind::DL, "sum_node_quarter_4", "sum"},
        {NodeKind::DL, "sum_node_semi_1", "sum"},
        {NodeKind::DL, "sum_node_semi_2", "sum"},
        {NodeKind::DL, "sum_node_final_1", "sum"},
        {NodeKind::EXIT, "response"},
    };

    pipeline_connections_t connections;

    // Quarter
    connections["sum_node_quarter_1"] = {
        {"request", {{"request_input_for_quarter_1_a", SUM_MODEL_INPUT_NAME_1},
                        {"request_input_for_quarter_1_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_2"] = {
        {"request", {{"request_input_for_quarter_2_a", SUM_MODEL_INPUT_NAME_1},
                        {"request_input_for_quarter_2_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_3"] = {
        {"request", {{"request_input_for_quarter_3_a", SUM_MODEL_INPUT_NAME_1},
                        {"request_input_for_quarter_3_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_4"] = {
        {"request", {{"request_input_for_quarter_4_a", SUM_MODEL_INPUT_NAME_1},
                        {"request_input_for_quarter_4_b", SUM_MODEL_INPUT_NAME_2}}}};

    // Semi
    connections["sum_node_semi_1"] = {
        {"sum_node_quarter_1", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_1}}},
        {"sum_node_quarter_2", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_semi_2"] = {
        {"sum_node_quarter_3", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_1}}},
        {"sum_node_quarter_4", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_2}}}};

    // Final
    connections["sum_node_final_1"] = {
        {"sum_node_semi_1", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_1}}},
        {"sum_node_semi_2", {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_2}}}};

    connections["response"] = {
        {"sum_node_final_1", {{SUM_MODEL_OUTPUT_NAME, "final_sum"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 8);
    for (size_t i = 1; i <= 4; i++) {
        const std::string name_a = "request_input_for_quarter_" + std::to_string(i) + "_a";
        const std::string name_b = "request_input_for_quarter_" + std::to_string(i) + "_b";

        ASSERT_NE(inputs.find(name_a), inputs.end());
        ASSERT_NE(inputs.find(name_b), inputs.end());

        EXPECT_EQ(inputs.find(name_a)->second->getShape(), shape_t({1, SUM_MODEL_INPUT_SIZE}));
        EXPECT_EQ(inputs.find(name_a)->second->getPrecision(), InferenceEngine::Precision::FP32);
        EXPECT_EQ(inputs.find(name_b)->second->getShape(), shape_t({1, SUM_MODEL_INPUT_SIZE}));
        EXPECT_EQ(inputs.find(name_b)->second->getPrecision(), InferenceEngine::Precision::FP32);
    }
}

TEST(EnsembleMetadata, OneUnavailableNode) {
    /*
        This test creates pipeline definition with one DL model node which has model that is unavailable due to:
            a) no model version available
            b) model version is retired
        Test ensures we receive error status by calling getInputsInfo.
    */

    const model_version_t UNAVAILABLE_DUMMY_VERSION = 99;

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"},
        {NodeKind::DL, "dummy_node", "dummy"},
        {NodeKind::EXIT, "response"},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {"request", {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["response"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    config.setModelVersionPolicy(std::make_shared<SpecificModelVersionPolicy>(model_versions_t{UNAVAILABLE_DUMMY_VERSION}));
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    tensor_map_t inputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::MODEL_MISSING);

    config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    auto instance = manager.findModelInstance("dummy", 0);
    ASSERT_NE(instance, nullptr);
    instance->unloadModel();

    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}
