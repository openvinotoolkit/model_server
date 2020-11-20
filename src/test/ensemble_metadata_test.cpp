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
        Test ensures we receive correct metadata - one input and one output for the DL model node.
    */

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs, outputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);
    ASSERT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), shape_t({1, DUMMY_MODEL_INPUT_SIZE}));
    EXPECT_EQ(input->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& output = outputs.at("request_output_name");
    EXPECT_EQ(output->getShape(), shape_t({1, DUMMY_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(output->getPrecision(), InferenceEngine::Precision::FP32);
}

TEST(EnsembleMetadata, MultipleNodesOnDifferentLevelsUsingTheSamePipelineInputs) {
    /*
        This test creates pipeline definition with multiple connections refering to entry node.
        This way we use the same input tensors in nodes on different levels of depth.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo/getOutptusInfo does not return duplicated information.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig increment_model_config = DUMMY_MODEL_CONFIG;
    increment_model_config.setName("increment");
    ASSERT_EQ(manager.reloadModelWithVersions(increment_model_config), StatusCode::OK);

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK);

    const std::string& INCREMENT_MODEL_INPUT_NAME = DUMMY_MODEL_INPUT_NAME;
    const std::string& INCREMENT_MODEL_OUTPUT_NAME = DUMMY_MODEL_OUTPUT_NAME;
    const int INCREMENT_MODEL_INPUT_SIZE = DUMMY_MODEL_INPUT_SIZE;
    const int INCREMENT_MODEL_OUTPUT_SIZE = DUMMY_MODEL_OUTPUT_SIZE;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {
                                                                 {"request_input_for_N1", "request_input_for_N1"},
                                                                 {"request_input_for_N2_and_exit", "request_input_for_N2_and_exit"},
                                                             }},
        {NodeKind::DL, "N1", "increment", std::nullopt, {{INCREMENT_MODEL_OUTPUT_NAME, INCREMENT_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "N2", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["N1"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_N1", INCREMENT_MODEL_INPUT_NAME}}}};

    connections["N2"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_N2_and_exit", SUM_MODEL_INPUT_NAME_1}}},
        {"N1", {{INCREMENT_MODEL_OUTPUT_NAME, SUM_MODEL_INPUT_NAME_2}}}};

    connections[EXIT_NODE_NAME] = {
        {"N1", {{INCREMENT_MODEL_OUTPUT_NAME, "intermediate_result_from_increment"}}},
        {"N2", {{SUM_MODEL_OUTPUT_NAME, "intermediate_result_from_sum"}}},
        {ENTRY_NODE_NAME, {{"request_input_for_N2_and_exit", "original_input_for_N2"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs, outputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);
    ASSERT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(outputs.size(), 3);
    ASSERT_NE(inputs.find("request_input_for_N1"), inputs.end());
    ASSERT_NE(inputs.find("request_input_for_N2_and_exit"), inputs.end());
    ASSERT_NE(outputs.find("intermediate_result_from_increment"), outputs.end());
    ASSERT_NE(outputs.find("intermediate_result_from_sum"), outputs.end());
    ASSERT_NE(outputs.find("original_input_for_N2"), outputs.end());

    const auto& request_input_for_N1 = inputs.at("request_input_for_N1");
    EXPECT_EQ(request_input_for_N1->getShape(), shape_t({1, INCREMENT_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N1->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& request_input_for_N2_and_exit = inputs.at("request_input_for_N2_and_exit");
    EXPECT_EQ(request_input_for_N2_and_exit->getShape(), shape_t({1, SUM_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N2_and_exit->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& intermediate_result_from_increment = outputs.at("intermediate_result_from_increment");
    EXPECT_EQ(intermediate_result_from_increment->getShape(), shape_t({1, INCREMENT_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(intermediate_result_from_increment->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& intermediate_result_from_sum = outputs.at("intermediate_result_from_sum");
    EXPECT_EQ(intermediate_result_from_sum->getShape(), shape_t({1, SUM_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(intermediate_result_from_sum->getPrecision(), InferenceEngine::Precision::FP32);

    const auto& original_input_for_N2 = outputs.at("original_input_for_N2");
    EXPECT_EQ(original_input_for_N2->getShape(), shape_t({}));
    EXPECT_EQ(original_input_for_N2->getPrecision(), InferenceEngine::Precision::UNSPECIFIED);
}

TEST(EnsembleMetadata, EmptyPipelineReturnsCorrectInputAndOutputInfo) {
    /*
        This test creates pipeline definition with no DL model nodes.
        The only thing that pipeline does is passing request data into response with no inferences in between.
        Test ensures such pipeline will return named input and output information but with undefined shape and precision.
    */
    ConstructorEnabledModelManager manager;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"name_from_entry", "name_from_entry"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections[EXIT_NODE_NAME] = {
        {ENTRY_NODE_NAME, {{"name_from_entry", "name_for_response"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs, outputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);
    ASSERT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::OK);

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("name_from_entry"), inputs.end());
    ASSERT_NE(outputs.find("name_for_response"), outputs.end());

    const auto& name_from_entry = inputs.at("name_from_entry");
    EXPECT_EQ(name_from_entry->getShape(), shape_t({}));
    EXPECT_EQ(name_from_entry->getPrecision(), InferenceEngine::Precision::UNSPECIFIED);

    const auto& name_for_response = outputs.at("name_for_response");
    EXPECT_EQ(name_for_response->getShape(), shape_t({}));
    EXPECT_EQ(name_for_response->getPrecision(), InferenceEngine::Precision::UNSPECIFIED);
}

TEST(EnsembleMetadata, ParallelDLModelNodesReferingToManyPipelineInputs) {
    /*
        This test creates pipeline definition with 4 parallel DL model nodes, all refering to entry node.
        All nodes require 2 inputs, all from different sources, so in total we need 8 different pipeline inputs.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo contains all 8 required inputs and one output.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {
                                                                 {"request_input_for_quarter_1_a", "request_input_for_quarter_1_a"},
                                                                 {"request_input_for_quarter_1_b", "request_input_for_quarter_1_b"},
                                                                 {"request_input_for_quarter_2_a", "request_input_for_quarter_2_a"},
                                                                 {"request_input_for_quarter_2_b", "request_input_for_quarter_2_b"},
                                                                 {"request_input_for_quarter_3_a", "request_input_for_quarter_3_a"},
                                                                 {"request_input_for_quarter_3_b", "request_input_for_quarter_3_b"},
                                                                 {"request_input_for_quarter_4_a", "request_input_for_quarter_4_a"},
                                                                 {"request_input_for_quarter_4_b", "request_input_for_quarter_4_b"},
                                                             }},
        {NodeKind::DL, "sum_node_quarter_1", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_quarter_2", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_quarter_3", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_quarter_4", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_semi_1", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_semi_2", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "sum_node_final_1", "sum", std::nullopt, {{SUM_MODEL_OUTPUT_NAME, SUM_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // Quarter
    connections["sum_node_quarter_1"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_quarter_1_a", SUM_MODEL_INPUT_NAME_1},
                              {"request_input_for_quarter_1_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_2"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_quarter_2_a", SUM_MODEL_INPUT_NAME_1},
                              {"request_input_for_quarter_2_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_3"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_quarter_3_a", SUM_MODEL_INPUT_NAME_1},
                              {"request_input_for_quarter_3_b", SUM_MODEL_INPUT_NAME_2}}}};

    connections["sum_node_quarter_4"] = {
        {ENTRY_NODE_NAME, {{"request_input_for_quarter_4_a", SUM_MODEL_INPUT_NAME_1},
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

    connections[EXIT_NODE_NAME] = {
        {"sum_node_final_1", {{SUM_MODEL_OUTPUT_NAME, "final_sum"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    tensor_map_t inputs, outputs;
    ASSERT_EQ(def->getInputsInfo(inputs, manager), StatusCode::OK);
    ASSERT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::OK);

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

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(outputs.find("final_sum"), outputs.end());
    EXPECT_EQ(outputs.find("final_sum")->second->getShape(), shape_t({1, SUM_MODEL_INPUT_SIZE}));
    EXPECT_EQ(outputs.find("final_sum")->second->getPrecision(), InferenceEngine::Precision::FP32);
}

TEST(EnsembleMetadata, OneUnavailableNode) {
    /*
        This test creates pipeline definition with one DL model node which has model that is unavailable due to:
            a) no model version available
            b) model version is retired
            c) model is not loaded yet
        Test ensures we receive error status by calling getInputsInfo and getOutputsInfo.
    */

    const model_version_t UNAVAILABLE_DUMMY_VERSION = 99;

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);

    config.setModelVersionPolicy(std::make_shared<SpecificModelVersionPolicy>(model_versions_t{UNAVAILABLE_DUMMY_VERSION}));
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);

    tensor_map_t inputs, outputs;
    EXPECT_EQ(def->getInputsInfo(inputs, manager), StatusCode::MODEL_MISSING);
    EXPECT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::MODEL_MISSING);

    config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    auto instance = manager.findModelInstance("dummy", 0);
    ASSERT_NE(instance, nullptr);
    instance->unloadModel();

    EXPECT_EQ(def->getInputsInfo(inputs, manager), StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
    EXPECT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);

    config.setLocalPath("/tmp/non_existing_path_j3nmc783n");
    ASSERT_EQ(instance->loadModel(config), StatusCode::PATH_INVALID);

    EXPECT_EQ(def->getInputsInfo(inputs, manager), StatusCode::MODEL_VERSION_NOT_LOADED_YET);
    EXPECT_EQ(def->getOutputsInfo(outputs, manager), StatusCode::MODEL_VERSION_NOT_LOADED_YET);
}
