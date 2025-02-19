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
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../dags/custom_node_library_manager.hpp"
#include "../dags/entry_node.hpp"
#include "../dags/exit_node.hpp"
#include "../dags/pipeline_factory.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../model_version_policy.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"

using namespace ovms;

TEST(EnsembleMetadata, OneNode) {
    /*
        This test creates pipeline definition with one DL model node.
        Test ensures we receive correct metadata - one input and one output for the DL model node.
    */

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({1, DUMMY_MODEL_INPUT_SIZE}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output = outputs.at("request_output_name");
    EXPECT_EQ(output->getShape(), Shape({1, DUMMY_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, OneModelNodeWithShapeRange) {
    /*
        This test creates pipeline definition with one DL model node which has dimensions accepting range of dimensions.
        Test ensures we receive correct metadata - one input and one output for the DL model node.
    */

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(1:20, 3:17)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({{1, 20}, {3, 17}}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output = outputs.at("request_output_name");
    EXPECT_EQ(output->getShape(), Shape({{1, 20}, {3, 17}}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, TwoParallelModelsShapeRangeIntersection) {
    /*
        This test creates pipeline definition with two parallel DL model nodes which has dimensions accepting different ranges of dimensions.
        Test ensures we receive correct metadata - intersection of both dimensions.
    */

    ConstructorEnabledModelManager manager;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_A");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(1:20, 3:17)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_B");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(15:30, 1:4)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node_A", "dummy_A", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_B", "dummy_B", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_A"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_B"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_A", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_A"}}},
        {"dummy_node_B", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_B"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_A"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_B"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({{15, 20}, {3, 4}}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output_A = outputs.at("request_output_name_A");
    EXPECT_EQ(output_A->getShape(), Shape({{1, 20}, {3, 17}}));  // ?
    EXPECT_EQ(output_A->getPrecision(), ovms::Precision::FP32);

    const auto& output_B = outputs.at("request_output_name_B");
    EXPECT_EQ(output_B->getShape(), Shape({{15, 30}, {1, 4}}));  // ?
    EXPECT_EQ(output_B->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, TwoParallelModelsOneShapeRangeOneStaticIntersection) {
    /*
        This test creates pipeline definition with two parallel DL model nodes, which one of them have dimensions accepting ranges of dimensions.
        Test ensures we receive correct metadata - intersection of both dimensions, meaning the final input shape is static.
    */

    ConstructorEnabledModelManager manager;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_A");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(17, 3)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_B");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(15:30, 1:4)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node_A", "dummy_A", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_B", "dummy_B", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_A"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_B"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_A", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_A"}}},
        {"dummy_node_B", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_B"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_A"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_B"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({17, 3}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output_A = outputs.at("request_output_name_A");
    EXPECT_EQ(output_A->getShape(), Shape({17, 3}));  // ?
    EXPECT_EQ(output_A->getPrecision(), ovms::Precision::FP32);

    const auto& output_B = outputs.at("request_output_name_B");
    EXPECT_EQ(output_B->getShape(), Shape({{15, 30}, {1, 4}}));  // ?
    EXPECT_EQ(output_B->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, TwoParallelModelsOneShapeAnyOneStaticIntersection) {
    /*
        This test creates pipeline definition with two parallel DL model nodes, which one of them have dimensions accepting any dimensions.
        Test ensures we receive correct metadata - intersection of both dimensions, meaning the final input shape is static.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_A");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(17, 3)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_B");
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(-1, -1)");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node_A", "dummy_A", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_B", "dummy_B", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_A"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_B"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node_A", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_A"}}},
        {"dummy_node_B", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_B"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_A"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_B"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({17, 3}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output_A = outputs.at("request_output_name_A");
    EXPECT_EQ(output_A->getShape(), Shape({17, 3}));  // ?
    EXPECT_EQ(output_A->getPrecision(), ovms::Precision::FP32);

    const auto& output_B = outputs.at("request_output_name_B");
    EXPECT_EQ(output_B->getShape(), Shape({Dimension::any(), Dimension::any()}));  // ?
    EXPECT_EQ(output_B->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, TwoParallelModelsLayoutIntersection) {
    /*
        This test creates pipeline definition with two parallel DL model nodes, which one of them have complete information about layout and the other one not.
        Test ensures we end up with complete information about layout - intersection of both layouts.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setName("dummy");
    config.setBatchSize(std::nullopt);
    config.parseLayoutParameter("{\"b\":\"N...\",\"a\":\"N...\"}");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_2");
    config.setBatchSize(std::nullopt);
    config.parseLayoutParameter("{\"b\":\"NC\",\"a\":\"NC\"}");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_2_node", "dummy_2", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_2_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_A"}}},
        {"dummy_2_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_B"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_A"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_B"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getLayout(), ovms::Layout{"NC"});

    const auto& output_A = outputs.at("request_output_name_A");
    EXPECT_EQ(output_A->getLayout(), ovms::Layout{"N..."});

    const auto& output_B = outputs.at("request_output_name_B");
    EXPECT_EQ(output_B->getLayout(), ovms::Layout{"NC"});
}

TEST(EnsembleMetadata, TwoParallelModelsDemultiplyEntryLayoutIntersection) {
    /*
        This test creates pipeline definition with two parallel DL model nodes, demultiplied at entry, which one of them have complete information about layout and the other one not.
        Test ensures we end up with complete information about layout - intersection of both layouts.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setName("dummy");
    config.setBatchSize(std::nullopt);
    config.parseLayoutParameter("{\"b\":\"N...\",\"a\":\"N...\"}");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    config = DUMMY_MODEL_CONFIG;
    config.setName("dummy_2");
    config.setBatchSize(std::nullopt);
    config.parseLayoutParameter("{\"b\":\"NC\",\"a\":\"NC\"}");
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::optional<int32_t> demultiplyCount = -1;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}, demultiplyCount},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_2_node", "dummy_2", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {ENTRY_NODE_NAME}},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_2_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_A"}}},
        {"dummy_2_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name_B"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_A"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_B"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getLayout(), ovms::Layout{"N?C"});

    const auto& output_A = outputs.at("request_output_name_A");
    EXPECT_EQ(output_A->getLayout(), ovms::Layout{"..."});

    const auto& output_B = outputs.at("request_output_name_B");
    EXPECT_EQ(output_B->getLayout(), ovms::Layout{"..."});
}

TEST(EnsembleMetadata, MultipleNodesOnDifferentLevelsUsingTheSamePipelineInputs) {
    /*
        This test creates pipeline definition with multiple connections referring to entry node.
        This way we use the same input tensors in nodes on different levels of depth.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo/getOutptusInfo does not return duplicated information.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig increment_model_config = DUMMY_MODEL_CONFIG;
    increment_model_config.setName("increment");
    ASSERT_EQ(manager.reloadModelWithVersions(increment_model_config), StatusCode::OK_RELOADED);

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK_RELOADED);

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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(outputs.size(), 3);
    ASSERT_NE(inputs.find("request_input_for_N1"), inputs.end());
    ASSERT_NE(inputs.find("request_input_for_N2_and_exit"), inputs.end());
    ASSERT_NE(outputs.find("intermediate_result_from_increment"), outputs.end());
    ASSERT_NE(outputs.find("intermediate_result_from_sum"), outputs.end());
    ASSERT_NE(outputs.find("original_input_for_N2"), outputs.end());

    const auto& request_input_for_N1 = inputs.at("request_input_for_N1");
    EXPECT_EQ(request_input_for_N1->getShape(), Shape({1, INCREMENT_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N1->getPrecision(), ovms::Precision::FP32);

    const auto& request_input_for_N2_and_exit = inputs.at("request_input_for_N2_and_exit");
    EXPECT_EQ(request_input_for_N2_and_exit->getShape(), Shape({1, SUM_MODEL_INPUT_SIZE}));
    EXPECT_EQ(request_input_for_N2_and_exit->getPrecision(), ovms::Precision::FP32);

    const auto& intermediate_result_from_increment = outputs.at("intermediate_result_from_increment");
    EXPECT_EQ(intermediate_result_from_increment->getShape(), Shape({1, INCREMENT_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(intermediate_result_from_increment->getPrecision(), ovms::Precision::FP32);

    const auto& intermediate_result_from_sum = outputs.at("intermediate_result_from_sum");
    EXPECT_EQ(intermediate_result_from_sum->getShape(), Shape({1, SUM_MODEL_OUTPUT_SIZE}));
    EXPECT_EQ(intermediate_result_from_sum->getPrecision(), ovms::Precision::FP32);

    const auto& original_input_for_N2 = outputs.at("original_input_for_N2");
    EXPECT_EQ(original_input_for_N2->getShape(), Shape({}));
    EXPECT_EQ(original_input_for_N2->getPrecision(), ovms::Precision::UNDEFINED);
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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("name_from_entry"), inputs.end());
    ASSERT_NE(outputs.find("name_for_response"), outputs.end());

    const auto& name_from_entry = inputs.at("name_from_entry");
    EXPECT_EQ(name_from_entry->getShape(), Shape({}));
    EXPECT_EQ(name_from_entry->getPrecision(), ovms::Precision::UNDEFINED);

    const auto& name_for_response = outputs.at("name_for_response");
    EXPECT_EQ(name_for_response->getShape(), Shape({}));
    EXPECT_EQ(name_for_response->getPrecision(), ovms::Precision::UNDEFINED);
}

TEST(EnsembleMetadata, ParallelDLModelNodesReferingToManyPipelineInputs) {
    /*
        This test creates pipeline definition with 4 parallel DL model nodes, all referring to entry node.
        All nodes require 2 inputs, all from different sources, so in total we need 8 different pipeline inputs.
        Test ensures that metadata returned by PipelineDefinition::getInptusInfo contains all 8 required inputs and one output.
    */
    ConstructorEnabledModelManager manager;

    ModelConfig sum_model_config = SUM_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(sum_model_config), StatusCode::OK_RELOADED);

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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 8);
    for (size_t i = 1; i <= 4; i++) {
        const std::string name_a = "request_input_for_quarter_" + std::to_string(i) + "_a";
        const std::string name_b = "request_input_for_quarter_" + std::to_string(i) + "_b";

        ASSERT_NE(inputs.find(name_a), inputs.end());
        ASSERT_NE(inputs.find(name_b), inputs.end());

        EXPECT_EQ(inputs.find(name_a)->second->getShape(), Shape({1, SUM_MODEL_INPUT_SIZE}));
        EXPECT_EQ(inputs.find(name_a)->second->getPrecision(), ovms::Precision::FP32);
        EXPECT_EQ(inputs.find(name_b)->second->getShape(), Shape({1, SUM_MODEL_INPUT_SIZE}));
        EXPECT_EQ(inputs.find(name_b)->second->getPrecision(), ovms::Precision::FP32);
    }

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(outputs.find("final_sum"), outputs.end());
    EXPECT_EQ(outputs.find("final_sum")->second->getShape(), Shape({1, SUM_MODEL_INPUT_SIZE}));
    EXPECT_EQ(outputs.find("final_sum")->second->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, OneUnavailableNodeBeforeRevalidationShouldWork) {
    /*
        This test creates pipeline definition with one DL model node which has model that becomes unavailable due to:
            a) no model version available
            b) model version is retired
            c) model is not loaded yet
        Test ensures we still receive metadata when underlying model is unloaded but PipelineDefinition is not revalidated
        yet.
    */

    const model_version_t UNAVAILABLE_DUMMY_VERSION = 99;

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

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

    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    config.setModelVersionPolicy(std::make_shared<SpecificModelVersionPolicy>(model_versions_t{UNAVAILABLE_DUMMY_VERSION}));
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();
    ASSERT_GT(inputs.size(), 0);
    ASSERT_GT(outputs.size(), 0);

    config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    auto instance = manager.findModelInstance("dummy", 0);
    ASSERT_NE(instance, nullptr);
    instance->retireModel();

    // we should still be able to get metadata since pipeline definition was not reloaded
    auto inputs2 = def->getInputsInfo();
    auto outputs2 = def->getOutputsInfo();
    ASSERT_GT(inputs2.size(), 0);
    ASSERT_GT(outputs2.size(), 0);

    config.setLocalPath("/tmp/non_existing_path_j3nmc783n");
    ASSERT_EQ(instance->loadModel(config), StatusCode::PATH_INVALID);

    // we should still be able to get metadata since pipeline definition was not reloaded
    auto inputs3 = def->getInputsInfo();
    auto outputs3 = def->getOutputsInfo();
    ASSERT_GT(inputs3.size(), 0);
    ASSERT_GT(outputs3.size(), 0);
}

TEST(EnsembleMetadata, OneCustomNode) {
    ConstructorEnabledModelManager manager;
    CustomNodeLibraryManager libraryManager;
    ASSERT_EQ(libraryManager.loadLibrary("add_sub", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_add_sub.so")), StatusCode::OK);
    NodeLibrary library{};
    ASSERT_EQ(libraryManager.getLibrary("add_sub", library), StatusCode::OK);
    ASSERT_TRUE(library.isValid());

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"output_numbers", "output_numbers"}}, std::nullopt, {}, library, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", "input_numbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{"output_numbers", "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->initializeNodeResources(manager), StatusCode::OK);
    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);
    ASSERT_EQ(def->validateForCycles(), StatusCode::OK);
    ASSERT_EQ(def->validateDemultiplexerGatherNodesOrder(), StatusCode::OK);
    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({1, -1}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    const auto& output = outputs.at("request_output_name");
    EXPECT_EQ(output->getShape(), Shape({1, -1}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, ParallelCustomNodes) {
    ConstructorEnabledModelManager manager;
    CustomNodeLibraryManager libraryManager;
    ASSERT_EQ(libraryManager.loadLibrary("add_sub", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_add_sub.so")), StatusCode::OK);
    NodeLibrary library{};
    ASSERT_EQ(libraryManager.getLibrary("add_sub", library), StatusCode::OK);
    ASSERT_TRUE(library.isValid());

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::CUSTOM, "custom_node_0", "", std::nullopt, {{"output_numbers", "output_numbers"}}, std::nullopt, {}, library, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"output_numbers", "output_numbers"}}, std::nullopt, {}, library, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"output_numbers", "output_numbers"}}, std::nullopt, {}, library, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_0"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", "input_numbers"}}}};
    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", "input_numbers"}}}};
    connections["custom_node_2"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", "input_numbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_0", {{"output_numbers", "request_output_name_0"}}},
        {"custom_node_1", {{"output_numbers", "request_output_name_1"}}},
        {"custom_node_2", {{"output_numbers", "request_output_name_2"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->initializeNodeResources(manager), StatusCode::OK);
    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);
    ASSERT_EQ(def->validateForCycles(), StatusCode::OK);
    ASSERT_EQ(def->validateDemultiplexerGatherNodesOrder(), StatusCode::OK);
    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 3);
    ASSERT_NE(inputs.find("request_input_name"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name_0"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_1"), outputs.end());
    ASSERT_NE(outputs.find("request_output_name_2"), outputs.end());

    const auto& input = inputs.at("request_input_name");
    EXPECT_EQ(input->getShape(), Shape({1, -1}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::FP32);

    for (int i = 0; i < 3; i++) {
        const auto& output = outputs.at("request_output_name_" + std::to_string(i));
        EXPECT_EQ(output->getShape(), Shape({1, -1}));
        EXPECT_EQ(output->getPrecision(), ovms::Precision::FP32);
    }
}

struct MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode {
    static int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
        return 0;
    }
    static int deinitialize(void* customNodeLibraryInternalManager) {
        return 0;
    }
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        return 1;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        *infoCount = 2;
        *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

        (*info)[0].name = "input_A_1";
        (*info)[0].precision = FP32;
        (*info)[0].dimsCount = 2;
        (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[0].dims[0] = 1;
        (*info)[0].dims[1] = 1000;

        (*info)[1].name = "input_B_1";
        (*info)[1].precision = FP32;
        (*info)[1].dimsCount = 2;
        (*info)[1].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[1].dims[0] = 1;
        (*info)[1].dims[1] = 400;
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        *infoCount = 2;
        *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

        (*info)[0].name = "output_A_1";
        (*info)[0].precision = FP32;
        (*info)[0].dimsCount = 3;
        (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[0].dims[0] = 3;
        (*info)[0].dims[1] = 1;
        (*info)[0].dims[2] = 10;

        (*info)[1].name = "output_B_1";
        (*info)[1].precision = FP32;
        (*info)[1].dimsCount = 3;
        (*info)[1].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[1].dims[0] = 3;
        (*info)[1].dims[1] = 1;
        (*info)[1].dims[2] = 4;
        return 0;
    }
    static int release(void* ptr, void* customNodeLibraryInternalManager) {
        free(ptr);
        return 0;
    }
};

struct MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode {
    static int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
        return 0;
    }
    static int deinitialize(void* customNodeLibraryInternalManager) {
        return 0;
    }
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        return 1;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        *infoCount = 2;
        *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

        (*info)[0].name = "input_A_2";
        (*info)[0].precision = FP32;
        (*info)[0].dimsCount = 2;
        (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[0].dims[0] = 1;
        (*info)[0].dims[1] = 10;

        (*info)[1].name = "input_B_2";
        (*info)[1].precision = FP32;
        (*info)[1].dimsCount = 2;
        (*info)[1].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[1].dims[0] = 1;
        (*info)[1].dims[1] = 4;
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam*, int, void* customNodeLibraryInternalManager) {
        *infoCount = 1;
        *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

        (*info)[0].name = "output_A_2";
        (*info)[0].precision = FP32;
        (*info)[0].dimsCount = 3;
        (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
        (*info)[0].dims[0] = 4;
        (*info)[0].dims[1] = 1;
        (*info)[0].dims[2] = 10;
        return 0;
    }
    static int release(void* ptr, void* customNodeLibraryInternalManager) {
        free(ptr);
        return 0;
    }
};

TEST(EnsembleMetadata, CustomNodeMultipleDemultiplexers) {
    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    NodeLibrary libraryMatchingFollowingNode{
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::initialize,
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::deinitialize,
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::execute,
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::getInputsInfo,
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::getOutputsInfo,
        MockLibraryDemultiplexer2Inputs2OutputsMatchingFollowingNode::release};
    NodeLibrary libraryMatchingPreviousNode{
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::initialize,
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::deinitialize,
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::execute,
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::getInputsInfo,
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::getOutputsInfo,
        MockLibraryDemultiplexer2Inputs1OutputMatchingPreviousNode::release};
    ASSERT_TRUE(libraryMatchingFollowingNode.isValid());
    ASSERT_TRUE(libraryMatchingPreviousNode.isValid());

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name_A", "request_input_name_A"}, {"request_input_name_B", "request_input_name_B"}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"output_A_1", "output_A_1"}, {"output_B_1", "output_B_1"}}, 3, {}, libraryMatchingFollowingNode, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"output_A_2", "output_A_2"}}, 4, {}, libraryMatchingPreviousNode, parameters_t{{"add_value", "0.5"}, {"sub_value", "1.2"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, std::nullopt, {}, {}, {}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1", "custom_node_2"}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{"request_input_name_A", "input_A_1"}, {"request_input_name_B", "input_B_1"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"output_A_1", "input_A_2"}, {"output_B_1", "input_B_2"}}}};

    connections["dummy_node"] = {
        {"custom_node_2", {{"output_A_2", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->initializeNodeResources(manager), StatusCode::OK);
    ASSERT_EQ(def->validateNodes(manager), StatusCode::OK);
    ASSERT_EQ(def->validateForCycles(), StatusCode::OK);
    ASSERT_EQ(def->validateDemultiplexerGatherNodesOrder(), StatusCode::OK);
    ASSERT_EQ(def->validate(manager), StatusCode::OK);

    auto inputs = def->getInputsInfo();
    auto outputs = def->getOutputsInfo();

    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("request_input_name_A"), inputs.end());
    ASSERT_NE(inputs.find("request_input_name_B"), inputs.end());
    ASSERT_NE(outputs.find("request_output_name"), outputs.end());

    const auto& input_A = inputs.at("request_input_name_A");
    EXPECT_EQ(input_A->getShape(), Shape({1, 1000}));
    EXPECT_EQ(input_A->getPrecision(), ovms::Precision::FP32);

    const auto& input_B = inputs.at("request_input_name_B");
    EXPECT_EQ(input_B->getShape(), Shape({1, 400}));
    EXPECT_EQ(input_B->getPrecision(), ovms::Precision::FP32);

    const auto& output = outputs.at("request_output_name");
    EXPECT_EQ(output->getShape(), Shape({3, 4, 1, 10}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::FP32);
}

TEST(EnsembleMetadata, GatherFromNotExistingNode) {
    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, std::nullopt, {"no_node"}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_EXISTING_NODE);
}

TEST(EnsembleMetadata, GatherFromNotDemultiplexer) {
    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, std::nullopt, {"request"}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{"request_input_name", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "request_output_name"}}}};

    auto def = std::make_unique<PipelineDefinition>(
        "my_new_pipeline", info, connections);

    ASSERT_EQ(def->validateNodes(manager), StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_DEMULTIPLEXER);
}

TEST(EnsembleMetadata, DemultiplyFromEntryNodeIsNotAllowed) {
    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{"request_input_name", "request_input_name"}}, 4},
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
}
