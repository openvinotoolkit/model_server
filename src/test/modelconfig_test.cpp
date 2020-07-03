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

#include <fstream>
#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "../status.hpp"

#include "test_utils.hpp"

using namespace testing;
using ::testing::UnorderedElementsAre;

TEST(ModelConfig, getters_setters) {
    ovms::ModelConfig config;

    config.setName("alexnet");
    auto name = config.getName();
    EXPECT_EQ(name, "alexnet");

    config.setBasePath("/path");
    auto path = config.getBasePath();
    EXPECT_EQ(path, "/path");

    config.setBackend("GPU");
    auto backend = config.getBackend();
    EXPECT_EQ(backend, "GPU");

    config.setBatchSize(5);
    auto batchSize = config.getBatchSize();
    EXPECT_EQ(batchSize, 5);

    config.setNireq(11);
    auto nireq = config.getNireq();
    EXPECT_EQ(nireq, 11);

    ovms::model_version_t ver = 500;
    config.setVersion(ver);
    auto version = config.getVersion();
    EXPECT_EQ(version, ver);
}

TEST(ModelConfig, layout_single) {
    ovms::ModelConfig config;

    config.setLayout("NHWC");
    auto l1 = config.getLayout();
    auto l2 = config.getLayouts();
    EXPECT_EQ(l1, "NHWC");
    EXPECT_EQ(l2.size(), 0);
}

TEST(ModelConfig, layout_multi) {
    ovms::ModelConfig config;

    ovms::layouts_map_t layouts;
    layouts["A"] = "layout_A";
    layouts["B"] = "layout_B";

    config.setLayout("NHWC");
    config.setLayouts(layouts);

    auto l1 = config.getLayout();
    auto l2 = config.getLayouts();
    EXPECT_EQ(l1, "");
    EXPECT_THAT(l2, UnorderedElementsAre(
                        Pair("A", "layout_A"),
                        Pair("B", "layout_B")));

    config.setLayout("NHWC");
    l2 = config.getLayouts();
    EXPECT_EQ(l2.size(), 0);

    config.setLayouts(layouts);
    config.addLayout("C", "layout_C");
    l1 = config.getLayout();
    l2 = config.getLayouts();
    EXPECT_EQ(l1, "");
    EXPECT_THAT(l2, UnorderedElementsAre(
                        Pair("A", "layout_A"),
                        Pair("B", "layout_B"),
                        Pair("C", "layout_C")));
}

TEST(ModelConfig, shape) {
    ovms::ModelConfig config;

    ovms::ShapeInfo s1{ovms::FIXED, {1, 2, 3}};
    ovms::ShapeInfo s2{ovms::FIXED, {6, 6, 200, 300}};
    ovms::ShapeInfo s3{ovms::FIXED, {100, 500}};

    ovms::shapes_map_t shapeMap;
    shapeMap["first"] = s1;
    shapeMap["second"] = s2;

    config.setShapes(shapeMap);
    auto gs1 = config.getShapes();
    EXPECT_EQ(gs1.size(), 2);
    EXPECT_THAT(gs1["first"].shape, ElementsAre(1, 2, 3));
    EXPECT_THAT(gs1["second"].shape, ElementsAre(6, 6, 200, 300));

    // mutli shape
    config.setShapes(shapeMap);
    config.addShape("third", s3);

    gs1 = config.getShapes();
    EXPECT_EQ(gs1.size(), 3);
    EXPECT_THAT(gs1["third"].shape, ElementsAre(100, 500));
}

TEST(ModelConfig, parseShapeFromString) {
    ovms::ModelConfig config;
    // Valid
    std::string auto_str = "auto";
    std::string valid_str1 = "(64,128,256,   300)";
    std::string valid_str2 = "   (     64 , 300   )   ";
    ovms::ShapeInfo shapeInfo;

    config.parseShape(shapeInfo, auto_str);
    EXPECT_EQ(shapeInfo.shapeMode, ovms::AUTO);
    EXPECT_EQ(shapeInfo.shape.size(), 0);

    config.parseShape(shapeInfo, valid_str1);
    EXPECT_EQ(shapeInfo.shapeMode, ovms::FIXED);
    EXPECT_THAT(shapeInfo.shape, ElementsAre(64, 128, 256, 300));

    config.parseShape(shapeInfo, valid_str2);
    EXPECT_EQ(shapeInfo.shapeMode, ovms::FIXED);
    EXPECT_THAT(shapeInfo.shape, ElementsAre(64, 300));

    // Invalid
    std::string invalid_str1 = "(1, 2, 3, 4]";
    std::string invalid_str2 = "(1, 2, 3.14, 4)";
    ovms::Status status;

    status = config.parseShape(shapeInfo, invalid_str1);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
    status = config.parseShape(shapeInfo, invalid_str2);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
}

TEST(ModelConfig, parseShapeParam) {
    ovms::ModelConfig config;
    // Valid
    std::string auto_str = "auto";
    std::string valid_str1 = "(64,128,256,300)";
    std::string valid_str2 = "{\"input\": \"(1, 3, 3, 200)\"}";
    std::string valid_str3 = "{\"input\": \"auto\", \"extra_input\": \"(10)\"}";

    config.parseShapeParameter(auto_str);
    auto shapes = config.getShapes();
    EXPECT_EQ(shapes[ovms::DEFAULT_INPUT_NAME].shapeMode, ovms::AUTO);

    config.parseShapeParameter(valid_str1);
    shapes = config.getShapes();
    EXPECT_EQ(shapes[ovms::DEFAULT_INPUT_NAME].shapeMode, ovms::FIXED);
    EXPECT_THAT(shapes[ovms::DEFAULT_INPUT_NAME].shape, ElementsAre(64, 128, 256, 300));

    config.parseShapeParameter(valid_str2);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::FIXED);
    EXPECT_THAT(shapes["input"].shape, ElementsAre(1, 3, 3, 200));

    config.parseShapeParameter(valid_str3);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::AUTO);
    EXPECT_EQ(shapes["input"].shape.size(), 0);
    EXPECT_EQ(shapes["extra_input"].shapeMode, ovms::FIXED);
    EXPECT_THAT(shapes["extra_input"].shape, ElementsAre(10));

    // Invalid

    std::string invalid_str1 = "string";
    std::string invalid_str2 = "[1, 3, 43]";
    std::string invalid_str3 = "{\"input\": \"auto\", \"extra_input\": \"10\"}";

    auto status = config.parseShapeParameter(invalid_str1);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);

    status = config.parseShapeParameter(invalid_str2);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);

    status = config.parseShapeParameter(invalid_str3);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
}

TEST(ModelConfig, setShapesFromRequest) {
    ovms::ModelConfig config;
    ovms::ShapeInfo shapeInfo;
    shapeInfo.shapeMode = ovms::AUTO;
    shapeInfo.shape = {1, 2, 3, 4};
    ovms::shapes_map_t shapes = { {"input", shapeInfo} };
    config.setShapes(shapes);

    std::map<std::string, ovms::shape_t> req1 { {"input", {10, 20, 30, 40}} };
    config.setShapesFromRequest(req1);
    auto r_shapes = config.getShapes();
    EXPECT_THAT(r_shapes["input"].shape, ElementsAre(10, 20, 30, 40));

    shapeInfo.shapeMode = ovms::AUTO;
    shapeInfo.shape = {};
    shapes = { {ovms::DEFAULT_INPUT_NAME, shapeInfo} };
    config.setShapes(shapes);

    std::map<std::string, ovms::shape_t> req2 { {"input", {100, 200, 300, 400}} };
    config.setShapesFromRequest(req2);
    r_shapes = config.getShapes();
    EXPECT_THAT(r_shapes[ovms::DEFAULT_INPUT_NAME].shape, ElementsAre(100, 200, 300, 400));

    shapes = { {"input1", shapeInfo}, {"input2", shapeInfo} };
    config.setShapes(shapes);

    std::map<std::string, ovms::shape_t> req3 { {"input1", {1, 1, 1}}, {"input2", {2, 2, 2}} };
    config.setShapesFromRequest(req3);
    r_shapes = config.getShapes();
    EXPECT_THAT(r_shapes["input1"].shape, ElementsAre(1, 1, 1));
    EXPECT_THAT(r_shapes["input2"].shape, ElementsAre(2, 2, 2));
}

TEST(ModelConfig, plugin_config) {
    ovms::ModelConfig config;
    ovms::plugin_config_t pluginConfig{
        {"OptionA", "ValueA"},
        {"OptionX", "ValueX"},
    };

    config.setPluginConfig(pluginConfig);

    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_THAT(actualPluginConfig, UnorderedElementsAre(
                                        Pair("OptionA", "ValueA"),
                                        Pair("OptionX", "ValueX")));
}

TEST(ModelConfig, mappingInputs) {
    ovms::ModelConfig config;
    ovms::mapping_config_t mapping{
        {"resnet", "value"},
        {"output", "input"}};

    config.setMappingInputs(mapping);
    auto ret = config.getMappingInputs();
    EXPECT_THAT(ret, UnorderedElementsAre(
                         Pair("resnet", "value"),
                         Pair("output", "input")));

    auto in = config.getMappingInputByKey("output");
    auto empty = config.getMappingInputByKey("notexist");

    EXPECT_EQ(in, "input");
    EXPECT_EQ(empty, "");
}

TEST(ModelConfig, mappingOutputs) {
    ovms::ModelConfig config;
    ovms::mapping_config_t mapping{
        {"resnet", "value"},
        {"output", "input"}};

    config.setMappingOutputs(mapping);
    auto ret = config.getMappingOutputs();
    EXPECT_THAT(ret, UnorderedElementsAre(
                         Pair("resnet", "value"),
                         Pair("output", "input")));

    auto in = config.getMappingOutputByKey("output");
    auto empty = config.getMappingOutputByKey("notexist");

    EXPECT_EQ(in, "input");
    EXPECT_EQ(empty, "");
}

TEST(ModelConfig, parseModelMappingWhenJsonMatchSchema) {
    ovms::ModelConfig config;

    const char* json = R"({
       "inputs":{
            "key":"value1"
        },
       "outputs":{
            "key":"value2"
        }
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), false);
    EXPECT_EQ(config.getMappingOutputs().empty(), false);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenOutputsMissingInConfig) {
    ovms::ModelConfig config;

    const char* json = R"({
       "inputs":{
            "key":"value1"
        }
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    std::unordered_map<std::string, std::string> expectedInputs = {
            { "key", "value1"}
    };

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), false);
    EXPECT_EQ(config.getMappingInputs(), expectedInputs);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenInputsMissingInConfig) {
    ovms::ModelConfig config;

    const char* json = R"({
       "outputs":{
            "key":"value2"
        }
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    std::unordered_map<std::string, std::string> expectedOutputs = {
            { "key", "value2"}
    };

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), true);
    EXPECT_EQ(config.getMappingOutputs(), expectedOutputs);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenAdditionalObjectInConfig) {
    ovms::ModelConfig config;

    const char* json = R"({
       "inputs":{
            "key":"value1"
        },
       "outputs":{
            "key":"value2"
        },
       "object":{
            "key":"value3"
        }
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), false);
    EXPECT_EQ(config.getMappingOutputs().empty(), false);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenInputsIsNotAnObject) {
    ovms::ModelConfig config;

    const char* json = R"({
       "inputs":["Array", "is", "not", "an", "object"],
       "outputs":{
            "key":"value2"
        }
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), true);
    EXPECT_EQ(config.getMappingOutputs().empty(), false);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenOutputsIsNotAnObject) {
    ovms::ModelConfig config;

    const char* json = R"({
       "inputs":{
            "key":"value"
        },
       "outputs":["Array", "is", "not", "an", "object"]
    })";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(json, filename);

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), false);
    EXPECT_EQ(config.getMappingOutputs().empty(), true);
    EXPECT_EQ(ret, ovms::StatusCode::OK);
}

TEST(ModelConfig, parseModelMappingWhenConfigIsNotJson) {
    ovms::ModelConfig config;

    const char* invalidJson = "asdasdasd";

    std::string tmp_dir = "/tmp";
    int16_t version = 0;
    config.setBasePath(tmp_dir);
    config.setVersion(version);
    std::string path = tmp_dir + "/" + std::to_string(version);
    std::filesystem::create_directories(path);

    std::string filename = path + "/" + ovms::MAPPING_CONFIG_JSON;
    createConfigFileWithContent(invalidJson, filename);

    auto ret = config.parseModelMapping();
    EXPECT_EQ(config.getMappingInputs().empty(), true);
    EXPECT_EQ(config.getMappingOutputs().empty(), true);
    EXPECT_EQ(ret, ovms::StatusCode::JSON_INVALID);
}
