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
#include <sys/stat.h>
#include <sys/types.h>

#include "../modelconfig.hpp"
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

    config.setTargetDevice("GPU");
    auto device = config.getTargetDevice();
    EXPECT_EQ(device, "GPU");

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
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shapeMode, ovms::AUTO);

    config.parseShapeParameter(valid_str1);
    shapes = config.getShapes();
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shapeMode, ovms::FIXED);
    EXPECT_THAT(shapes[ovms::ANONYMOUS_INPUT_NAME].shape, ElementsAre(64, 128, 256, 300));

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

TEST(ModelConfig, isDeviceUsed) {
    ovms::ModelConfig config;
    config.setTargetDevice("GPU");
    EXPECT_FALSE(config.isDeviceUsed("CPU"));
    config.setTargetDevice("CPU");
    EXPECT_TRUE(config.isDeviceUsed("CPU"));
    config.setTargetDevice("HETERO:MYRIAD,CPU");
    EXPECT_TRUE(config.isDeviceUsed("CPU"));
    config.setTargetDevice("HETERO:MYRIAD,GPU");
    EXPECT_FALSE(config.isDeviceUsed("CPU"));
}

TEST(ModelConfig, shapeConfigurationEqual_SingleInput) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::AUTO, {}}}});
    rhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::AUTO, {}}}});
    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 224, 224}}}});
    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 224, 224}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_SingleInput_WrongShape) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::AUTO, {}}}});
    rhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 100}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 225, 225}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{ANONYMOUS_INPUT_NAME, {Mode::FIXED, {1, 3, 225, 225}}}});
    rhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.setShapes({{"a", {Mode::FIXED, {1, 3, 224, 224}}}});
    rhs.setShapes({{"a", {Mode::FIXED, {1, 3, 225, 225}}}});
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    shapes_map_t shapesMap = {
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}}};

    lhs.setShapes(shapesMap);
    rhs.setShapes(shapesMap);

    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_Anonymous) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.parseShapeParameter("auto");
    rhs.parseShapeParameter("auto");

    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("(1,3,224,224)");
    rhs.parseShapeParameter("(1,3,224,224)");

    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("{\"a\": \"auto\"}");
    rhs.parseShapeParameter("{\"a\": \"auto\"}");

    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("(1,3,224,224)");
    rhs.parseShapeParameter("auto");

    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("auto");
    rhs.parseShapeParameter("(1,3,224,224)");

    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("auto");
    rhs.parseShapeParameter("{\"a\": \"auto\"}");

    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));

    lhs.parseShapeParameter("auto");
    rhs.parseShapeParameter("{\"a\": \"auto\", \"b\": \"auto\"}");

    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_WrongShape) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    rhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 225, 225}}},
        {"c", {Mode::AUTO, {}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_WrongShapeMode) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    rhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::FIXED, {1, 1000}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_WrongInputName) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    rhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"wrong_input", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_WrongNumberOfInputs) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    rhs.setShapes({
        {"a", {Mode::AUTO, {}}},
        {"b", {Mode::FIXED, {1, 3, 224, 224}}},
        {"c", {Mode::AUTO, {}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, modelVersionPolicyIncorrect) {
    std::string command = "{\"test\": {\"versions\":[1, 3, 4]}}";
    ovms::ModelConfig config;
    auto result = config.parseModelVersionPolicy(command);
    EXPECT_EQ(result, ovms::StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY);
}

TEST(ModelConfig, ConfigParseNodeWithForbiddenShapeName) {
    std::string config = R"#(
        {
        "model_config_list": [
            {
                "config": {
                    "name": "alpha",
                    "base_path": "/tmp/models/dummy1",
                    "shape": {")#" +
                         ovms::ANONYMOUS_INPUT_NAME + R"#(": "(1, 3, 600, 600)"}
                }
            }
        ]
    }
    )#";

    rapidjson::Document configJson;
    rapidjson::ParseResult parsingSucceeded = configJson.Parse(config.c_str());
    ASSERT_EQ(parsingSucceeded, true);

    const auto modelConfigList = configJson.FindMember("model_config_list");
    ASSERT_NE(modelConfigList, configJson.MemberEnd());
    const auto& configs = modelConfigList->value.GetArray();
    ASSERT_EQ(configs.Size(), 1);
    ovms::ModelConfig modelConfig;
    auto status = modelConfig.parseNode(configs[0]["config"]);

    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(modelConfig.getShapes().size(), 0);
}
