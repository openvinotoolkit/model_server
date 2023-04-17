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

    config.setRootDirectoryPath("/pathto/");
    config.setBasePath("relative/path");
    path = config.getBasePath();
    EXPECT_EQ(path, "/pathto/relative/path");

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

    config.setStateful(true);
    auto is = config.isStateful();
    EXPECT_EQ(is, true);

    config.setLowLatencyTransformation(true);
    is = config.isLowLatencyTransformationUsed();
    EXPECT_EQ(is, true);

    config.setMaxSequenceNumber(11);
    auto seq = config.getMaxSequenceNumber();
    EXPECT_EQ(seq, 11);
}

TEST(ModelConfig, layout_single) {
    ovms::ModelConfig config;

    config.setLayout(ovms::LayoutConfiguration{"NCHW", "NHWC"});
    auto l1 = config.getLayout();
    auto l2 = config.getLayouts();
    EXPECT_EQ(l1.getTensorLayout(), "NCHW");
    EXPECT_EQ(l1.getModelLayout(), "NHWC");
    EXPECT_EQ(l2.size(), 0);
}

TEST(ModelConfig, layout_multi) {
    ovms::ModelConfig config;

    ovms::layout_configurations_map_t layouts;
    layouts["A"] = ovms::LayoutConfiguration{"NCHW", "NHWC"};
    layouts["B"] = ovms::LayoutConfiguration{"CN", "NC"};

    config.setLayout("NHWC");
    config.setLayouts(layouts);

    auto l1 = config.getLayout();
    auto l2 = config.getLayouts();
    EXPECT_EQ(l1.isSet(), false);
    ASSERT_EQ(l2.count("A"), 1);
    ASSERT_EQ(l2.count("B"), 1);
    EXPECT_EQ(l2.find("A")->second.getTensorLayout(), "NCHW");
    EXPECT_EQ(l2.find("A")->second.getModelLayout(), "NHWC");
    EXPECT_EQ(l2.find("B")->second.getTensorLayout(), "CN");
    EXPECT_EQ(l2.find("B")->second.getModelLayout(), "NC");

    config.setLayout("NHWC");
    l1 = config.getLayout();
    l2 = config.getLayouts();
    EXPECT_EQ(l1.isSet(), true);
    EXPECT_EQ(l1.getTensorLayout(), "NHWC");
    EXPECT_EQ(l1.getModelLayout(), "NHWC");
    EXPECT_EQ(l2.size(), 0);

    config.setLayouts(layouts);
    l1 = config.getLayout();
    l2 = config.getLayouts();
    EXPECT_EQ(l1.isSet(), false);
    EXPECT_EQ(l2.size(), 2);
}

TEST(ModelConfig, parseLayoutParam_single) {
    using namespace ovms;
    ModelConfig config;
    // Valid
    std::string valid_str1 = "";
    std::string valid_str2 = "nchw";
    std::string valid_str3 = " Nhwc : ncHW ";
    std::string valid_str4 = "nC";

    ASSERT_EQ(config.parseLayoutParameter(valid_str1), StatusCode::OK);
    EXPECT_EQ(config.getLayouts().size(), 0);
    EXPECT_EQ(config.getLayout().getTensorLayout(), "");
    EXPECT_EQ(config.getLayout().getModelLayout(), "");

    ASSERT_EQ(config.parseLayoutParameter(valid_str2), StatusCode::OK);
    EXPECT_EQ(config.getLayouts().size(), 0);
    EXPECT_EQ(config.getLayout().getTensorLayout(), "NCHW");
    EXPECT_EQ(config.getLayout().getModelLayout(), "NCHW");

    ASSERT_EQ(config.parseLayoutParameter(valid_str3), StatusCode::OK);
    EXPECT_EQ(config.getLayouts().size(), 0);
    EXPECT_EQ(config.getLayout().getTensorLayout(), "NHWC");
    EXPECT_EQ(config.getLayout().getModelLayout(), "NCHW");

    ASSERT_EQ(config.parseLayoutParameter(valid_str4), StatusCode::OK);
    EXPECT_EQ(config.getLayouts().size(), 0);
    EXPECT_EQ(config.getLayout().getTensorLayout(), "NC");
    EXPECT_EQ(config.getLayout().getModelLayout(), "NC");

    // Invalid
    std::vector<std::string> invalid_str{
        std::string{"nc::nc"},
        std::string{":nc:nc"},
        std::string{"nc>nc"},
    };

    for (std::string str : invalid_str) {
        auto status = config.parseLayoutParameter(str);
        EXPECT_EQ(status, ovms::StatusCode::LAYOUT_WRONG_FORMAT) << " Failed for: " << str;
        EXPECT_EQ(config.getLayout().getTensorLayout(), "");
        EXPECT_EQ(config.getLayout().getModelLayout(), "");
        EXPECT_EQ(config.getLayouts().size(), 0);
    }
}

TEST(ModelConfig, parseLayoutParam_multi) {
    using namespace ovms;
    ModelConfig config;
    // Valid
    std::string valid_str1 = " { \"input\": \"nchw:nhwc\", \"output\": \"nc\" } ";

    ASSERT_EQ(config.parseLayoutParameter(valid_str1), StatusCode::OK);
    EXPECT_EQ(config.getLayouts().size(), 2);
    ASSERT_EQ(config.getLayouts().count("input"), 1);
    ASSERT_EQ(config.getLayouts().count("output"), 1);
    EXPECT_EQ(config.getLayouts().find("input")->second.getTensorLayout(), "NCHW");
    EXPECT_EQ(config.getLayouts().find("input")->second.getModelLayout(), "NHWC");
    EXPECT_EQ(config.getLayouts().find("output")->second.getTensorLayout(), "NC");
    EXPECT_EQ(config.getLayouts().find("output")->second.getModelLayout(), "NC");

    // Invalid
    std::vector<std::string> invalid_str{
        std::string{" { \"input\": \"nchw>nhwc\", \"output\": \"nc\" } "},
        std::string{" { \"input\": \"nchw:nhwc:nchw\", \"output\": \"nc\" } "},
    };

    for (std::string str : invalid_str) {
        auto status = config.parseLayoutParameter(str);
        EXPECT_EQ(status, ovms::StatusCode::LAYOUT_WRONG_FORMAT) << " Failed for: " << str;
        EXPECT_EQ(config.getLayout().getTensorLayout(), "");
        EXPECT_EQ(config.getLayout().getModelLayout(), "");
        EXPECT_EQ(config.getLayouts().size(), 0);
    }
}

TEST(ModelConfig, shape) {
    ovms::ModelConfig config;

    ovms::ShapeInfo s1{ovms::FIXED, {1, 2, 3}};
    ovms::ShapeInfo s2{ovms::FIXED, {6, 6, 200, 300}};
    ovms::ShapeInfo s3{ovms::FIXED, {100, 500}};

    ovms::shapes_info_map_t shapeMap;
    shapeMap["first"] = s1;
    shapeMap["second"] = s2;

    config.setShapes(shapeMap);
    auto gs1 = config.getShapes();
    EXPECT_EQ(gs1.size(), 2);
    EXPECT_EQ((gs1["first"].shape), (ovms::Shape{1, 2, 3}));
    EXPECT_EQ((gs1["second"].shape), (ovms::Shape{6, 6, 200, 300}));

    // mutli shape
    config.setShapes(shapeMap);
    config.addShape("third", s3);

    gs1 = config.getShapes();
    EXPECT_EQ(gs1.size(), 3);
    EXPECT_EQ((gs1["third"].shape), (ovms::Shape{100, 500}));
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
    EXPECT_EQ(shapeInfo.shape, (ovms::Shape{64, 128, 256, 300}));

    config.parseShape(shapeInfo, valid_str2);
    EXPECT_EQ(shapeInfo.shapeMode, ovms::FIXED);
    EXPECT_EQ(shapeInfo.shape, (ovms::Shape{64, 300}));

    // Invalid
    std::string invalid_str1 = "(1, 2, 3, 4]";
    std::string invalid_str2 = "(1, 2, 3.14, 4)";
    std::string invalid_str3 = "(1,2221413523534234632463462346234562)";
    std::string invalid_str4 = "(auto, 2, 3, 4)";
    ovms::Status status;

    status = config.parseShape(shapeInfo, invalid_str1);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
    status = config.parseShape(shapeInfo, invalid_str2);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);

    status = config.parseShape(shapeInfo, invalid_str3);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
    status = config.parseShape(shapeInfo, invalid_str4);
    EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
}

TEST(ModelConfig, parseDimParam) {
    ovms::ModelConfig config;
    // Valid
    std::string auto_str = "auto";
    std::string valid_str1 = " 24 ";
    std::string valid_str2 = " 30:32 ";
    std::string valid_str3 = " -1 ";

    config.setBatchingParams(auto_str);
    EXPECT_EQ(config.getBatchingMode(), ovms::AUTO);
    EXPECT_EQ(config.getBatchSize(), std::nullopt);

    config.setBatchingParams(valid_str1);
    EXPECT_EQ(config.getBatchingMode(), ovms::FIXED);
    EXPECT_EQ(config.getBatchSize(), ovms::Dimension{24});

    config.setBatchingParams(valid_str2);
    EXPECT_EQ(config.getBatchingMode(), ovms::FIXED);
    EXPECT_EQ(config.getBatchSize(), ovms::Dimension(30, 32));

    config.setBatchingParams(valid_str3);
    EXPECT_EQ(config.getBatchingMode(), ovms::FIXED);
    EXPECT_EQ(config.getBatchSize(), ovms::Dimension::any());

    // Invalid

    std::vector<std::string> invalid_str{
        std::string{"word"},
        std::string{":9"},
        std::string{"9:"},
        std::string{"9-30"},
        std::string{"9..30"},
        std::string{"0"},
        std::string{"9::30"},
        std::string{"-90:10"},
        std::string{"?"},
        std::string{"2.5:3"},
        std::string{"500000000000000000"},
    };

    for (std::string str : invalid_str) {
        config.setBatchingParams(str);
        EXPECT_EQ(config.getBatchingMode(), ovms::FIXED) << " invalid for str " << str;
        EXPECT_EQ(config.getBatchSize(), std::nullopt) << " invalid for str " << str;
    }
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
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shape, (ovms::Shape{64, 128, 256, 300}));

    config.parseShapeParameter(valid_str2);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["input"].shape, (ovms::Shape{1, 3, 3, 200}));

    config.parseShapeParameter(valid_str3);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::AUTO);
    EXPECT_EQ(shapes["input"].shape.size(), 0);
    EXPECT_EQ(shapes["extra_input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["extra_input"].shape, (ovms::Shape{10}));

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

TEST(ModelConfig, parseShapeDynamicParam) {
    using namespace ovms;
    ovms::ModelConfig config;
    // Valid
    std::string valid_str1 = "(64:128,128,256:512,300:301)";
    std::string valid_str2 = "{\"input\": \"(1, 3:6, 3, 200:100000)\"}";
    std::string valid_str3 = "{\"input\": \"auto\", \"extra_input\": \"(10:20)\"}";

    ASSERT_EQ(config.parseShapeParameter(valid_str1), StatusCode::OK);
    auto shapes = config.getShapes();
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shape, (ovms::Shape{{64, 128}, 128, {256, 512}, {300, 301}}));

    auto shape = ovms::Shape{1, 5, 10};

    ASSERT_EQ(config.parseShapeParameter(valid_str2), StatusCode::OK);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["input"].shape, (ovms::Shape{1, {3, 6}, 3, {200, 100000}}));

    ASSERT_EQ(config.parseShapeParameter(valid_str3), StatusCode::OK);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::AUTO);
    EXPECT_EQ(shapes["input"].shape.size(), 0);
    EXPECT_EQ(shapes["extra_input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["extra_input"].shape, (ovms::Shape{{10, 20}}));

    // Invalid

    std::vector<std::string> invalid_str{
        std::string{"[1:50, 300]"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(9:10,,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(:9,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(9:,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(9-30,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(9..30,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(0,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(9::30,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(-90:10,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(?,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(2.5:3,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(1,20,500000000000000000)\"}"},
    };

    for (std::string str : invalid_str) {
        auto status = config.parseShapeParameter(str);
        EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
    }
}

TEST(ModelConfig, dynamicShapeToString) {
    using namespace ovms;
    Shape shape{1, 5, {10, 20}, Dimension::any(), 3, {1, 290}};
    EXPECT_EQ(shape.toString(), "(1,5,[10~20],-1,3,[1~290])");
}

TEST(ModelConfig, parseShapeAnyDimParam) {
    using namespace ovms;
    ovms::ModelConfig config;
    // Valid
    std::string valid_str1 = "(-1,3,224,224)";
    std::string valid_str2 = "{\"input\": \"(-1,  5, -1, 2)\"}";
    std::string valid_str3 = "{\"input\": \"auto\", \"extra_input\": \"(10:20,-1)\"}";

    ASSERT_EQ(config.parseShapeParameter(valid_str1), StatusCode::OK);
    auto shapes = config.getShapes();
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes[ovms::ANONYMOUS_INPUT_NAME].shape, (Shape{Dimension::any(), 3, 224, 224}));

    auto shape = ovms::Shape{1, 5, 10};

    ASSERT_EQ(config.parseShapeParameter(valid_str2), StatusCode::OK);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["input"].shape, (Shape{Dimension::any(), 5, Dimension::any(), 2}));

    ASSERT_EQ(config.parseShapeParameter(valid_str3), StatusCode::OK);
    shapes = config.getShapes();
    EXPECT_EQ(shapes["input"].shapeMode, ovms::AUTO);
    EXPECT_EQ(shapes["input"].shape.size(), 0);
    EXPECT_EQ(shapes["extra_input"].shapeMode, ovms::FIXED);
    EXPECT_EQ(shapes["extra_input"].shape, (Shape{{10, 20}, Dimension::any()}));

    // Invalid

    std::vector<std::string> invalid_str{
        std::string{"[-1, 300]"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(--30,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(-5,20,50)\"}"},
        std::string{"{\"input\": \"auto\", \"extra_input\": \"(5-,20,50)\"}"},
    };

    for (std::string str : invalid_str) {
        auto status = config.parseShapeParameter(str);
        EXPECT_EQ(status, ovms::StatusCode::SHAPE_WRONG_FORMAT);
    }
}

TEST(ModelConfig, plugin_config_number) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"OptionA\":1,\"OptionX\":2.45}";

    auto status = config.parsePluginConfig(pluginConfig_str);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_THAT(actualPluginConfig, UnorderedElementsAre(
                                        Pair("OptionA", "1"),
                                        Pair("OptionX", "2.450000")));
}

TEST(ModelConfig, plugin_config_string) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"OptionA\":\"1\",\"OptionX\":\"2.45\"}";

    auto status = config.parsePluginConfig(pluginConfig_str);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_THAT(actualPluginConfig, UnorderedElementsAre(
                                        Pair("OptionA", "1"),
                                        Pair("OptionX", "2.45")));
}

TEST(ModelConfig, plugin_config_invalid) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"OptionX\":{}}";

    auto status = config.parsePluginConfig(pluginConfig_str);
    EXPECT_EQ(status, ovms::StatusCode::PLUGIN_CONFIG_WRONG_FORMAT);
    auto actualPluginConfig = config.getPluginConfig();
}

TEST(ModelConfig, plugin_config_legacy_cpu) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"CPU_THROUGHPUT_STREAMS\":\"CPU_THROUGHPUT_AUTO\"}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["PERFORMANCE_HINT"], "THROUGHPUT");
}

TEST(ModelConfig, plugin_config_legacy_cpu_num) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"CPU_THROUGHPUT_STREAMS\":5}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["NUM_STREAMS"], "5");
}

TEST(ModelConfig, plugin_config_legacy_cpu_str) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"CPU_THROUGHPUT_STREAMS\":\"5\", \"CPU_BIND_THREAD\":\"NO\", \"CPU_THREADS_NUM\": \"2\"}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["NUM_STREAMS"], "5");
    EXPECT_EQ(actualPluginConfig["AFFINITY"], "NONE");
    EXPECT_EQ(actualPluginConfig["INFERENCE_NUM_THREADS"], "2");
    EXPECT_EQ(actualPluginConfig.count("CPU_THREADS_NUM"), 0);
    EXPECT_EQ(actualPluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
    EXPECT_EQ(actualPluginConfig.count("CPU_BIND_THREAD"), 0);
}

TEST(ModelConfig, plugin_config_legacy_gpu) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"GPU_THROUGHPUT_STREAMS\":\"GPU_THROUGHPUT_AUTO\"}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["PERFORMANCE_HINT"], "THROUGHPUT");
}

TEST(ModelConfig, plugin_config_cpu_bind_thread) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"CPU_BIND_THREAD\":\"YES\"}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["AFFINITY"], "CORE");
}

TEST(ModelConfig, plugin_config_legacy_gpu_num) {
    ovms::ModelConfig config;
    std::string pluginConfig_str = "{\"GPU_THROUGHPUT_STREAMS\":5}";
    auto status = config.parsePluginConfig(pluginConfig_str);
    auto actualPluginConfig = config.getPluginConfig();
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(actualPluginConfig["NUM_STREAMS"], "5");
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

TEST(ModelConfig, mappingRealInputs) {
    ovms::ModelConfig config;
    ovms::mapping_config_t realMapping{
        {"value", "resnet"},
        {"input", "output"}};

    config.setRealMappingInputs(realMapping);
    auto ret = config.getRealMappingInputs();
    EXPECT_THAT(ret, UnorderedElementsAre(
                         Pair("value", "resnet"),
                         Pair("input", "output")));

    auto in = config.getRealInputNameByValue("input");
    auto empty = config.getRealInputNameByValue("notexist");

    EXPECT_EQ(in, "output");
    EXPECT_EQ(empty, "");
}

TEST(ModelConfig, mappingRealOutputs) {
    ovms::ModelConfig config;
    ovms::mapping_config_t realMapping{
        {"value", "resnet"},
        {"input", "output"}};

    config.setRealMappingOutputs(realMapping);
    auto ret = config.getRealMappingOutputs();
    EXPECT_THAT(ret, UnorderedElementsAre(
                         Pair("value", "resnet"),
                         Pair("input", "output")));

    auto in = config.getRealOutputNameByValue("input");
    auto empty = config.getRealOutputNameByValue("notexist");

    EXPECT_EQ(in, "output");
    EXPECT_EQ(empty, "");
}

TEST(ModelConfig, isSingleDeviceUsed) {
    ovms::ModelConfig config;
    config.setTargetDevice("GPU");
    EXPECT_FALSE(config.isSingleDeviceUsed("CPU"));
    config.setTargetDevice("CPU");
    EXPECT_TRUE(config.isSingleDeviceUsed("CPU"));
    config.setTargetDevice("HETERO:MYRIAD,CPU");
    EXPECT_FALSE(config.isSingleDeviceUsed("CPU"));
    config.setTargetDevice("HETERO:MYRIAD,GPU");
    EXPECT_FALSE(config.isSingleDeviceUsed("CPU"));
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

    shapes_info_map_t shapesMap = {
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

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_EqualRanges) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {224, 1024}, {224, 512}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    rhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {224, 1024}, {224, 512}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_EqualAnyResolution) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, Dimension::any(), Dimension::any()}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    rhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, Dimension::any(), Dimension::any()}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    EXPECT_TRUE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_AnyColorVsRangeColor) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, Dimension::any(), Dimension::any()}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    rhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, Dimension::any(), Dimension::any()}}},
        {"c", {Mode::FIXED, Shape{1, Dimension::any(), 224, 224}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_DifferentMinResolution) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {224, 1024}, {224, 512}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    rhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {100, 1024}, {100, 512}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    EXPECT_FALSE(lhs.isShapeConfigurationEqual(rhs));
}

TEST(ModelConfig, shapeConfigurationEqual_MultipleInputs_DifferentMaxResolution) {
    using namespace ovms;
    ModelConfig lhs, rhs;

    lhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {224, 1024}, {224, 512}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
    });
    rhs.setShapes({
        {"b", {Mode::FIXED, Shape{1, 3, {224, 300}, {224, 300}}}},
        {"c", {Mode::FIXED, Shape{1, {1, 3}, 224, 224}}},
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

TEST(ModelConfig, ConfigParseNodeWithInvalidShapeFormatArray) {
    std::string config = R"#(
        {
        "model_config_list": [
            {
                "config": {
                    "name": "alpha",
                    "base_path": "/tmp/models/dummy1",
                    "shape": {
                        "input": [
                            "auto",
                            3, 
                            600, 
                            600
                            ]
                        }
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

TEST(ModelConfig, ConfigParseNodeWithInvalidShapeFormatString) {
    std::string config = R"#(
        {
        "model_config_list": [
            {
                "config": {
                    "name": "alpha",
                    "base_path": "/tmp/models/dummy1",
                    "shape": {
                        "input": "(auto, 2, 244, 244)"
                        }
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

TEST(ModelConfig, ConfigParseNodeWithValidShapeFormatArray) {
    std::string config = R"#(
        {
        "model_config_list": [
            {
                "config": {
                    "name": "alpha",
                    "base_path": "/tmp/models/dummy1",
                    "shape": {
                        "input": [
                            1,
                            3, 
                            600, 
                            600
                            ]
                        }
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
    EXPECT_EQ(modelConfig.getShapes().size(), 1);
    auto shapes = modelConfig.getShapes();
    ASSERT_TRUE(shapes.find("input") != shapes.end());
    EXPECT_EQ(shapes["input"].shape, (ovms::Shape{1, 3, 600, 600}));
}

static std::string config_low_latency_no_stateful = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_low_latency",
                "base_path": "/tmp/models/dummy1",
                "low_latency_transformation": true
            }
        }
    ]
}
)#";

static std::string config_low_latency_non_stateful = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_low_latency_stateful",
                "base_path": "/tmp/models/dummy1",
                "stateful": false,
                "low_latency_transformation": true
            }
        }
    ]
}
)#";

static std::string config_idle_sequence_cleanup_non_stateful = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_timeout_stateful",
                "base_path": "/tmp/models/dummy1",
                "stateful": false,
                "idle_sequence_cleanup": true
            }
        }
    ]
}
)#";

static std::string config_max_sequence_number_non_stateful = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_max_sequence_number_stateful",
                "stateful": false,
                "base_path": "/tmp/models/dummy1",
                "max_sequence_number": 1000
            }
        }
    ]
}
)#";

static std::string config_max_sequence_number = R"#(
        {
        "model_config_list": [
            {
                "config": {
                    "name": "config_max_sequence_number",
                    "base_path": "/tmp/models/dummy1",
                    "max_sequence_number": 1
                }
            }
        ]
    }
    )#";

static std::string config_stateful_should_pass = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_stateful_should_pass",
                "base_path": "/tmp/models/dummy1",
                "stateful": true,
                "max_sequence_number": 1,
                "low_latency_transformation": true
            }
        }
    ]
}
)#";

static std::string config_low_invalid_max_seq = R"#(
    {
    "model_config_list": [
        {
            "config": {
                "name": "config_low_invalid_max_seq",
                "base_path": "/tmp/models/dummy1",
                "stateful": true,
                "max_sequence_number": 5294967295,
                "low_latency_transformation": true
            }
        }
    ]
}
)#";

class ModelConfigParseModel : public ::testing::TestWithParam<std::pair<std::string, ovms::StatusCode>> {
};

TEST_P(ModelConfigParseModel, SetWithStateful) {
    std::pair<std::string, ovms::StatusCode> testPair = GetParam();
    std::string config = testPair.first;
    rapidjson::Document configJson;
    rapidjson::ParseResult parsingSucceeded = configJson.Parse(config.c_str());
    ASSERT_EQ(parsingSucceeded, true);

    const auto modelConfigList = configJson.FindMember("model_config_list");
    ASSERT_NE(modelConfigList, configJson.MemberEnd());
    const auto& configs = modelConfigList->value.GetArray();
    ASSERT_EQ(configs.Size(), 1);
    ovms::ModelConfig modelConfig;
    std::cout << "Testing config named: " << configs[0]["config"]["name"].GetString() << std::endl;

    auto status = modelConfig.parseNode(configs[0]["config"]);

    ASSERT_EQ(status, testPair.second);
    if (testPair.second == ovms::StatusCode::OK) {
        ASSERT_EQ(modelConfig.isLowLatencyTransformationUsed(), true);
        ASSERT_EQ(modelConfig.isStateful(), true);
        ASSERT_EQ(modelConfig.getMaxSequenceNumber(), 1);
    }
}

std::vector<std::pair<std::string, ovms::StatusCode>> configs = {
    {config_low_latency_no_stateful, ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER},
    {config_max_sequence_number, ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER},
    {config_max_sequence_number_non_stateful, ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER},
    {config_idle_sequence_cleanup_non_stateful, ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER},
    {config_low_latency_non_stateful, ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER},
    {config_low_invalid_max_seq, ovms::StatusCode::INVALID_MAX_SEQUENCE_NUMBER},
    {config_stateful_should_pass, ovms::StatusCode::OK}};

INSTANTIATE_TEST_SUITE_P(
    Test,
    ModelConfigParseModel,
    ::testing::ValuesIn(configs));
