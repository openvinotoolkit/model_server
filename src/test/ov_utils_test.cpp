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
#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../filesystem.hpp"
#include "../modelinstance.hpp"
#include "../ov_utils.hpp"

using testing::ElementsAre;

TEST(OVUtils, CopyTensorDoesNotAllocateNewData) {
    const std::vector<size_t> shape{2, 3, 4, 5};
    const auto elementType = ov::element::Type(ov::element::Type_t::f32);
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * elementType.size();

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    ov::Tensor originalTensor(elementType, shape, data.data());
    ov::Tensor copyTensor = originalTensor;

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor.get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor.get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), totalByteSize);
    ASSERT_EQ(copyTensor.get_byte_size(), totalByteSize);

    ASSERT_EQ(copyTensor.get_strides(), originalTensor.get_strides());

    std::vector<float> originalTensorActualData;
    originalTensorActualData.assign(static_cast<float*>(originalTensor.data()), static_cast<float*>(originalTensor.data()) + elementsCount);

    std::vector<float> copyTensorActualData;
    copyTensorActualData.assign(static_cast<float*>(copyTensor.data()), static_cast<float*>(copyTensor.data()) + elementsCount);

    EXPECT_EQ(originalTensorActualData, data);
    EXPECT_EQ(copyTensorActualData, data);

    // Expect memory addresses to be the same and no new buffers were allocated
    EXPECT_EQ(originalTensor.data(), copyTensor.data());
}

TEST(OVUtils, CopyTensor) {
    const std::vector<size_t> shape{2, 3, 4, 5};
    const auto elementType = ov::element::Type(ov::element::Type_t::f32);
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * elementType.size();

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    ov::Tensor originalTensor(elementType, shape, data.data());
    ov::Tensor copyTensor;

    ASSERT_EQ(ovms::tensorClone(copyTensor, originalTensor), ovms::StatusCode::OK);

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor.get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor.get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), totalByteSize);
    ASSERT_EQ(copyTensor.get_byte_size(), totalByteSize);

    ASSERT_EQ(copyTensor.get_strides(), originalTensor.get_strides());

    std::vector<float> originalTensorActualData;
    originalTensorActualData.assign(static_cast<float*>(originalTensor.data()), static_cast<float*>(originalTensor.data()) + elementsCount);

    std::vector<float> copyTensorActualData;
    copyTensorActualData.assign(static_cast<float*>(copyTensor.data()), static_cast<float*>(copyTensor.data()) + elementsCount);

    EXPECT_EQ(originalTensorActualData, data);
    EXPECT_EQ(copyTensorActualData, data);

    // Expect memory addresses to differ since cloning should allocate new memory space for the cloned tensor
    EXPECT_NE(originalTensor.data(), copyTensor.data());
}

TEST(OVUtils, CloneStringTensor) {
    const auto elementType = ov::element::Type(ov::element::Type_t::string);

    std::vector<std::string> data{"abc", "", "defgh"};

    ov::Shape shape{data.size()};
    ov::Tensor originalTensor(elementType, shape, &data[0]);
    ov::Tensor copyTensor;

    ASSERT_EQ(ovms::tensorClone(copyTensor, originalTensor), ovms::StatusCode::OK);

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor.get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor.get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), copyTensor.get_byte_size());

    ASSERT_EQ(copyTensor.get_strides(), originalTensor.get_strides());

    std::string* actualData = copyTensor.data<std::string>();
    std::string* originalData = originalTensor.data<std::string>();
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_EQ(actualData[i], originalData[i]);
        EXPECT_NE(actualData[i].data(), originalData[i].data());
    }
}

TEST(OVUtils, ConstCopyTensor) {
    const std::vector<size_t> shape{2, 3, 4, 5};
    const auto elementType = ov::element::Type(ov::element::Type_t::f32);
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * elementType.size();

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    ov::Tensor originalTensor(elementType, shape, data.data());
    ov::Tensor copyTensor;

    ASSERT_EQ(ovms::tensorClone(copyTensor, originalTensor), ovms::StatusCode::OK);

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor.get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor.get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), totalByteSize);
    ASSERT_EQ(copyTensor.get_byte_size(), totalByteSize);

    ASSERT_EQ(copyTensor.get_strides(), originalTensor.get_strides());

    std::vector<float> originalTensorActualData;
    const void* start = (const void*)(originalTensor.data());
    originalTensorActualData.assign((float*)start, (float*)start + elementsCount);

    std::vector<float> copyTensorActualData;
    copyTensorActualData.assign(static_cast<float*>(copyTensor.data()), static_cast<float*>(copyTensor.data()) + elementsCount);

    EXPECT_EQ(originalTensorActualData, data);
    EXPECT_EQ(copyTensorActualData, data);

    // Expect memory addresses to differ since cloning should allocate new memory space for the cloned tensor
    EXPECT_NE(originalTensor.data(), copyTensor.data());
}

TEST(OVUtils, GetLayoutFromRTMap) {
    const std::string layoutStr = "N?...CH";

    // Empty rtmap
    ov::RTMap rtMap;
    auto layout = ovms::getLayoutFromRTMap(rtMap);
    EXPECT_EQ(layout, std::nullopt);

    // Rtmap with layout
    rtMap.insert(std::make_pair("param", ov::LayoutAttribute(ov::Layout(layoutStr))));
    layout = ovms::getLayoutFromRTMap(rtMap);
    EXPECT_EQ(layout, ov::Layout(layoutStr));

    // Rtmap with unknown param
    rtMap = ov::RTMap();
    rtMap.insert(std::make_pair("param_str", std::string{"string param"}));
    layout = ovms::getLayoutFromRTMap(rtMap);
    EXPECT_EQ(layout, std::nullopt);

    // Rtmap with both unknown and layout param
    rtMap.insert(std::make_pair("param", ov::LayoutAttribute(ov::Layout(layoutStr))));
    layout = ovms::getLayoutFromRTMap(rtMap);
    EXPECT_EQ(layout, ov::Layout(layoutStr));
}

TEST(OVUtils, ValidatePluginConfigurationPositive) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ovms::ModelConfig config;
    config.setTargetDevice("CPU");
    config.setPluginConfig({{"NUM_STREAMS", "10"}});
    ovms::plugin_config_t supportedPluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    auto status = ovms::validatePluginConfiguration(supportedPluginConfig, "CPU", ieCore);
    EXPECT_TRUE(status.ok());
}

TEST(OVUtils, ValidatePluginConfigurationPositiveBatch) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ovms::ModelConfig config;
    config.setTargetDevice("BATCH:CPU(4)");
    config.setPluginConfig({{"AUTO_BATCH_TIMEOUT", 10}});
    ovms::plugin_config_t supportedPluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    auto status = ovms::validatePluginConfiguration(supportedPluginConfig, "BATCH:CPU(4)", ieCore);
    EXPECT_TRUE(status.ok());
}

TEST(OVUtils, ValidatePluginConfigurationNegative) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ovms::ModelConfig config;
    config.setTargetDevice("CPU");
    config.setPluginConfig({{"WRONG_KEY ", "10"}});
    ovms::plugin_config_t unsupportedPluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    auto status = ovms::validatePluginConfiguration(unsupportedPluginConfig, "CPU", ieCore);
    EXPECT_FALSE(status.ok());
}

// Multi stage (read_model & compile_model time) plugin config
TEST(OVUtils, ValidatePluginConfigurationAllowEnableMmap) {
    ov::Core ieCore;
    ovms::ModelConfig config;
    config.setTargetDevice("CPU");
    config.setPluginConfig({{"ENABLE_MMAP", "NO"}, {"NUM_STREAMS", "1"}});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    auto status = ovms::validatePluginConfiguration(pluginConfig, "CPU", ieCore);
    EXPECT_TRUE(status.ok());
    auto model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml", {}, pluginConfig);
    auto compiledModel = ieCore.compile_model(model, "CPU", pluginConfig);
}
