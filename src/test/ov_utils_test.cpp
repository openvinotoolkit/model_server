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

#include <openvino/runtime/core.hpp>

#include "src/filesystem/filesystem.hpp"
#include "../modelinstance.hpp"
#include "../ov_utils.hpp"
#include "test_utils.hpp"

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
    adjustConfigToAllowModelFileRemovalWhenLoaded(config);
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    auto status = ovms::validatePluginConfiguration(pluginConfig, "CPU", ieCore);
    EXPECT_TRUE(status.ok());
    auto model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml", {}, pluginConfig);
    auto compiledModel = ieCore.compile_model(model, "CPU", pluginConfig);
}

#ifdef __linux__
// Tests using the testable overload with explicit system parameters

TEST(OVUtils, ApplyDefaultCpuPropertiesLatencyConstrainedContainer) {
    // Simulate: 8 cores available in docker, 24 physical cores per socket, 2 sockets
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = "LATENCY";

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/8, /*physicalCoresPerSocket=*/24, /*socketsCount=*/2, /*dockerCpuQuota=*/8);
    ASSERT_TRUE(status.ok());

    // coreCount(8) <= physicalCoresPerSocket(24) * socketsCount(2) = 48, so threads should be set
    ASSERT_NE(properties.find(ov::inference_num_threads.name()), properties.end());
    EXPECT_EQ(properties[ov::inference_num_threads.name()].as<int>(), 8);

    // NUM_STREAMS should not be set for LATENCY mode
    EXPECT_EQ(properties.find(ov::num_streams.name()), properties.end());

    // CPU pinning: dockerCpuQuota > 0 → pinning disabled
    ASSERT_NE(properties.find(ov::hint::enable_cpu_pinning.name()), properties.end());
    EXPECT_EQ(properties[ov::hint::enable_cpu_pinning.name()].as<bool>(), false);
}

TEST(OVUtils, ApplyDefaultCpuPropertiesThroughputConstrainedContainer) {
    // Simulate: 8 cores available in docker, 24 physical cores per socket, 2 sockets
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = "THROUGHPUT";

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/8, /*physicalCoresPerSocket=*/24, /*socketsCount=*/2, /*dockerCpuQuota=*/8);
    ASSERT_TRUE(status.ok());

    // num_streams = min(8, 24*2) = 8
    ASSERT_NE(properties.find(ov::num_streams.name()), properties.end());
    EXPECT_EQ(properties[ov::num_streams.name()].as<int>(), 8);

    // inference_num_threads = 8 (coreCount <= totalPhysical)
    ASSERT_NE(properties.find(ov::inference_num_threads.name()), properties.end());
    EXPECT_EQ(properties[ov::inference_num_threads.name()].as<int>(), 8);
}

TEST(OVUtils, ApplyDefaultCpuPropertiesThroughputUnconstrainedContainer) {
    // Simulate: 96 cores available, 24 physical cores per socket, 2 sockets (48 physical total)
    // coreCount(96) > physicalCoresPerSocket(24) * socketsCount(2) = 48
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = "THROUGHPUT";

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/96, /*physicalCoresPerSocket=*/24, /*socketsCount=*/2, /*dockerCpuQuota=*/0);
    ASSERT_TRUE(status.ok());

    // num_streams = min(96, 48) = 48
    ASSERT_NE(properties.find(ov::num_streams.name()), properties.end());
    EXPECT_EQ(properties[ov::num_streams.name()].as<int>(), 48);

    // inference_num_threads NOT set (coreCount > totalPhysical, let OV decide)
    EXPECT_EQ(properties.find(ov::inference_num_threads.name()), properties.end());

    // CPU pinning: dockerCpuQuota == 0 → pinning enabled
    ASSERT_NE(properties.find(ov::hint::enable_cpu_pinning.name()), properties.end());
    EXPECT_EQ(properties[ov::hint::enable_cpu_pinning.name()].as<bool>(), true);
}

TEST(OVUtils, ApplyDefaultCpuPropertiesLatencyUnconstrainedContainer) {
    // Simulate: 96 cores (with HT), 24 physical cores per socket, 2 sockets
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = "LATENCY";

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/96, /*physicalCoresPerSocket=*/24, /*socketsCount=*/2, /*dockerCpuQuota=*/0);
    ASSERT_TRUE(status.ok());

    // coreCount(96) > 48, so inference_num_threads should NOT be set
    EXPECT_EQ(properties.find(ov::inference_num_threads.name()), properties.end());
    EXPECT_EQ(properties.find(ov::num_streams.name()), properties.end());
}

TEST(OVUtils, ApplyDefaultCpuPropertiesNoPerformanceHint) {
    // No PERFORMANCE_HINT set - should only set pinning and threads
    ov::AnyMap properties;

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/16, /*physicalCoresPerSocket=*/24, /*socketsCount=*/1, /*dockerCpuQuota=*/16);
    ASSERT_TRUE(status.ok());

    // No num_streams (no throughput hint)
    EXPECT_EQ(properties.find(ov::num_streams.name()), properties.end());

    // inference_num_threads set (16 <= 24)
    ASSERT_NE(properties.find(ov::inference_num_threads.name()), properties.end());
    EXPECT_EQ(properties[ov::inference_num_threads.name()].as<int>(), 16);
}

TEST(OVUtils, ApplyDefaultCpuPropertiesDoesNotOverrideExistingValues) {
    // Pre-set values should not be overwritten
    ov::AnyMap properties;
    properties[ov::hint::performance_mode.name()] = "THROUGHPUT";
    properties[ov::num_streams.name()] = 4;
    properties[ov::inference_num_threads.name()] = 12;
    properties[ov::hint::enable_cpu_pinning.name()] = true;

    auto status = ovms::applyDefaultCpuProperties(properties, /*coreCount=*/8, /*physicalCoresPerSocket=*/24, /*socketsCount=*/2, /*dockerCpuQuota=*/8);
    ASSERT_TRUE(status.ok());

    // All values should remain unchanged
    EXPECT_EQ(properties[ov::num_streams.name()].as<int>(), 4);
    EXPECT_EQ(properties[ov::inference_num_threads.name()].as<int>(), 12);
    EXPECT_EQ(properties[ov::hint::enable_cpu_pinning.name()].as<bool>(), true);
}
#endif

TEST(RecommendTargetDevice, NoGpuDevicesReturnsCpu) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices;
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "CPU");
}

TEST(RecommendTargetDevice, SingleDiscreteGpuReturnsIt) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU", /*isDiscrete=*/true, /*freeMemBytes=*/8000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU");
}

TEST(RecommendTargetDevice, SingleIntegratedGpuReturnsIt) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU", /*isDiscrete=*/false, /*freeMemBytes=*/2000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU");
}

TEST(RecommendTargetDevice, MultipleDiscreteGpusReturnsMostFreeVram) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/true, /*freeMemBytes=*/4000000000},
        {"GPU.1", /*isDiscrete=*/true, /*freeMemBytes=*/8000000000},
        {"GPU.2", /*isDiscrete=*/true, /*freeMemBytes=*/6000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.1");
}

TEST(RecommendTargetDevice, DiscretePreferredOverIntegrated) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/false, /*freeMemBytes=*/2000000000},
        {"GPU.1", /*isDiscrete=*/true, /*freeMemBytes=*/8000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.1");
}

TEST(RecommendTargetDevice, MultipleIntegratedGpusReturnsFirst) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/false, /*freeMemBytes=*/2000000000},
        {"GPU.1", /*isDiscrete=*/false, /*freeMemBytes=*/4000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.0");
}

TEST(RecommendTargetDevice, MultipleDiscreteGpusWithEqualVramReturnsFirst) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/true, /*freeMemBytes=*/8000000000},
        {"GPU.1", /*isDiscrete=*/true, /*freeMemBytes=*/8000000000}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.0");
}

TEST(RecommendTargetDevice, MultipleDiscreteGpusWithZeroVramReturnsFirst) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/true, /*freeMemBytes=*/0},
        {"GPU.1", /*isDiscrete=*/true, /*freeMemBytes=*/0}};
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.0");
}

TEST(RecommendTargetDevice, MixedDiscreteAndIntegratedPrefersDiscrete) {
    std::vector<ovms::GpuDeviceInfo> gpuDevices{
        {"GPU.0", /*isDiscrete=*/false, /*freeMemBytes=*/16000000000},
        {"GPU.1", /*isDiscrete=*/true, /*freeMemBytes=*/1000000000}};
    // Even though integrated has more free memory, discrete is preferred
    EXPECT_EQ(ovms::recommendTargetDevice(gpuDevices), "GPU.1");
}
