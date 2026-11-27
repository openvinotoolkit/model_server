//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/core.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#ifdef __linux__
#include <unistd.h>
#endif

#include "src/status.hpp"
#include "src/gpu_memory_info_internal.hpp"
#include "../systeminfo.hpp"
#include "gpuenvironment.hpp"

using namespace testing;
using ovms::getCoreCount;
using ovms::StatusCode;

TEST(SystemInfo, getCoreCount) {
    uint16_t cpuCount = getCoreCount();
    EXPECT_GE(cpuCount, 1);
    EXPECT_LE(cpuCount, std::thread::hardware_concurrency());
}

namespace {
using ovms::GpuDeviceType;
using ovms::GpuMemoryInfo;
using ovms::GpuMemoryStatus;

struct MemoryModule {
    uint64_t size = 1024;
    uint64_t free = 512;
    bool tile = false;
    zes_mem_loc_t location = ZES_MEM_LOC_DEVICE;
    zes_mem_type_t type = ZES_MEM_TYPE_DDR;
    ze_result_t propertiesResult = ZE_RESULT_SUCCESS;
    ze_result_t stateResult = ZE_RESULT_SUCCESS;
};

struct EngineGroup {
    zes_engine_group_t type = ZES_ENGINE_GROUP_COMPUTE_ALL;
    bool tile = false;
    uint32_t tileId = 0;
    uint64_t activeTime = 100;
    uint64_t timestamp = 1000;
    ze_result_t propertiesResult = ZE_RESULT_SUCCESS;
    ze_result_t activityResult = ZE_RESULT_SUCCESS;
};

struct PhysicalGpu {
    uint8_t uuid = 1;
    std::vector<MemoryModule> modules{MemoryModule{}};
    ze_result_t enumerationResult = ZE_RESULT_SUCCESS;
    ze_result_t propertiesResult = ZE_RESULT_SUCCESS;
    std::vector<EngineGroup> engines{EngineGroup{}};
    ze_result_t engineEnumerationResult = ZE_RESULT_SUCCESS;
};

class GpuMemoryInfoTest : public ::testing::Test {
protected:
    static thread_local GpuMemoryInfoTest* active;
    std::vector<PhysicalGpu> physical{PhysicalGpu{}};
    ze_result_t initializationResult = ZE_RESULT_SUCCESS;
    ze_result_t driverResult = ZE_RESULT_SUCCESS;
    ze_result_t deviceResult = ZE_RESULT_SUCCESS;
    bool shrinkDeviceCount = false;
    bool retryDeviceCount = false;
    ovms::detail::SysmanApi api{init, drivers, devices, properties, modules, memoryProperties, memoryState,
        engineGroups, engineProperties, engineActivity};

    void SetUp() override { active = this; }
    void TearDown() override { active = nullptr; }

    static GpuMemoryInfo identity(uint8_t uuid = 1, GpuDeviceType type = GpuDeviceType::DISCRETE) {
        GpuMemoryInfo info;
        info.deviceId = "GPU." + std::to_string(uuid - 1);
        info.deviceType = type;
        info.uuid = std::array<uint8_t, 16>{};
        (*info.uuid)[0] = uuid;
        return info;
    }

    ovms::GpuMemoryInfoResult collect(std::vector<GpuMemoryInfo> identities = {identity()}) {
        return ovms::detail::collectGpuMemoryInfo(std::move(identities), api);
    }

    ovms::GpuUtilizationSample sample(std::vector<GpuMemoryInfo> identities = {identity()}) {
        std::vector<ovms::GpuUtilizationDeviceSample> devices;
        for (const auto& identity : identities) {
            ovms::GpuUtilizationDeviceSample device;
            device.deviceId = identity.deviceId;
            device.uuid = identity.uuid;
            device.deviceType = identity.deviceType;
            devices.push_back(std::move(device));
        }
        return ovms::detail::collectGpuUtilizationSample(std::move(devices), api);
    }

    static ze_result_t ZE_APICALL init(zes_init_flags_t flags) {
        EXPECT_EQ(flags, 0);
        return active->initializationResult;
    }
    static ze_result_t ZE_APICALL drivers(uint32_t* count, zes_driver_handle_t* handles) {
        if (!handles)
            *count = 1;
        else
            handles[0] = reinterpret_cast<zes_driver_handle_t>(active);
        return active->driverResult;
    }
    static ze_result_t ZE_APICALL devices(zes_driver_handle_t, uint32_t* count, zes_device_handle_t* handles) {
        const uint32_t available = static_cast<uint32_t>(active->physical.size());
        if (!handles) {
            *count = available;
        } else {
            if (active->retryDeviceCount) {
                active->retryDeviceCount = false;
                return ZE_RESULT_ERROR_INVALID_SIZE;
            }
            *count = std::min(*count, available);
            if (active->shrinkDeviceCount)
                *count = 0;
            for (uint32_t index = 0; index < *count; ++index)
                handles[index] = reinterpret_cast<zes_device_handle_t>(&active->physical[index]);
        }
        return active->deviceResult;
    }
    static ze_result_t ZE_APICALL properties(zes_device_handle_t device, zes_device_properties_t* output) {
        EXPECT_EQ(output->stype, ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES);
        auto* extended = static_cast<zes_device_ext_properties_t*>(output->pNext);
        EXPECT_EQ(extended->stype, ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES);
        const auto* gpu = reinterpret_cast<PhysicalGpu*>(device);
        extended->uuid.id[0] = gpu->uuid;
        extended->type = ZES_DEVICE_TYPE_GPU;
        return gpu->propertiesResult;
    }
    static ze_result_t ZE_APICALL modules(zes_device_handle_t device, uint32_t* count, zes_mem_handle_t* handles) {
        auto* gpu = reinterpret_cast<PhysicalGpu*>(device);
        if (!handles) {
            *count = static_cast<uint32_t>(gpu->modules.size());
        } else {
            *count = std::min(*count, static_cast<uint32_t>(gpu->modules.size()));
            for (uint32_t index = 0; index < *count; ++index)
                handles[index] = reinterpret_cast<zes_mem_handle_t>(&gpu->modules[index]);
        }
        return gpu->enumerationResult;
    }
    static ze_result_t ZE_APICALL memoryProperties(zes_mem_handle_t module, zes_mem_properties_t* output) {
        EXPECT_EQ(output->stype, ZES_STRUCTURE_TYPE_MEM_PROPERTIES);
        EXPECT_EQ(output->pNext, nullptr);
        const auto* memory = reinterpret_cast<MemoryModule*>(module);
        output->physicalSize = memory->size;
        output->onSubdevice = memory->tile;
        output->location = memory->location;
        output->type = memory->type;
        return memory->propertiesResult;
    }
    static ze_result_t ZE_APICALL memoryState(zes_mem_handle_t module, zes_mem_state_t* output) {
        EXPECT_EQ(output->stype, ZES_STRUCTURE_TYPE_MEM_STATE);
        EXPECT_EQ(output->pNext, nullptr);
        const auto* memory = reinterpret_cast<MemoryModule*>(module);
        output->free = memory->free;
        output->size = 1;
        return memory->stateResult;
    }
    static ze_result_t ZE_APICALL engineGroups(zes_device_handle_t device, uint32_t* count, zes_engine_handle_t* handles) {
        auto* gpu = reinterpret_cast<PhysicalGpu*>(device);
        if (!handles) {
            *count = static_cast<uint32_t>(gpu->engines.size());
        } else {
            *count = std::min(*count, static_cast<uint32_t>(gpu->engines.size()));
            for (uint32_t index = 0; index < *count; ++index)
                handles[index] = reinterpret_cast<zes_engine_handle_t>(&gpu->engines[index]);
        }
        return gpu->engineEnumerationResult;
    }
    static ze_result_t ZE_APICALL engineProperties(zes_engine_handle_t handle, zes_engine_properties_t* output) {
        EXPECT_EQ(output->stype, ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES);
        EXPECT_EQ(output->pNext, nullptr);
        const auto* engine = reinterpret_cast<EngineGroup*>(handle);
        output->type = engine->type;
        output->onSubdevice = engine->tile;
        output->subdeviceId = engine->tileId;
        return engine->propertiesResult;
    }
    static ze_result_t ZE_APICALL engineActivity(zes_engine_handle_t handle, zes_engine_stats_t* output) {
        const auto* engine = reinterpret_cast<EngineGroup*>(handle);
        output->activeTime = engine->activeTime;
        output->timestamp = engine->timestamp;
        return engine->activityResult;
    }
};
thread_local GpuMemoryInfoTest* GpuMemoryInfoTest::active = nullptr;

TEST_F(GpuMemoryInfoTest, PhysicalCapacityAndLiveFreeBytes) {
    auto result = collect();
    ASSERT_EQ(result.status, GpuMemoryStatus::AVAILABLE);
    ASSERT_EQ(result.devices.size(), 1);
    EXPECT_EQ(result.devices[0].totalPhysicalBytes, 1024);
    EXPECT_EQ(result.devices[0].availablePhysicalBytes, 512);
    physical[0].modules[0].free = 0;
    result = collect();
    EXPECT_EQ(result.devices[0].status, GpuMemoryStatus::AVAILABLE);
    EXPECT_EQ(result.devices[0].availablePhysicalBytes, 0);
}

TEST_F(GpuMemoryInfoTest, MapsReorderedMixedCardsByUuid) {
    physical = {PhysicalGpu{2}, PhysicalGpu{1}};
    physical[0].modules[0].location = ZES_MEM_LOC_SYSTEM;
    physical[0].modules[0].free = 128;
    const auto result = collect({identity(), identity(2, GpuDeviceType::INTEGRATED)});
    ASSERT_EQ(result.devices.size(), 2);
    EXPECT_EQ(result.devices[0].deviceId, "GPU.0");
    EXPECT_EQ(result.devices[0].availablePhysicalBytes, 512);
    EXPECT_FALSE(result.devices[0].sharedMemory);
    EXPECT_EQ(result.devices[1].deviceId, "GPU.1");
    EXPECT_EQ(result.devices[1].availablePhysicalBytes, 128);
    EXPECT_TRUE(result.devices[1].sharedMemory);
}

TEST_F(GpuMemoryInfoTest, MissingSymbolsAndInitializationFailuresAreExplicit) {
    api.memoryState = nullptr;
    EXPECT_EQ(collect().status, GpuMemoryStatus::UNAVAILABLE);
    api.memoryState = memoryState;
    initializationResult = ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS;
    const auto result = collect();
    EXPECT_EQ(result.status, GpuMemoryStatus::UNAVAILABLE);
    EXPECT_FALSE(result.devices[0].availablePhysicalBytes);
    initializationResult = ZE_RESULT_SUCCESS;
    EXPECT_EQ(collect().devices[0].availablePhysicalBytes, 512);
}

TEST_F(GpuMemoryInfoTest, EnumerationFailureIsNotNoDevices) {
    driverResult = ZE_RESULT_ERROR_DEVICE_LOST;
    EXPECT_EQ(collect().status, GpuMemoryStatus::ERROR);
    driverResult = ZE_RESULT_SUCCESS;
    deviceResult = ZE_RESULT_ERROR_DEVICE_LOST;
    EXPECT_EQ(collect().status, GpuMemoryStatus::ERROR);
    EXPECT_EQ(collect({}).status, GpuMemoryStatus::AVAILABLE);
}

TEST_F(GpuMemoryInfoTest, UnknownZeroUnmatchedAndDuplicateUuidsAreUnavailable) {
    auto unknown = identity();
    unknown.uuid.reset();
    EXPECT_FALSE(collect({unknown}).devices[0].availablePhysicalBytes);
    unknown.uuid.emplace(std::array<uint8_t, 16>{});
    EXPECT_FALSE(collect({unknown}).devices[0].availablePhysicalBytes);
    EXPECT_FALSE(collect({identity(3)}).devices[0].availablePhysicalBytes);
    EXPECT_FALSE(collect({identity(), identity()}).devices[0].availablePhysicalBytes);
    physical.push_back(PhysicalGpu{});
    EXPECT_FALSE(collect().devices[0].availablePhysicalBytes);
}

TEST_F(GpuMemoryInfoTest, ReturnedDeviceCountIsRespected) {
    shrinkDeviceCount = true;
    EXPECT_FALSE(collect().devices[0].availablePhysicalBytes);
    shrinkDeviceCount = false;
    retryDeviceCount = true;
    EXPECT_EQ(collect().devices[0].availablePhysicalBytes, 512);
}

TEST_F(GpuMemoryInfoTest, UnsupportedCardDoesNotDiscardHealthyCard) {
    physical.push_back(PhysicalGpu{2});
    physical[0].modules[0].stateResult = ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
    const auto result = collect({identity(), identity(2)});
    EXPECT_EQ(result.status, GpuMemoryStatus::AVAILABLE);
    EXPECT_EQ(result.devices[0].status, GpuMemoryStatus::UNAVAILABLE);
    EXPECT_FALSE(result.devices[0].availablePhysicalBytes);
    EXPECT_EQ(result.devices[0].totalPhysicalBytes, 1024);
    EXPECT_EQ(result.devices[1].availablePhysicalBytes, 512);
}

TEST_F(GpuMemoryInfoTest, UnknownCapacityDoesNotInvalidateFreeReading) {
    physical[0].modules[0].size = 0;
    const auto result = collect();
    EXPECT_FALSE(result.devices[0].totalPhysicalBytes);
    EXPECT_EQ(result.devices[0].availablePhysicalBytes, 512);
}

TEST_F(GpuMemoryInfoTest, AggregatesTilesButNotOverlappingRootAndTilePools) {
    physical[0].modules = {MemoryModule{}, MemoryModule{}};
    for (auto& module : physical[0].modules)
        module.tile = true;
    auto result = collect();
    EXPECT_EQ(result.devices[0].totalPhysicalBytes, 2048);
    EXPECT_EQ(result.devices[0].availablePhysicalBytes, 1024);
    physical[0].modules[0].tile = false;
    EXPECT_FALSE(collect().devices[0].availablePhysicalBytes);
}

TEST_F(GpuMemoryInfoTest, DoesNotMultiplySharedMemoryPools) {
    physical[0].modules = {MemoryModule{}, MemoryModule{}};
    for (auto& module : physical[0].modules)
        module.location = ZES_MEM_LOC_SYSTEM;
    const auto result = collect({identity(1, GpuDeviceType::INTEGRATED)});
    EXPECT_TRUE(result.devices[0].sharedMemory);
    EXPECT_FALSE(result.devices[0].availablePhysicalBytes);
}

TEST_F(GpuMemoryInfoTest, IgnoresCachesAndSystemPoolsOnDiscreteGpu) {
    physical[0].modules = {MemoryModule{}, MemoryModule{}, MemoryModule{}};
    physical[0].modules[1].location = ZES_MEM_LOC_SYSTEM;
    physical[0].modules[2].type = ZES_MEM_TYPE_L1;
    EXPECT_EQ(collect().devices[0].totalPhysicalBytes, 1024);
}

TEST_F(GpuMemoryInfoTest, PartialModuleFailureDoesNotPublishPartialTotals) {
    physical[0].modules.push_back(MemoryModule{});
    physical[0].modules[1].propertiesResult = ZE_RESULT_ERROR_DEVICE_LOST;
    const auto result = collect();
    EXPECT_EQ(result.devices[0].status, GpuMemoryStatus::ERROR);
    EXPECT_FALSE(result.devices[0].availablePhysicalBytes);
    EXPECT_FALSE(result.devices[0].totalPhysicalBytes);
}

TEST_F(GpuMemoryInfoTest, RejectsInvalidAndOverflowingCounters) {
    physical[0].modules[0].free = 1025;
    EXPECT_EQ(collect().devices[0].status, GpuMemoryStatus::ERROR);
    physical[0].modules = {MemoryModule{}, MemoryModule{}};
    physical[0].modules[0].size = std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(collect().devices[0].status, GpuMemoryStatus::ERROR);
}

TEST_F(GpuMemoryInfoTest, NoMemoryModulesAreUnavailableRatherThanZeroFree) {
    physical[0].modules.clear();
    const auto result = collect();
    EXPECT_EQ(result.devices[0].status, GpuMemoryStatus::UNAVAILABLE);
    EXPECT_FALSE(result.devices[0].availablePhysicalBytes);
}

class GpuUtilizationTest : public GpuMemoryInfoTest {};

TEST_F(GpuUtilizationTest, CalculatesPercentageFromTwoIndependentSnapshots) {
    const auto previous = sample();
    physical[0].engines[0].activeTime = 350;
    physical[0].engines[0].timestamp = 2000;
    const auto current = sample();
    const auto result = ovms::calculateGpuUtilization(previous, current);
    ASSERT_EQ(result.devices.size(), 1);
    ASSERT_EQ(result.devices[0].engines.size(), 1);
    EXPECT_EQ(result.devices[0].deviceId, "GPU.0");
    ASSERT_TRUE(result.devices[0].engines[0].percent);
    EXPECT_DOUBLE_EQ(*result.devices[0].engines[0].percent, 25.0);
    EXPECT_EQ(previous.devices[0].engines[0].activeTime, 100);
}

TEST_F(GpuUtilizationTest, IdleAndFullyBusyAreValidReadings) {
    const auto previous = sample();
    physical[0].engines[0].timestamp += 1000;
    auto result = ovms::calculateGpuUtilization(previous, sample());
    ASSERT_TRUE(result.devices[0].engines[0].percent);
    EXPECT_DOUBLE_EQ(*result.devices[0].engines[0].percent, 0.0);
    physical[0].engines[0].activeTime += 1000;
    result = ovms::calculateGpuUtilization(previous, sample());
    ASSERT_TRUE(result.devices[0].engines[0].percent);
    EXPECT_DOUBLE_EQ(*result.devices[0].engines[0].percent, 100.0);
}

TEST_F(GpuUtilizationTest, ResetZeroIntervalAndInconsistentCountersAreUnavailable) {
    const auto previous = sample();
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    physical[0].engines[0].timestamp = 999;
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    physical[0].engines[0].timestamp = 2000;
    physical[0].engines[0].activeTime = 99;
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    physical[0].engines[0].activeTime = 1101;
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
}

TEST_F(GpuUtilizationTest, MatchesDeviceAndTileEngineIdentityNotOrder) {
    physical = {PhysicalGpu{1}, PhysicalGpu{2}};
    physical[0].engines = {EngineGroup{}, EngineGroup{}};
    physical[0].engines[1].type = ZES_ENGINE_GROUP_COPY_ALL;
    physical[0].engines[1].tile = true;
    physical[0].engines[1].tileId = 1;
    const auto previous = sample({identity(), identity(2, GpuDeviceType::INTEGRATED)});
    physical[0].engines[0].timestamp += 1000;
    physical[0].engines[0].activeTime += 250;
    physical[0].engines[1].timestamp += 1000;
    physical[0].engines[1].activeTime += 500;
    std::reverse(physical[0].engines.begin(), physical[0].engines.end());
    std::reverse(physical.begin(), physical.end());
    const auto result = ovms::calculateGpuUtilization(previous, sample({identity(2, GpuDeviceType::INTEGRATED), identity()}));
    ASSERT_EQ(result.devices.size(), 2);
    EXPECT_EQ(result.devices[1].deviceId, "GPU.0");
    ASSERT_EQ(result.devices[1].engines.size(), 2);
    ASSERT_TRUE(result.devices[1].engines[0].percent);
    EXPECT_DOUBLE_EQ(*result.devices[1].engines[0].percent, 50.0);
    EXPECT_TRUE(result.devices[1].engines[0].onSubdevice);
    EXPECT_EQ(result.devices[1].engines[0].subdeviceId, 1);
    ASSERT_TRUE(result.devices[1].engines[1].percent);
    EXPECT_DOUBLE_EQ(*result.devices[1].engines[1].percent, 25.0);
}

TEST_F(GpuUtilizationTest, MissingAndDuplicatePreviousEnginesAreUnavailable) {
    auto previous = sample();
    physical[0].engines[0].timestamp += 1000;
    previous.devices[0].engines.push_back(previous.devices[0].engines[0]);
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    previous.devices[0].engines.clear();
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    previous.devices.clear();
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
}

TEST_F(GpuUtilizationTest, MissingOrDuplicateDeviceIdentityIsUnavailable) {
    auto previous = sample();
    physical[0].engines[0].timestamp += 1000;
    previous.devices[0].uuid.reset();
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    previous = sample();
    previous.devices.push_back(previous.devices[0]);
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, sample()).devices[0].engines[0].percent);
    physical.push_back(PhysicalGpu{});
    EXPECT_EQ(sample().devices[0].status, GpuMemoryStatus::UNAVAILABLE);
}

TEST_F(GpuUtilizationTest, UnsupportedEngineCountersDoNotBreakMemoryReporting) {
    api.engineActivity = nullptr;
    EXPECT_EQ(sample().devices[0].status, GpuMemoryStatus::UNAVAILABLE);
    EXPECT_EQ(collect().devices[0].availablePhysicalBytes, 512);
    api.engineActivity = engineActivity;
    api.memoryState = nullptr;
    EXPECT_EQ(sample().devices[0].engines[0].status, GpuMemoryStatus::AVAILABLE);
}

TEST_F(GpuUtilizationTest, CounterFailureIsNotIdleAndOtherEnginesStillWork) {
    physical[0].engines.push_back(EngineGroup{ZES_ENGINE_GROUP_COPY_ALL});
    const auto previous = sample();
    physical[0].engines[0].activityResult = ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS;
    physical[0].engines[1].timestamp += 1000;
    physical[0].engines[1].activeTime += 500;
    const auto result = ovms::calculateGpuUtilization(previous, sample());
    EXPECT_FALSE(result.devices[0].engines[0].percent);
    ASSERT_TRUE(result.devices[0].engines[1].percent);
    EXPECT_DOUBLE_EQ(*result.devices[0].engines[1].percent, 50.0);
}

TEST_F(GpuUtilizationTest, IgnoresIndividualEnginesAndReportsNoAggregateGroups) {
    physical[0].engines[0].type = ZES_ENGINE_GROUP_COMPUTE_SINGLE;
    const auto result = sample();
    EXPECT_EQ(result.devices[0].status, GpuMemoryStatus::UNAVAILABLE);
    EXPECT_TRUE(result.devices[0].engines.empty());
    EXPECT_EQ(sample({}).status, GpuMemoryStatus::AVAILABLE);
}

TEST_F(GpuUtilizationTest, EnumerationAndPropertyFailuresAreExplicit) {
    physical[0].engineEnumerationResult = ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
    EXPECT_EQ(sample().devices[0].status, GpuMemoryStatus::UNAVAILABLE);
    physical[0].engineEnumerationResult = ZE_RESULT_SUCCESS;
    physical[0].engines[0].propertiesResult = ZE_RESULT_ERROR_DEVICE_LOST;
    EXPECT_EQ(sample().devices[0].status, GpuMemoryStatus::ERROR);
}

TEST_F(GpuUtilizationTest, PartialEnumerationFailurePreservesHealthyCardIntervals) {
    physical.push_back(PhysicalGpu{2});
    const auto previous = sample({identity(), identity(2)});
    physical[0].engines[0].timestamp += 1000;
    physical[0].engines[0].activeTime += 250;
    physical[1].propertiesResult = ZE_RESULT_ERROR_DEVICE_LOST;
    const auto current = sample({identity(), identity(2)});
    ASSERT_EQ(current.status, GpuMemoryStatus::ERROR);
    const auto result = ovms::calculateGpuUtilization(previous, current);
    ASSERT_TRUE(result.devices[0].engines[0].percent);
    EXPECT_DOUBLE_EQ(*result.devices[0].engines[0].percent, 25.0);
    EXPECT_EQ(result.devices[1].status, GpuMemoryStatus::UNAVAILABLE);
}

TEST_F(GpuUtilizationTest, AggregateCategoriesAndRootTileCountersAreNotSummed) {
    physical[0].engines = {
        EngineGroup{ZES_ENGINE_GROUP_ALL}, EngineGroup{ZES_ENGINE_GROUP_COMPUTE_ALL},
        EngineGroup{ZES_ENGINE_GROUP_RENDER_ALL}, EngineGroup{ZES_ENGINE_GROUP_COPY_ALL},
        EngineGroup{ZES_ENGINE_GROUP_MEDIA_ALL}, EngineGroup{ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL},
        EngineGroup{ZES_ENGINE_GROUP_COMPUTE_ALL, true, 1}};
    const auto previous = sample();
    ASSERT_EQ(previous.devices[0].engines.size(), 7);
    for (auto& engine : physical[0].engines) {
        engine.timestamp += 1000;
        engine.activeTime += 500;
    }
    const auto result = ovms::calculateGpuUtilization(previous, sample());
    ASSERT_EQ(result.devices[0].engines.size(), 7);
    for (const auto& engine : result.devices[0].engines) {
        ASSERT_TRUE(engine.percent);
        EXPECT_DOUBLE_EQ(*engine.percent, 50.0);
    }
}

TEST_F(GpuUtilizationTest, DuplicateCurrentEngineAndPreviousFailedCounterAreUnavailable) {
    const auto previous = sample();
    physical[0].engines.push_back(EngineGroup{});
    for (auto& engine : physical[0].engines)
        engine.timestamp += 1000;
    auto current = sample();
    EXPECT_FALSE(ovms::calculateGpuUtilization(previous, current).devices[0].engines[0].percent);
    physical[0].engines.resize(1);
    physical[0].engines[0].activityResult = ZE_RESULT_ERROR_DEVICE_LOST;
    const auto failed = sample();
    physical[0].engines[0].activityResult = ZE_RESULT_SUCCESS;
    current = sample();
    EXPECT_FALSE(ovms::calculateGpuUtilization(failed, current).devices[0].engines[0].percent);
}
}  // namespace

TEST(SystemInfo, GpuMemoryHardware) {
    if (!gpuTestsRequested())
        GTEST_SKIP() << "Set RUN_GPU_TESTS=1 to query real GPU hardware";
    ov::Core core;
    const auto result = ovms::getGpuMemoryInfo(core);
    std::cout << "GPU enumeration status: " << static_cast<int>(result.status) << " " << result.reason << std::endl;
    ASSERT_EQ(result.status, GpuMemoryStatus::AVAILABLE) << result.reason;
    ASSERT_FALSE(result.devices.empty()) << "No OpenVINO GPUs discovered";
    for (const auto& info : result.devices) {
        std::cout << info.deviceId << " " << info.deviceName << " shared=" << info.sharedMemory
                  << " status=" << static_cast<int>(info.status)
                  << " total=" << (info.totalPhysicalBytes ? std::to_string(*info.totalPhysicalBytes) : "unavailable")
                  << " free=" << (info.availablePhysicalBytes ? std::to_string(*info.availablePhysicalBytes) : "unavailable")
                  << " reason=" << info.reason << std::endl;
        if (info.availablePhysicalBytes && info.totalPhysicalBytes) {
            EXPECT_LE(*info.availablePhysicalBytes, *info.totalPhysicalBytes);
        }
        if (info.status != GpuMemoryStatus::AVAILABLE) {
            EXPECT_FALSE(info.availablePhysicalBytes);
        }
    }
    std::vector<std::thread> workers;
    for (size_t index = 0; index < 4; ++index) {
        workers.emplace_back([&core]() {
            const auto snapshot = ovms::getGpuMemoryInfo(core);
            EXPECT_EQ(snapshot.status, GpuMemoryStatus::AVAILABLE) << snapshot.reason;
        });
    }
    for (auto& worker : workers)
        worker.join();
    const auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    const auto output = std::make_shared<ov::op::v0::Result>(parameter);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{output}, ov::ParameterVector{parameter});
    for (const auto& info : result.devices) {
        auto compiled = core.compile_model(model, info.deviceId);
        auto request = compiled.create_infer_request();
        request.get_input_tensor().data<float>()[0] = 3.0f;
        request.infer();
        EXPECT_FLOAT_EQ(request.get_output_tensor().data<float>()[0], 3.0f);
        const auto snapshot = ovms::getGpuMemoryInfo(core);
        EXPECT_EQ(snapshot.status, GpuMemoryStatus::AVAILABLE) << snapshot.reason;
    }
}

TEST(SystemInfo, GpuUtilizationHardware) {
    if (!gpuTestsRequested())
        GTEST_SKIP() << "Set RUN_GPU_TESTS=1 to sample real GPU engines";
    ov::Core core;
    const auto previous = ovms::sampleGpuUtilization(core);
    ASSERT_EQ(previous.status, ovms::GpuTelemetryStatus::AVAILABLE) << previous.reason;
    ASSERT_FALSE(previous.devices.empty());
    const auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 64});
    const auto activation = std::make_shared<ov::op::v0::Relu>(parameter);
    const auto output = std::make_shared<ov::op::v0::Result>(activation);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{output}, ov::ParameterVector{parameter});
    for (const auto& device : previous.devices) {
        auto compiled = core.compile_model(model, device.deviceId);
        auto request = compiled.create_infer_request();
        auto tensor = request.get_input_tensor();
        std::fill_n(tensor.data<float>(), tensor.get_size(), -1.0f);
        for (size_t iteration = 0; iteration < 20; ++iteration)
            request.infer();
        EXPECT_FLOAT_EQ(request.get_output_tensor().data<float>()[0], 0.0f);
    }
    const auto current = ovms::sampleGpuUtilization(core);
    const auto result = ovms::calculateGpuUtilization(previous, current);
    ASSERT_EQ(result.status, ovms::GpuTelemetryStatus::AVAILABLE) << result.reason;
    for (const auto& device : result.devices) {
        std::cout << device.deviceId << " utilization status=" << static_cast<int>(device.status)
                  << " reason=" << device.reason << std::endl;
        for (const auto& engine : device.engines) {
            std::cout << "  engine=" << static_cast<int>(engine.type) << " tile=" << engine.onSubdevice
                      << ":" << engine.subdeviceId << " percent=" << (engine.percent ? std::to_string(*engine.percent) : "unavailable")
                      << " reason=" << engine.reason << std::endl;
            if (engine.percent) {
                EXPECT_GE(*engine.percent, 0.0);
                EXPECT_LE(*engine.percent, 100.0);
            } else {
                EXPECT_NE(engine.status, ovms::GpuTelemetryStatus::AVAILABLE);
            }
        }
    }
}
