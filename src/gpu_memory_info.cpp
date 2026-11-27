//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "gpu_memory_info_internal.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <utility>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ovms {
namespace {
using Uuid = std::array<uint8_t, 16>;

bool validUuid(const Uuid& uuid) {
    return std::any_of(uuid.begin(), uuid.end(), [](uint8_t value) { return value != 0; });
}

GpuMemoryStatus failureStatus(ze_result_t result) {
    return result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE ||
                   result == ZE_RESULT_ERROR_UNINITIALIZED ||
                   result == ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS ||
                   result == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE
               ? GpuMemoryStatus::UNAVAILABLE
               : GpuMemoryStatus::ERROR;
}

std::string failureReason(const char* operation, ze_result_t result) {
    return std::string(operation) + " failed (Level Zero result " + std::to_string(static_cast<uint32_t>(result)) + ")";
}

template <typename Handle, typename Enumerate>
ze_result_t enumerate(std::vector<Handle>& handles, Enumerate&& query) {
    for (size_t attempt = 0; attempt < 2; ++attempt) {
        uint32_t count = 0;
        ze_result_t result = query(&count, nullptr);
        if (result != ZE_RESULT_SUCCESS || count == 0)
            return result;
        handles.resize(count);
        result = query(&count, handles.data());
        if (result == ZE_RESULT_SUCCESS && count <= handles.size()) {
            handles.resize(count);
            return result;
        }
        handles.clear();
        if (result != ZE_RESULT_SUCCESS && result != ZE_RESULT_ERROR_INVALID_SIZE)
            return result;
    }
    return ZE_RESULT_ERROR_INVALID_SIZE;
}

struct SysmanLoader {
    std::shared_ptr<void> library;
    detail::SysmanApi api;

    SysmanLoader() {
#ifdef _WIN32
        HMODULE handle = LoadLibraryExA("ze_loader.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!handle)
            return;
        library = std::shared_ptr<void>(handle, [](void* module) { FreeLibrary(static_cast<HMODULE>(module)); });
#else
        void* handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle)
            return;
        library = std::shared_ptr<void>(handle, [](void* module) { dlclose(module); });
#endif
        api.init = symbol<decltype(api.init)>("zesInit");
        api.driverGet = symbol<decltype(api.driverGet)>("zesDriverGet");
        api.deviceGet = symbol<decltype(api.deviceGet)>("zesDeviceGet");
        api.deviceProperties = symbol<decltype(api.deviceProperties)>("zesDeviceGetProperties");
        api.memoryModules = symbol<decltype(api.memoryModules)>("zesDeviceEnumMemoryModules");
        api.memoryProperties = symbol<decltype(api.memoryProperties)>("zesMemoryGetProperties");
        api.memoryState = symbol<decltype(api.memoryState)>("zesMemoryGetState");
        api.engineGroups = symbol<decltype(api.engineGroups)>("zesDeviceEnumEngineGroups");
        api.engineProperties = symbol<decltype(api.engineProperties)>("zesEngineGetProperties");
        api.engineActivity = symbol<decltype(api.engineActivity)>("zesEngineGetActivity");
    }

    template <typename Function>
    Function symbol(const char* name) const {
#ifdef _WIN32
        return reinterpret_cast<Function>(GetProcAddress(static_cast<HMODULE>(library.get()), name));
#else
        return reinterpret_cast<Function>(dlsym(library.get(), name));
#endif
    }
};

void queryMemory(GpuMemoryInfo& info, zes_device_handle_t device, const detail::SysmanApi& api) {
    if (!api.memoryModules || !api.memoryProperties || !api.memoryState) {
        info.reason = "Level Zero memory telemetry symbols are unavailable";
        return;
    }
    std::vector<zes_mem_handle_t> modules;
    ze_result_t result = enumerate(modules, [&](uint32_t* count, zes_mem_handle_t* handles) {
        return api.memoryModules(device, count, handles);
    });
    if (result != ZE_RESULT_SUCCESS) {
        info.status = failureStatus(result);
        info.reason = failureReason("zesDeviceEnumMemoryModules", result);
        return;
    }
    uint64_t total = 0;
    uint64_t free = 0;
    bool capacityKnown = true;
    bool freeKnown = true;
    bool rootMemory = false;
    bool tileMemory = false;
    size_t selectedModules = 0;
    for (const auto& module : modules) {
        zes_mem_properties_t properties{};
        properties.stype = ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
        result = api.memoryProperties(module, &properties);
        if (result != ZE_RESULT_SUCCESS) {
            info.status = failureStatus(result);
            info.reason = failureReason("zesMemoryGetProperties", result);
            return;
        }
        if (properties.type == ZES_MEM_TYPE_SRAM || properties.type == ZES_MEM_TYPE_L1 ||
            properties.type == ZES_MEM_TYPE_L3 || properties.type == ZES_MEM_TYPE_GRF || properties.type == ZES_MEM_TYPE_SLM)
            continue;
        if (properties.location != (info.sharedMemory ? ZES_MEM_LOC_SYSTEM : ZES_MEM_LOC_DEVICE))
            continue;
        rootMemory |= !properties.onSubdevice;
        tileMemory |= properties.onSubdevice != 0;
        ++selectedModules;
        if ((rootMemory && tileMemory) || (info.sharedMemory && selectedModules > 1)) {
            info.reason = "Memory pools may overlap; no aggregate is available";
            return;
        }
        zes_mem_state_t state{};
        state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
        result = api.memoryState(module, &state);
        if (result != ZE_RESULT_SUCCESS) {
            freeKnown = false;
            info.status = failureStatus(result);
            info.reason = failureReason("zesMemoryGetState", result);
        }
        if ((result == ZE_RESULT_SUCCESS && properties.physicalSize && state.free > properties.physicalSize) ||
            properties.physicalSize > std::numeric_limits<uint64_t>::max() - total ||
            (result == ZE_RESULT_SUCCESS && state.free > std::numeric_limits<uint64_t>::max() - free)) {
            info.status = GpuMemoryStatus::ERROR;
            info.reason = "Invalid or overflowing physical memory counters";
            return;
        }
        capacityKnown &= properties.physicalSize != 0;
        total += properties.physicalSize;
        if (result == ZE_RESULT_SUCCESS)
            free += state.free;
    }
    if (selectedModules == 0) {
        info.reason = "No supported global memory modules reported by Sysman";
        return;
    }
    if (capacityKnown)
        info.totalPhysicalBytes = total;
    if (freeKnown) {
        info.availablePhysicalBytes = free;
        info.status = GpuMemoryStatus::AVAILABLE;
        info.reason = capacityKnown ? "" : "Physical capacity is unknown";
    }
}
std::optional<GpuEngineType> engineType(zes_engine_group_t type) {
    switch (type) {
    case ZES_ENGINE_GROUP_ALL:
        return GpuEngineType::ALL;
    case ZES_ENGINE_GROUP_COMPUTE_ALL:
        return GpuEngineType::COMPUTE;
    case ZES_ENGINE_GROUP_RENDER_ALL:
        return GpuEngineType::RENDER;
    case ZES_ENGINE_GROUP_COPY_ALL:
        return GpuEngineType::COPY;
    case ZES_ENGINE_GROUP_MEDIA_ALL:
        return GpuEngineType::MEDIA;
    case ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL:
        return GpuEngineType::RENDER_COMPUTE;
    default:
        return std::nullopt;
    }
}

void queryEngines(GpuUtilizationDeviceSample& info, zes_device_handle_t device, const detail::SysmanApi& api) {
    if (!api.engineGroups || !api.engineProperties || !api.engineActivity) {
        info.reason = "Level Zero engine telemetry symbols are unavailable";
        return;
    }
    std::vector<zes_engine_handle_t> handles;
    ze_result_t result = enumerate(handles, [&](uint32_t* count, zes_engine_handle_t* engines) {
        return api.engineGroups(device, count, engines);
    });
    if (result != ZE_RESULT_SUCCESS) {
        info.status = failureStatus(result);
        info.reason = failureReason("zesDeviceEnumEngineGroups", result);
        return;
    }
    if (handles.empty()) {
        info.reason = "Sysman reported no engine counter handles";
        return;
    }
    for (const auto& handle : handles) {
        zes_engine_properties_t properties{};
        properties.stype = ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES;
        result = api.engineProperties(handle, &properties);
        if (result != ZE_RESULT_SUCCESS) {
            info.status = failureStatus(result);
            info.reason = failureReason("zesEngineGetProperties", result);
            continue;
        }
        const auto type = engineType(properties.type);
        if (!type) {
            info.reason += " Unsupported Sysman engine group " + std::to_string(static_cast<uint32_t>(properties.type));
            continue;
        }
        GpuEngineSample sample;
        sample.type = *type;
        sample.onSubdevice = properties.onSubdevice != 0;
        sample.subdeviceId = sample.onSubdevice ? properties.subdeviceId : 0;
        zes_engine_stats_t counters{};
        result = api.engineActivity(handle, &counters);
        if (result == ZE_RESULT_SUCCESS) {
            sample.activeTime = counters.activeTime;
            sample.timestamp = counters.timestamp;
            sample.status = GpuTelemetryStatus::AVAILABLE;
        } else {
            sample.status = failureStatus(result);
            sample.reason = failureReason("zesEngineGetActivity", result);
        }
        info.engines.push_back(std::move(sample));
    }
    if (info.status != GpuTelemetryStatus::ERROR) {
        info.status = info.engines.empty() ? GpuTelemetryStatus::UNAVAILABLE : GpuTelemetryStatus::AVAILABLE;
        if (info.engines.empty() && info.reason.empty())
            info.reason = "No supported aggregate engine groups reported by Sysman";
    }
}

template <typename Result, typename Query>
Result collectGpuInfo(Result output, const detail::SysmanApi& api, Query&& query, bool telemetrySupported) {
    if (output.devices.empty()) {
        output.status = GpuMemoryStatus::AVAILABLE;
        return output;
    }
    const auto fail = [&](GpuMemoryStatus status, const std::string& reason) {
        output.status = status;
        output.reason = reason;
        for (auto& info : output.devices) {
            info.status = status;
            info.reason = reason;
        }
    };
    if (!api.init || !api.driverGet || !api.deviceGet || !api.deviceProperties || !telemetrySupported) {
        fail(GpuMemoryStatus::UNAVAILABLE, "Level Zero loader or standalone Sysman symbols are unavailable");
        return output;
    }
    ze_result_t result = api.init(0);
    if (result != ZE_RESULT_SUCCESS) {
        fail(failureStatus(result), failureReason("zesInit", result));
        return output;
    }
    std::vector<zes_driver_handle_t> drivers;
    result = enumerate(drivers, [&](uint32_t* count, zes_driver_handle_t* handles) { return api.driverGet(count, handles); });
    if (result != ZE_RESULT_SUCCESS) {
        fail(failureStatus(result), failureReason("zesDriverGet", result));
        return output;
    }
    std::vector<std::pair<Uuid, zes_device_handle_t>> identities;
    for (const auto& driver : drivers) {
        std::vector<zes_device_handle_t> handles;
        result = enumerate(handles, [&](uint32_t* count, zes_device_handle_t* devices) { return api.deviceGet(driver, count, devices); });
        if (result != ZE_RESULT_SUCCESS) {
            output.status = failureStatus(result);
            output.reason = failureReason("zesDeviceGet", result);
            continue;
        }
        for (const auto& handle : handles) {
            zes_device_ext_properties_t extended{};
            extended.stype = ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES;
            zes_device_properties_t properties{};
            properties.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            properties.pNext = &extended;
            properties.core.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            result = api.deviceProperties(handle, &properties);
            if (result != ZE_RESULT_SUCCESS) {
                output.status = failureStatus(result);
                output.reason = failureReason("zesDeviceGetProperties", result);
                continue;
            }
            Uuid uuid{};
            std::copy_n(extended.uuid.id, uuid.size(), uuid.begin());
            if (extended.type == ZES_DEVICE_TYPE_GPU && !(extended.flags & ZES_DEVICE_PROPERTY_FLAG_SUBDEVICE) && validUuid(uuid))
                identities.emplace_back(uuid, handle);
        }
    }
    if (output.reason.empty())
        output.status = GpuMemoryStatus::AVAILABLE;
    for (auto& info : output.devices) {
        if (!info.uuid || !validUuid(*info.uuid) || info.deviceType == GpuDeviceType::UNKNOWN) {
            info.reason = "OpenVINO GPU identity or device type is unavailable";
            continue;
        }
        const size_t matches = std::count_if(identities.begin(), identities.end(), [&](const auto& identity) { return identity.first == *info.uuid; });
        const size_t aliases = std::count_if(output.devices.begin(), output.devices.end(), [&](const auto& candidate) { return candidate.uuid == info.uuid; });
        if (matches != 1 || aliases != 1) {
            info.reason = "No unique physical Sysman GPU matches the OpenVINO UUID";
            continue;
        }
        const auto identity = std::find_if(identities.begin(), identities.end(), [&](const auto& candidate) { return candidate.first == *info.uuid; });
        query(info, identity->second, api);
    }
    return output;
}

template <typename Info>
std::vector<Info> openvinoGpuIdentities(const ov::Core& core) {
    std::vector<Info> devices;
    for (const auto& device : core.get_available_devices()) {
        if (device != "GPU" && device.rfind("GPU.", 0) != 0)
            continue;
        if (device.rfind("GPU.", 0) == 0 && device.find('.', 4) != std::string::npos)
            continue;
        Info info;
        info.deviceId = device;
        try {
            info.deviceName = core.get_property(device, ov::device::full_name);
        } catch (const std::exception&) {
            info.deviceName = device;
        }
        try {
            const auto uuid = core.get_property(device, ov::device::uuid);
            info.uuid = uuid.uuid;
            const auto type = core.get_property(device, ov::device::type);
            if (type == ov::device::Type::INTEGRATED)
                info.deviceType = GpuDeviceType::INTEGRATED;
            else if (type == ov::device::Type::DISCRETE)
                info.deviceType = GpuDeviceType::DISCRETE;
        } catch (const std::exception&) {
            info.reason = "OpenVINO GPU identity properties are unavailable";
        }
        devices.push_back(std::move(info));
    }
    return devices;
}

const detail::SysmanApi& sysmanApi() {
    static const SysmanLoader loader;
    return loader.api;
}

template <typename First, typename Second>
bool sameEngine(const First& first, const Second& second) {
    return first.type == second.type && first.onSubdevice == second.onSubdevice &&
           (!first.onSubdevice || first.subdeviceId == second.subdeviceId);
}
}  // namespace

namespace detail {
GpuMemoryInfoResult collectGpuMemoryInfo(std::vector<GpuMemoryInfo> devices, const SysmanApi& api) {
    GpuMemoryInfoResult output;
    output.devices = std::move(devices);
    for (auto& info : output.devices) {
        info.totalPhysicalBytes.reset();
        info.availablePhysicalBytes.reset();
        info.status = GpuMemoryStatus::UNAVAILABLE;
        info.reason.clear();
        info.sharedMemory = info.deviceType == GpuDeviceType::INTEGRATED;
    }
    return collectGpuInfo(std::move(output), api, queryMemory, api.memoryModules && api.memoryProperties && api.memoryState);
}

GpuUtilizationSample collectGpuUtilizationSample(std::vector<GpuUtilizationDeviceSample> devices, const SysmanApi& api) {
    GpuUtilizationSample output;
    output.devices = std::move(devices);
    for (auto& info : output.devices) {
        info.engines.clear();
        info.status = GpuTelemetryStatus::UNAVAILABLE;
        info.reason.clear();
    }
    return collectGpuInfo(std::move(output), api, queryEngines, api.engineGroups && api.engineProperties && api.engineActivity);
}
}  // namespace detail

GpuMemoryInfoResult getGpuMemoryInfo(const ov::Core& core) {
    try {
        return detail::collectGpuMemoryInfo(openvinoGpuIdentities<GpuMemoryInfo>(core), sysmanApi());
    } catch (const std::exception& error) {
        return {GpuMemoryStatus::ERROR, error.what(), {}};
    }
}

GpuUtilizationSample sampleGpuUtilization(const ov::Core& core) {
    try {
        return detail::collectGpuUtilizationSample(openvinoGpuIdentities<GpuUtilizationDeviceSample>(core), sysmanApi());
    } catch (const std::exception& error) {
        return {GpuTelemetryStatus::ERROR, error.what(), {}};
    }
}

GpuUtilizationResult calculateGpuUtilization(const GpuUtilizationSample& previous, const GpuUtilizationSample& current) {
    GpuUtilizationResult output;
    output.status = current.status;
    output.reason = current.reason;
    for (const auto& device : current.devices) {
        GpuDeviceUtilization info;
        info.deviceId = device.deviceId;
        info.deviceName = device.deviceName;
        info.uuid = device.uuid;
        info.status = device.status;
        info.reason = device.reason;
        const auto matches = [&](const auto& candidate) { return device.uuid && validUuid(*device.uuid) && candidate.uuid == device.uuid; };
        const auto earlier = std::find_if(previous.devices.begin(), previous.devices.end(), matches);
        const bool validDevice = earlier != previous.devices.end() &&
                                 std::count_if(previous.devices.begin(), previous.devices.end(), matches) == 1 &&
                                 std::count_if(current.devices.begin(), current.devices.end(), matches) == 1;
        for (const auto& engine : device.engines) {
            GpuEngineUtilization utilization;
            utilization.type = engine.type;
            utilization.onSubdevice = engine.onSubdevice;
            utilization.subdeviceId = engine.subdeviceId;
            utilization.status = engine.status;
            utilization.reason = engine.reason;
            if (device.status != GpuTelemetryStatus::AVAILABLE || engine.status != GpuTelemetryStatus::AVAILABLE) {
                utilization.status = engine.status == GpuTelemetryStatus::ERROR ? GpuTelemetryStatus::ERROR : GpuTelemetryStatus::UNAVAILABLE;
                if (utilization.reason.empty())
                    utilization.reason = "Current GPU or engine sample is unavailable";
            } else if (!validDevice || earlier->status != GpuTelemetryStatus::AVAILABLE) {
                utilization.status = GpuTelemetryStatus::UNAVAILABLE;
                utilization.reason = "No unique valid previous GPU sample";
            } else {
                const auto matchingEngine = [&](const auto& candidate) { return sameEngine(candidate, engine); };
                const auto oldEngine = std::find_if(earlier->engines.begin(), earlier->engines.end(), matchingEngine);
                if (oldEngine == earlier->engines.end() ||
                    std::count_if(earlier->engines.begin(), earlier->engines.end(), matchingEngine) != 1 ||
                    std::count_if(device.engines.begin(), device.engines.end(), matchingEngine) != 1 ||
                    oldEngine->status != GpuTelemetryStatus::AVAILABLE) {
                    utilization.status = GpuTelemetryStatus::UNAVAILABLE;
                    utilization.reason = "No unique valid previous engine sample";
                } else if (engine.timestamp <= oldEngine->timestamp || engine.activeTime < oldEngine->activeTime ||
                           engine.activeTime - oldEngine->activeTime > engine.timestamp - oldEngine->timestamp) {
                    utilization.status = GpuTelemetryStatus::UNAVAILABLE;
                    utilization.reason = "Zero interval, reset or inconsistent engine counters";
                } else {
                    utilization.percent = 100.0 * static_cast<double>(engine.activeTime - oldEngine->activeTime) /
                                          static_cast<double>(engine.timestamp - oldEngine->timestamp);
                    utilization.status = GpuTelemetryStatus::AVAILABLE;
                    utilization.reason.clear();
                }
            }
            info.engines.push_back(std::move(utilization));
        }
        if (info.status == GpuTelemetryStatus::AVAILABLE &&
            std::none_of(info.engines.begin(), info.engines.end(), [](const auto& engine) { return engine.percent.has_value(); })) {
            info.status = GpuTelemetryStatus::UNAVAILABLE;
            info.reason = "No valid engine intervals available";
        }
        output.devices.push_back(std::move(info));
    }
    return output;
}
}  // namespace ovms
