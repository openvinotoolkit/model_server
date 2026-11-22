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
#pragma once

#include <vector>

#include <level_zero/zes_api.h>

#include "systeminfo.hpp"

namespace ovms {
namespace detail {
struct SysmanApi {
    decltype(&zesInit) init = nullptr;
    decltype(&zesDriverGet) driverGet = nullptr;
    decltype(&zesDeviceGet) deviceGet = nullptr;
    decltype(&zesDeviceGetProperties) deviceProperties = nullptr;
    decltype(&zesDeviceEnumMemoryModules) memoryModules = nullptr;
    decltype(&zesMemoryGetProperties) memoryProperties = nullptr;
    decltype(&zesMemoryGetState) memoryState = nullptr;
    decltype(&zesDeviceEnumEngineGroups) engineGroups = nullptr;
    decltype(&zesEngineGetProperties) engineProperties = nullptr;
    decltype(&zesEngineGetActivity) engineActivity = nullptr;
};

GpuMemoryInfoResult collectGpuMemoryInfo(std::vector<GpuMemoryInfo> devices, const SysmanApi& api);
GpuUtilizationSample collectGpuUtilizationSample(std::vector<GpuUtilizationDeviceSample> devices, const SysmanApi& api);
}  // namespace detail
}  // namespace ovms
