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
#pragma once
#include <stdint.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace ov {
class Core;
}

namespace ovms {
/**
 * @brief Internal Intel GPU telemetry shared by the memory and utilization APIs.
 *
 * Linux and Windows require a Level Zero loader with standalone Sysman entry
 * points (libze_loader.so.1 or ze_loader.dll) and a compatible Intel GPU driver.
 * On Windows the loader must be in a trusted DLL search location. Linux
 * containers also need GPU device access and suitable permissions. Driver and
 * environment support determines which readings are available.
 *
 * GPU.N identifiers come from the supplied OpenVINO Core. Root physical devices
 * are matched to Sysman by a nonzero, unique UUID, not enumeration order; tile
 * aliases are not returned as extra cards. Missing or ambiguous identities,
 * unsupported telemetry and insufficient permissions are reported explicitly.
 * No host-memory or OpenVINO allocation-statistics estimates are substituted,
 * no environment variables are changed and no background polling is started.
 *
 * In the OVMS build container, deterministic tests run with:
 *   bazel test --config=mp_on --test_output=all //src:systeminfo_test
 * Enable the real GPU tests with --test_env=RUN_GPU_TESTS=1, or use
 *   make run_unit_tests RUN_GPU_TESTS=1
 * The hardware tests also check concurrent memory queries and GPU inference.
 *
 * Linux testing on two Intel Data Center GPU Flex 140 cards confirmed free
 * memory and UUID matching, but physical capacity and engine handles were not
 * reported by that driver configuration. Linux iGPU, Windows GPU and numerical
 * utilization hardware validation remain pending.
 */
enum class GpuMemoryStatus { AVAILABLE,
    UNAVAILABLE,
    ERROR };
enum class GpuDeviceType { UNKNOWN,
    INTEGRATED,
    DISCRETE };

/** @brief A physical GPU's memory snapshot; absent byte counts are not zero readings. */
struct GpuMemoryInfo {
    std::string deviceId;
    std::string deviceName;
    std::optional<std::array<uint8_t, 16>> uuid;
    GpuDeviceType deviceType = GpuDeviceType::UNKNOWN;
    bool sharedMemory = false;
    std::optional<uint64_t> totalPhysicalBytes;
    std::optional<uint64_t> availablePhysicalBytes;
    GpuMemoryStatus status = GpuMemoryStatus::UNAVAILABLE;
    std::string reason;
};

/** @brief Enumeration status is independent of individual device telemetry failures. */
struct GpuMemoryInfoResult {
    GpuMemoryStatus status = GpuMemoryStatus::UNAVAILABLE;
    std::string reason;
    std::vector<GpuMemoryInfo> devices;
};

/**
 * @brief Query Intel physical GPU memory using Level Zero Sysman on Linux and Windows.
 * @param core Existing OpenVINO Core defining the GPU.N device identifiers.
 * @return Per-card snapshots in bytes, including explicit unavailable/error states.
 * Free physical memory is not an allocation guarantee or an OS memory budget.
 * Integrated GPU memory is shared with the host, not exclusive GPU VRAM.
 * Measurements across modules/cards are not atomic. Unsupported identity or
 * telemetry is never estimated from OpenVINO allocations or host RAM.
 * Physical capacity and free bytes can be absent independently; an absent
 * value is different from zero. Ambiguous overlapping pools are not summed.
 * Sysman's deprecated allocatable-size field is not used as physical capacity.
 * Successful enumeration with no GPUs returns AVAILABLE and an empty vector.
 */
GpuMemoryInfoResult getGpuMemoryInfo(const ov::Core& core);

using GpuTelemetryStatus = GpuMemoryStatus;

enum class GpuEngineType { ALL,
    COMPUTE,
    RENDER,
    COPY,
    MEDIA,
    RENDER_COMPUTE };

/** @brief Raw aggregate engine counters, valid only when status is AVAILABLE. */
struct GpuEngineSample {
    GpuEngineType type = GpuEngineType::ALL;
    bool onSubdevice = false;
    uint32_t subdeviceId = 0;
    uint64_t activeTime = 0;
    uint64_t timestamp = 0;
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
};

/** @brief Engine samples for one physical OpenVINO GPU.N device. */
struct GpuUtilizationDeviceSample {
    std::string deviceId;
    std::string deviceName;
    std::optional<std::array<uint8_t, 16>> uuid;
    GpuDeviceType deviceType = GpuDeviceType::UNKNOWN;
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
    std::vector<GpuEngineSample> engines;
};

struct GpuUtilizationSample {
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
    std::vector<GpuUtilizationDeviceSample> devices;
};

/** @brief Interval utilization for one aggregate engine, not additive across engines. */
struct GpuEngineUtilization {
    GpuEngineType type = GpuEngineType::ALL;
    bool onSubdevice = false;
    uint32_t subdeviceId = 0;
    std::optional<double> percent;
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
};

struct GpuDeviceUtilization {
    std::string deviceId;
    std::string deviceName;
    std::optional<std::array<uint8_t, 16>> uuid;
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
    std::vector<GpuEngineUtilization> engines;
};

struct GpuUtilizationResult {
    GpuTelemetryStatus status = GpuTelemetryStatus::UNAVAILABLE;
    std::string reason;
    std::vector<GpuDeviceUtilization> devices;
};

/**
 * @brief Sample Level Zero Sysman aggregate engine counters on Linux and Windows.
 * @param core Existing OpenVINO Core defining the GPU.N device identifiers.
 * @return Per-card counters for supported all/compute/render/copy/media groups.
 * Callers choose the sampling interval, e.g. from a periodic monitoring loop.
 * This method never waits for an interval or starts a polling thread. Root and
 * tile groups are retained separately and individual engines are not substituted
 * for aggregate groups. Enumeration status and per-engine availability are
 * separate; a partial failure does not discard another GPU's valid intervals.
 */
GpuUtilizationSample sampleGpuUtilization(const ov::Core& core);

/**
 * @brief Calculate utilization as 100 * delta(activeTime) / delta(timestamp).
 * @param previous Earlier sample from this process and the same driver session.
 * @param current Later sample, defining the returned devices and engines.
 * @return Percentages matched by GPU UUID and aggregate engine type/tile identity.
 * Failed, missing, ambiguous, reset, inconsistent or zero-interval counters
 * yield no percentage. A valid zero is an idle engine, not unavailable data.
 * Engines run concurrently; their percentages must not be added together. For
 * example, with a previous sample collected in the same process and driver session:
 * @code
 * const auto current = sampleGpuUtilization(core);
 * const auto utilization = calculateGpuUtilization(previous, current);
 * @endcode
 */
GpuUtilizationResult calculateGpuUtilization(const GpuUtilizationSample& previous, const GpuUtilizationSample& current);

/**
 * @brief Get cpu core count on system. This can be limited by the container environment. In case of failure reading system constraints it will return total number of available cores. If it won't work the function will return 1
 * @return uint16_t Available number of cores in the system
 */
uint16_t getCoreCount();
uint64_t getMaxOpenFilesLimit();
#ifdef __linux__
bool isRunningInDocker();
/**
 * @brief Get number of CPUs available via CPU affinity mask
 * @return uint16_t Number of CPUs in the affinity mask, or total hardware concurrency if affinity is not set
 */
uint16_t getCpuAffinityCount();
/**
 * @brief Get CPU limit from cgroup (docker run --cpus constraint)
 * @return uint16_t Number of CPUs allowed by quota, or 0 if no quota is set
 */
uint16_t getDockerCpuQuota();
/**
 * @brief Get number of physical (non-hyperthreaded) CPU cores
 * @return uint16_t Number of physical cores, or hardware_concurrency if detection fails
 */
uint16_t getPhysicalCoresPerSocket();

/**
 * @brief Get number of CPU sockets
 * @return uint16_t Number of CPU sockets, or 1 if detection fails
 */
uint16_t getSocketsCount();

#endif
}  // namespace ovms
