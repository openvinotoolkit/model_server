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
#include "systeminfo.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#include <sys/resource.h>
#endif

#include "logging.hpp"
#include "status.hpp"

namespace ovms {
uint16_t getCoreCount() {
    uint16_t detectedCoreCount = static_cast<uint16_t>(std::thread::hardware_concurrency());
#ifdef __linux__
    if (isRunningInDocker()) {
        const uint16_t affinityCount = getCpuAffinityCount();
        const uint16_t quotaCount = getDockerCpuQuota();
        if (quotaCount > 0) {
            detectedCoreCount = std::min(affinityCount, quotaCount);
        } else {
            detectedCoreCount = affinityCount;
        }
    }
#endif
    return detectedCoreCount;
}

uint64_t getMaxOpenFilesLimit() {
#ifdef __linux__
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0) {
        return limit.rlim_cur;
    }
#endif
    return std::numeric_limits<uint64_t>::max();
}

#ifdef __linux__

bool isRunningInDocker() {
    // Check for /.dockerenv file
    std::ifstream dockerenv("/.dockerenv");
    if (dockerenv.good()) {
        return true;
    }

    // Check /proc/self/cgroup for docker references
    std::ifstream cgroup("/proc/self/cgroup");
    if (cgroup.is_open()) {
        std::string line;
        while (std::getline(cgroup, line)) {
            if (line.find("docker") != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

uint16_t getCpuAffinityCount() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(mask), &mask) == -1) {
        return std::thread::hardware_concurrency();
    }

    int cpu_count = CPU_COUNT(&mask);
    return static_cast<uint16_t>(cpu_count);
}

uint16_t getDockerCpuQuota() {
    // Try cgroup v2 cpu.max (format: "quota period")
    std::ifstream cpu_max_v2("/sys/fs/cgroup/cpu.max");
    if (cpu_max_v2.is_open()) {
        std::string line;
        if (std::getline(cpu_max_v2, line)) {
            std::istringstream iss(line);
            std::string quota_str, period_str;
            if (iss >> quota_str >> period_str) {
                try {
                    uint64_t quota = std::stoull(quota_str);
                    uint64_t period = std::stoull(period_str);
                    if (quota > 0 && period > 0 && quota != ULLONG_MAX) {
                        uint16_t cpu_count = static_cast<uint16_t>((quota + period - 1) / period);
                        return cpu_count;
                    }
                } catch (const std::exception&) {
                    // Parsing failed, continue
                }
            }
        }
    }

    // Try cgroup v1 cpu.cfs_quota_us and cpu.cfs_period_us
    std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");

    if (quota_file.is_open() && period_file.is_open()) {
        std::string quota_str, period_str;
        if (std::getline(quota_file, quota_str) && std::getline(period_file, period_str)) {
            // Trim whitespace
            quota_str.erase(quota_str.find_last_not_of(" \n\r\t") + 1);
            period_str.erase(period_str.find_last_not_of(" \n\r\t") + 1);
            try {
                uint64_t quota = std::stoull(quota_str);
                uint64_t period = std::stoull(period_str);
                if (quota > 0 && period > 0) {
                    uint16_t cpu_count = static_cast<uint16_t>((quota + period - 1) / period);
                    return cpu_count;
                }
            } catch (const std::exception&) {
                // Parsing failed, continue
            }
        }
    }

    return 0;  // No quota set
}

#endif  // __linux__

}  // namespace ovms
