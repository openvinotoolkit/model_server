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

#include <fstream>
#include <sstream>
#include <string>

#include "logging.hpp"
#include "status.hpp"
#include "systeminfo_impl.hpp"

namespace ovms {
const char* CPUSET_FILENAME = "/sys/fs/cgroup/cpuset/cpuset.cpus";
uint16_t getCPUCountLimit() {
    std::ifstream fs;
    auto status = getCPUSetFile(fs, CPUSET_FILENAME);
    if (!status.ok()) {
        return 1;
    }
    std::string cpusets;
    fs >> cpusets;
    return getCPUCountLimitImpl(cpusets);
}
}  // namespace ovms
